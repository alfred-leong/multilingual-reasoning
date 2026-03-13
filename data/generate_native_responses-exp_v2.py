#!/usr/bin/env python3
import argparse
import json
import logging
import math
import multiprocessing as mp
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import openai
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.translate_dpo_prompts import get_translate_dpo_prompts

# Regex for extracting A/B/C/D
_ANSWER_RE = re.compile(r"\b([A-D])\b")
# Regex for MGSM boxed answer
_BOXED_RE = re.compile(r"\\boxed{([^}]*)}")

LANGUAGE_CODES = {
    "bn": "BN_BD",
    "ja": "JA_JP",
    "sw": "SW_KE"
}

def _setup_logger(name: str, log_dir: Path, filename: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_dir / filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(sh)

    return logger

def _extract_mmmlu_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else ""

def _extract_mgsm_answer(text: str) -> str:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""

def _format_mmmlu_question(row: dict) -> str:
    q = row["Question"]  # CAIS MMLU has 'question', OpenAI MMMLU has 'Question'
    choices = "\n".join(
        f"{letter}. {row[letter]}" for letter in ("A", "B", "C", "D")
    )
    return f"{q}\n\n{choices}"

def build_messages(system_prompt: str, question: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

def generate_response_api(
    client: openai.OpenAI,
    model_name: str,
    messages: list,
    thinking_prefix: str,
    max_tokens: int = 31000,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Tuple[str, str]:
    extra = {"chat_template_kwargs": {"enable_thinking": True}}

    if thinking_prefix:
        messages = list(messages) + [
            {"role": "assistant", "content": f"<think>{thinking_prefix}"}
        ]
        extra["continue_final_message"] = True
        extra["add_generation_prompt"] = False

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra,
    )
    msg = response.choices[0].message

    generated_thinking = (getattr(msg, "reasoning_content", None) or "").strip()
    if thinking_prefix:
        thinking_trace = f"{thinking_prefix}\n{generated_thinking}".strip()
    else:
        thinking_trace = generated_thinking
    answer_text = (msg.content or "").strip()

    if thinking_trace:
        full_response = f"<think>\n{thinking_trace}\n</think>\n{answer_text}"
    else:
        full_response = answer_text

    return full_response, thinking_trace

def _native_generation_worker(
    rank: int,
    rows: List[dict],
    dataset: str,
    target_lang: str,
    tmp_path: str,
    port: int,
    system_prompt: str,
    thinking_prefix: str,
    log_dir: str,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(
        name=f"{dataset}_native.{target_lang}.w{rank}",
        log_dir=Path(log_dir),
        filename=f"{dataset}_native_gen_{target_lang}_w{rank}_{ts}.log",
    )
    logger.info("Worker %d processing %d rows for %s", rank, len(rows), target_lang)

    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    model_name = "Qwen/Qwen3-1.7B"

    with (
        open(tmp_path, "w", encoding="utf-8") as fout,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        for idx, row in enumerate(tqdm(rows, desc=f"[{target_lang}|w{rank}]", position=rank)):
            
            # OpenAI MMMLU schema diff
            if dataset == "mgsm":
                native_question = row["question"]
                gold_answer = str(row["answer"]).strip()
            else:
                native_question = _format_mmmlu_question(row)
                gold_answer = row.get("Answer", "").strip()
                if not gold_answer and "answer" in row: # default to english fallback if missing
                    # some datasets have answer instead of Answer
                    ans_val = row["answer"]
                    if isinstance(ans_val, int):
                        gold_answer = chr(65 + ans_val)
                    else:
                        gold_answer = str(ans_val)

            msgs = build_messages(system_prompt, native_question)
            
            fut: Future = pool.submit(
                generate_response_api,
                client, model_name, msgs, thinking_prefix,
                max_tokens=31000,
            )
            
            native_resp, native_thinking = fut.result()
            
            # Extract final answer
            final_text = native_resp.split("</think>")[-1] if "</think>" in native_resp else native_resp
            if dataset == "mgsm":
                native_answer = _extract_mgsm_answer(final_text)
            else:
                native_answer = _extract_mmmlu_answer(final_text)

            record = {
                "question_index": row["question_index"],
                "native_question": native_question,
                "native_response": native_resp,
                "native_answer": native_answer,
                "ground_truth": gold_answer,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("[w%d] Done", rank)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["bn", "ja", "sw"], required=True)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # We need to load questions from the native language split train file natively.
    # The native file has the questions translated according to the huggingface dataset.
    input_file = PROJECT_ROOT / "data" / "exp_v2" / args.dataset / f"train_{args.language}.jsonl"
    out_dir = PROJECT_ROOT / "data" / "exp_v2" / "qwen3-1_7b" / args.dataset / "native"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "outputs-exp_v2" / "logs"

    print(f"Loading {input_file} to generate native text...")
    rows = []
    if input_file.exists():
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    else:
        print(f"Error: {input_file} does not exist.")
        return

    lang_code = LANGUAGE_CODES[args.language]
    prompts = get_translate_dpo_prompts(lang_code)

    # For math datasets vs MMMLU, we might need to adjust the system prompt slightly to include the json or boxed rule
    if args.dataset == "mgsm":
        system_prompt = prompts["native_system_prompt_mgsm"]
    else:
        system_prompt = prompts["native_system_prompt_mmmlu"]
    thinking_prefix = prompts["thinking_prefix"]

    total = len(rows)
    print(f"Loaded {total} rows. Distributing to {args.workers} workers on port {args.port}.")

    chunk_size = math.ceil(total / args.workers)
    chunks = [rows[i * chunk_size : (i + 1) * chunk_size] for i in range(args.workers) if i * chunk_size < total]
    tmp_paths = [str(out_dir / f"part_native_{args.language}_{i}.jsonl") for i in range(len(chunks))]

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (chunk, tmp_path) in enumerate(zip(chunks, tmp_paths)):
        p = ctx.Process(
            target=_native_generation_worker,
            args=(rank, chunk, args.dataset, args.language, tmp_path, args.port, system_prompt, thinking_prefix, str(log_dir)),
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError(f"Native generation workers for {args.language} failed.")

    # Merge
    out_path = out_dir / f"native_{args.language}_responses.jsonl"
    total_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for tmp_path in tmp_paths:
            tmp = Path(tmp_path)
            if tmp.exists():
                with open(tmp, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        total_written += 1
                tmp.unlink()

    print(f"Successfully generated {total_written} native responses for {args.language} -> {out_path}")

if __name__ == "__main__":
    main()
