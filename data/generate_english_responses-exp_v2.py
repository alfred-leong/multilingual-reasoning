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

# Regex for extracting A/B/C/D
_ANSWER_RE = re.compile(r"\b([A-D])\b")
# Regex for MGSM boxed answer
_BOXED_RE = re.compile(r"\\boxed{([^}]*)}")

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
    """Return the last standalone A/B/C/D letter found in *text*."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else ""

def _extract_mgsm_answer(text: str) -> str:
    """Extract answer from \boxed{} in *text*."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""

def _format_mmmlu_question(row: dict) -> str:
    q = row["question"]
    letters = ("A", "B", "C", "D")
    choices = "\n".join(
        f"{letters[i]}. {c}" for i, c in enumerate(row["choices"])
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
    max_tokens: int = 31000,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Tuple[str, str]:
    """Call a vLLM server and return ``(full_response, thinking_trace)``."""
    extra = {"chat_template_kwargs": {"enable_thinking": True}}

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
    answer_text = (msg.content or "").strip()

    if generated_thinking:
        full_response = f"<think>\n{generated_thinking}\n</think>\n{answer_text}"
    else:
        full_response = answer_text

    return full_response, generated_thinking

def _generation_worker(
    rank: int,
    rows: List[dict],
    dataset: str,
    tmp_path: str,
    port: int,
    log_dir: str,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(
        name=f"gen_eng.w{rank}",
        log_dir=Path(log_dir),
        filename=f"gen_eng_w{rank}_{ts}.log",
    )
    logger.info("Worker %d processing %d rows", rank, len(rows))

    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    model_name = "Qwen/Qwen3-1.7B"

    # Set prompts based on dataset
    if dataset == "mgsm":
        system_prompt = "You are an expert math solver. Please reason step by step, and put your final answer within \\boxed{}."
    else:  # mmmlu
        system_prompt = 'You are an expert in multiple-choice questions. Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'

    with (
        open(tmp_path, "w", encoding="utf-8") as fout,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        for idx, row in enumerate(tqdm(rows, desc=f"[w{rank}]", position=rank)):
            # Format question
            if dataset == "mgsm":
                english_question = row["question"] # mgsm already has it mapped cleanly
                gold_answer = str(row["answer"]).strip()
            else:
                english_question = _format_mmmlu_question(row)
                ans_idx = row.get("answer", -1)
                gold_answer = chr(65 + ans_idx) if ans_idx != -1 else ""

            msgs = build_messages(system_prompt, english_question)
            
            fut: Future = pool.submit(
                generate_response_api,
                client, model_name, msgs,
                max_tokens=31000,
            )
            
            english_resp, english_thinking = fut.result()
            
            # Extract final answer
            final_text = english_resp.split("</think>")[-1] if "</think>" in english_resp else english_resp
            if dataset == "mgsm":
                english_answer = _extract_mgsm_answer(final_text)
            else:
                english_answer = _extract_mmmlu_answer(final_text)

            record = {
                "question_index": row["question_index"],
                "english_question": english_question,
                "english_response": english_resp,
                "english_answer": english_answer,
                "ground_truth": gold_answer,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("[w%d] Done", rank)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    input_file = PROJECT_ROOT / "data" / "exp_v2" / args.dataset / "train_en.jsonl"
    out_dir = PROJECT_ROOT / "data" / "exp_v2" / "qwen3-1_7b" / args.dataset / "english"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "outputs-exp_v2" / "logs"

    print(f"Loading {input_file}...")
    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            
    total = len(rows)
    print(f"Loaded {total} rows. Distributing to {args.workers} workers.")

    chunk_size = math.ceil(total / args.workers)
    chunks = [rows[i * chunk_size : (i + 1) * chunk_size] for i in range(args.workers) if i * chunk_size < total]
    tmp_paths = [str(out_dir / f"part_{i}.jsonl") for i in range(len(chunks))]

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (chunk, tmp_path) in enumerate(zip(chunks, tmp_paths)):
        p = ctx.Process(
            target=_generation_worker,
            args=(rank, chunk, args.dataset, tmp_path, args.port, str(log_dir)),
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError("Generation workers failed.")

    # Merge
    out_path = out_dir / "english_responses.jsonl"
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

    print(f"Successfully generated {total_written} English responses -> {out_path}")

if __name__ == "__main__":
    main()
