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
import random
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Regex for extracting A/B/C/D
_ANSWER_RE = re.compile(r"\b([A-D])\b")
# Regex for MGSM boxed answer
_BOXED_RE = re.compile(r"\\boxed{([^}]*)}")

import torch
import numpy as np
def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    max_tokens: int = 31000,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> Tuple[str, str]:
    """Call a vLLM server and return ``(full_response, thinking_trace)``."""
    # extra = {"chat_template_kwargs": {"enable_thinking": True}}
    extra = {
        "chat_template_kwargs": {"enable_thinking": True}, # turn off reasoning for translation task (wasted native tokens)
        "repetition_penalty": 1.2,  # add repetition penalty to mitigate potential loops in generation
    } 
    
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
    model_name: str,
    rank: int,
    native_rows: List[dict],
    english_rows: List[dict],
    dataset: str,
    tmp_path: str,
    port: int,
    log_dir: str,
    lang_code: str = "ja",
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(
        name=f"{dataset}_to_en.w{rank}",
        log_dir=Path(log_dir),
        filename=f"{dataset}_to_en_w{rank}_{ts}.log",
    )
    logger.info("Worker %d processing %d rows", rank, len(native_rows))

    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")

    # Set prompts based on dataset
    if lang_code == "ja":
        lang_name = "Japanese"
    elif lang_code == "bn":
        lang_name = "Bengali"
    elif lang_code == "sw":
        lang_name = "Swahili"
    else:
        raise ValueError(f"Unsupported language code: {lang_code}")
    
    if dataset == "mgsm":
        # system_prompt = "You are an expert math solver. Please reason step by step, and put your final answer within \\boxed{}."
        system_prompt = f"You are an expert translator. Your role is to translate the given math questions from {lang_name} into English. Do not attempt to solve the problem. You should return only the English translation of the questions without adding any explanations or extra text."
    else:  # mmmlu
        # system_prompt = 'You are an expert in multiple-choice questions. Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
        system_prompt = f"You are an expert translator. Your role is to translate the given math problems and all available options from {lang_name} into English. You must ensure your translation is accurate and faithful to the original meaning of the question and options. Do not solve the problem. Do not provide any explanations or extra text. You must return only the English translation of the question and options in the following format:\n\nQuestion: <Question in English>\n\nA. <Option A in English>\nB. <Option B in English>\nC. <Option C in English>\nD. <Option D in English>."
            
        # system_prompt = "Wewe ni mtafsiri mtaalamu. Jukumu lako ni kutafsiri matatizo ya hisabati na chaguzi zinazopatikana kutoka Kiswahili hadi Kiingereza. Usijaribu kutatua tatizo. Unapaswa kurudisha tafsiri ya maswali na chaguo kwa Kiingereza bila kuongeza maelezo yoyote au maandishi ya ziada."

    with (
        open(tmp_path, "w", encoding="utf-8") as fout,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        for idx, (native_row, english_row) in enumerate(tqdm(zip(native_rows, english_rows), desc=f"[w{rank}]", position=rank)):
            # Format question
            if dataset == "mgsm":
                native_question = native_row["question"] # mgsm already has it mapped cleanly
                english_question = english_row["question"]
                gold_answer = str(native_row["answer"]).strip()
            else:
                native_question = _format_mmmlu_question(native_row)
                english_question = _format_mmmlu_question(english_row)
                gold_answer = native_row.get("Answer", "").strip()
                if not gold_answer and "answer" in native_row: # default to english fallback if missing
                    # some datasets have answer instead of Answer
                    ans_val = native_row["answer"]
                    if isinstance(ans_val, int):
                        gold_answer = chr(65 + ans_val)
                    else:
                        gold_answer = str(ans_val)

            msgs = build_messages(system_prompt, native_question)
            print("Messages:", msgs)
            
            fut: Future = pool.submit(
                generate_response_api,
                client, model_name, msgs,
                # max_tokens=31000,
                max_tokens=5000,
            )
            
            english_resp, english_thinking = fut.result()

            # Extract final answer
            final_text = english_resp.split("</think>")[-1] if "</think>" in english_resp else english_resp
            final_text = final_text.strip()
            # if dataset == "mgsm":
            #     english_answer = _extract_mgsm_answer(final_text)
            # else:
            #     english_answer = _extract_mmmlu_answer(final_text)
            print("Translated English question:", final_text)

            record = {
                "question_index": native_row["question_index"],
                "native_question": native_question,
                "self_translated_english_question": final_text,
                "expert_translated_english_question": english_question,
                "ground_truth": gold_answer,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("[w%d] Done", rank)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--language", type=str, choices=["ja", "bn", "sw"], required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_cfg_name", type=str)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    setup_seed(args.seed)

    model_suffix = f"_{args.model_cfg_name}" if args.model_cfg_name else ""
    data_suffix = f"_{args.subset}" if args.subset else ""
    suffix = model_suffix + data_suffix
    
    input_file = PROJECT_ROOT / "data" / "exp_v3" / args.dataset / f"test_{args.language}{data_suffix}.jsonl"
    english_input_file = PROJECT_ROOT / "data" / "exp_v3" / args.dataset / f"test_en{data_suffix}.jsonl"
    out_dir = PROJECT_ROOT / "data" / "exp_v3" / args.dataset / f"test_{args.language}-en{suffix}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    log_dir = PROJECT_ROOT / "outputs-exp_v3" / "logs"

    print(f"Loading {args.language} input dataset from {input_file}...")
    native_rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            native_rows.append(json.loads(line))
    
    print(f"Loading English dataset with expert translation from {english_input_file}...")
    english_rows = []
    with open(english_input_file, "r", encoding="utf-8") as f:
        for line in f:
            english_rows.append(json.loads(line))

    assert len(native_rows) == len(english_rows), "Native and English datasets must have the same number of rows and be aligned."
    total = len(native_rows)
    print(f"Loaded {total} rows. Distributing to {args.workers} workers.")

    chunk_size = math.ceil(total / args.workers)
    native_chunks = [native_rows[i * chunk_size : (i + 1) * chunk_size] for i in range(args.workers) if i * chunk_size < total]
    english_chunks = [english_rows[i * chunk_size : (i + 1) * chunk_size] for i in range(args.workers) if i * chunk_size < total]
    tmp_paths = [str(out_dir / f"part_{i}.jsonl") for i in range(len(native_chunks))]

    if args.model_path is not None and "lora" not in args.model_cfg_name.lower():
        model_name = args.model_path
    else:
        model_name = "Qwen/Qwen3-1.7B"

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (native_chunk, english_chunk, tmp_path) in enumerate(zip(native_chunks, english_chunks, tmp_paths)):
        p = ctx.Process(
            target=_generation_worker,
            args=(model_name, rank, native_chunk, english_chunk, args.dataset, tmp_path, args.port, str(log_dir), args.language),
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
    out_path = out_dir / f"test_{args.language}-en.jsonl"
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
