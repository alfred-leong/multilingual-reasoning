#!/usr/bin/env python3
import argparse
import json
import logging
import math
import multiprocessing as mp
import re
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import openai
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Regex for extracting A/B/C/D
_ANSWER_RE = re.compile(r"\b([A-D])\b")
# ISO 639-1 -> human-readable language name
_ISO_LANG_NAME: dict[str, str] = {
    "en": "English",
    "ja": "Japanese",
    "bn": "Bengali",
    "sw": "Swahili",
}

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
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
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else ""

def _extract_mgsm_answer(text: str) -> str:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""

def _split_response(response: str) -> Tuple[str, str]:
    """Split a response into (thinking_trace, final_answer).

    Returns ``("", response)`` when no ``<think>`` block is present.
    """
    match = _THINK_RE.search(response)
    if match:
        thinking = match.group(1).strip()
        answer = response[match.end():].strip()
        return thinking, answer
    return "", response.strip()

def _build_translate_prompt(
    text: str,
    target_lang_code: str,
    source_lang_code: str = "en",
) -> str:
    """Build the raw prompt string that reproduces TranslateGemma's chat template.

    vLLM's OpenAI-compatible chat API strips non-standard content-part fields
    (``source_lang_code``, ``target_lang_code``) that TranslateGemma's Jinja
    template requires, so we construct the prompt manually and use the
    completions API instead.
    """
    src_name = _ISO_LANG_NAME[source_lang_code]
    tgt_name = _ISO_LANG_NAME[target_lang_code]
    return (
        f"<bos><start_of_turn>user\n"
        f"You are a professional {src_name} ({source_lang_code}) to "
        f"{tgt_name} ({target_lang_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the "
        f"original {src_name} text while adhering to {tgt_name} grammar, "
        f"vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt_name} translation, without any additional "
        f"explanations or commentary. "
        f"Please translate the following {src_name} text into {tgt_name}:"
        f"\n\n\n{text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def translate_text(
    client: openai.OpenAI,
    model_name: str,
    text: str,
    target_lang_code: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    top_p: float = 0.95,
) -> str:
    """Translate *text* from English to *target_lang_code* using TranslateGemma.

    Args:
        client: OpenAI client pointed at the TranslateGemma vLLM server.
        model_name: HuggingFace model ID for TranslateGemma.
        text: Source text in English.
        target_lang_code: ISO 639-1 language code (e.g. ``"ja"``, ``"bn"``,
            ``"sw"``).
        max_tokens: Maximum tokens for the translation output.
        temperature: Sampling temperature.
        top_p: Top-p nucleus sampling.

    Returns:
        The translated text.
    """
    if not text.strip():
        return ""

    prompt = _build_translate_prompt(text, target_lang_code)

    for attempt in range(5):
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return (response.choices[0].text or "").strip()
        except (openai.APITimeoutError, openai.APIError) as e:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    return ""


def translate_response(
    client: openai.OpenAI,
    model_name: str,
    english_response: str,
    target_lang_code: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_think_chars: int = 2000,
    max_answer_chars: int = 2000,
) -> str:
    """Translate a full English response (thinking + answer) to native language.

    The ``<think>…</think>`` structure is preserved: the thinking trace and
    the final answer are translated independently, then reassembled.

    Both inputs are hard-capped to keep every request well under 60 seconds
    on a single GPU (27B model, ~30 tok/s):
    - max_think_chars=2000  → ~500 tokens input, 512 max output  → ~17s
    - max_answer_chars=2000 → ~500 tokens input, 256 max output  → ~10s
    The 9 MMMLU items with answers >50k chars would never complete otherwise.
    """
    thinking, answer = _split_response(english_response)

    if max_think_chars and len(thinking) > max_think_chars:
        thinking = thinking[:max_think_chars]
    if max_answer_chars and len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars]

    # Size max_tokens to the (truncated) input — no idle token budget.
    think_max_tokens = min(max(len(thinking) // 2, 128), 512)
    answer_max_tokens = min(max(len(answer) // 2, 64), 256)

    translated_thinking = translate_text(
        client, model_name, thinking, target_lang_code,
        think_max_tokens, temperature, top_p,
    )
    translated_answer = translate_text(
        client, model_name, answer, target_lang_code,
        answer_max_tokens, temperature, top_p,
    )

    if translated_thinking:
        return f"<think>\n{translated_thinking}\n</think>\n{translated_answer}"
    return translated_answer

def _translation_worker(
    rank: int,
    rows: List[dict],
    dataset: str,
    target_lang: str,
    tmp_path: str,
    port: int,
    log_dir: str,
    translate_model: str = "google/translategemma-27b-it",
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(
        name=f"{dataset}_trans.{target_lang}.w{rank}",
        log_dir=Path(log_dir),
        filename=f"{dataset}_trans_{target_lang}_w{rank}_{ts}.log",
    )
    logger.info("Worker %d processing %d rows for %s", rank, len(rows), target_lang)

    # Each request is now ~500-1000 tokens total (capped input + small output).
    # 5 workers × 3 concurrent = 15 requests/GPU max — fine for short requests.
    CONCURRENT = 3

    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY", timeout=90.0)
    model_name = translate_model

    def _make_record(row: dict, translated_response: str) -> dict:
        final_text = translated_response.split("</think>")[-1] if "</think>" in translated_response else translated_response
        if dataset == "mgsm":
            translated_answer = _extract_mgsm_answer(final_text)
        else:
            translated_answer = _extract_mmmlu_answer(final_text)
        if not translated_answer:
            translated_answer = row.get("english_answer", "")
        record = row.copy()
        record["translated_response"] = translated_response
        record["translated_answer"] = translated_answer
        record["english_ans=translated_ans"] = (row.get("english_answer", "") == translated_answer)
        return record

    with (
        open(tmp_path, "w", encoding="utf-8") as fout,
        ThreadPoolExecutor(max_workers=CONCURRENT) as pool,
    ):
        # Sliding window: keep up to CONCURRENT futures in-flight at once.
        # deque of (future, row) pairs; drain from the left when window is full.
        window: deque[tuple[Future, dict]] = deque()
        pbar = tqdm(total=len(rows), desc=f"[{target_lang}|w{rank}]", position=rank)

        def _drain_one() -> None:
            fut, row = window.popleft()
            try:
                translated_response = fut.result()
            except (openai.APITimeoutError, openai.APIError) as e:
                logger.warning("qid=%s failed after retries: %s. Skipping.", row.get("question_index"), e)
                pbar.update(1)
                return
            fout.write(json.dumps(_make_record(row, translated_response), ensure_ascii=False) + "\n")
            pbar.update(1)

        for row in rows:
            english_response = row.get("english_response", "")
            if not english_response:
                pbar.update(1)
                continue
            if len(window) >= CONCURRENT:
                _drain_one()
            window.append((pool.submit(translate_response, client, model_name, english_response, target_lang), row))

        # Drain remaining
        while window:
            _drain_one()

        pbar.close()

    logger.info("[w%d] Done", rank)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["bn", "ja", "sw"], required=True)
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-model", type=str, default="qwen3-1_7b",
                        help="Short name of the source model (e.g. qwen3-8b)")
    parser.add_argument("--translate-model", type=str, default="google/translategemma-27b-it",
                        help="HuggingFace model ID or local path for the translation model")
    args = parser.parse_args()

    input_file = PROJECT_ROOT / "data" / "exp_v2" / args.source_model / args.dataset / "english" / "english_responses.jsonl"
    out_dir = PROJECT_ROOT / "data" / "exp_v2" / args.source_model / args.dataset / "translated_gemma-27b"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "outputs-exp_v2" / "logs"

    print(f"Loading {input_file} to translate to {args.language}...")
    rows = []
    if input_file.exists():
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    else:
        print(f"Error: {input_file} does not exist.")
        return

    # Check if translated_{args.language}_responses.jsonl exists
    if (out_dir / f"translated_{args.language}_responses.jsonl").exists():
        print(f"Warning: {out_dir / f'translated_{args.language}_responses.jsonl'} already exists... Skipping translation.")
        return

    # Check for existing progress
    processed_qids = set()
    resumed_records = []
    existing_parts = list(out_dir.glob(f"part_{args.language}_*.jsonl"))
    if existing_parts:
        print(f"Found {len(existing_parts)} existing part files. Checking for progress...")
        for p in existing_parts:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        qid = record.get("question_index")
                        if qid is not None:
                            processed_qids.add(qid)
                        
                        # Add/Update the keys
                        eng_ans = record.get("english_answer", "")
                        trans_ans = record.get("translated_answer", "")
                        is_same = (eng_ans == trans_ans)
                        record["english_ans=translated_ans"] = is_same
                        resumed_records.append(record)
                    except json.JSONDecodeError:
                        continue
    
    rows = [r for r in rows if r.get("question_index") not in processed_qids]
    total = len(rows)
    print(f"Resuming: {len(processed_qids)} already done. {total} remaining. Distributing to {args.workers} workers on port {args.port}.")

    resumed_path = None
    if resumed_records:
        resumed_path = out_dir / f"part_{args.language}_resumed.jsonl"
        with open(resumed_path, "w", encoding="utf-8") as f:
            for rec in resumed_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Now safe to delete old parts that aren't the resumed one
        for p in existing_parts:
            if p != resumed_path:
                p.unlink()

    chunk_size = math.ceil(total / args.workers)
    chunks = [rows[i * chunk_size : (i + 1) * chunk_size] for i in range(args.workers) if i * chunk_size < total]
    tmp_paths = [str(out_dir / f"part_{args.language}_{i}.jsonl") for i in range(len(chunks))]

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (chunk, tmp_path) in enumerate(zip(chunks, tmp_paths)):
        p = ctx.Process(
            target=_translation_worker,
            args=(rank, chunk, args.dataset, args.language, tmp_path, args.port, str(log_dir), args.translate_model),
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError(f"Translation workers for {args.language} failed.")

    # Merge
    out_path = out_dir / f"translated_{args.language}_responses.jsonl"
    total_written = 0
    all_tmp_paths = tmp_paths
    if resumed_path and resumed_path.exists():
        all_tmp_paths = [str(resumed_path)] + all_tmp_paths

    with open(out_path, "w", encoding="utf-8") as fout:
        for tmp_path in all_tmp_paths:
            tmp = Path(tmp_path)
            if tmp.exists():
                with open(tmp, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        total_written += 1
                tmp.unlink()

    print(f"Successfully generated {total_written} translated responses for {args.language} -> {out_path}")

if __name__ == "__main__":
    main()
