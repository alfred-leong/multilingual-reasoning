#!/usr/bin/env python3
"""
Generate responses for the Translate-DPO experiment.

For each language and each MMMLU question, generates two responses from
Qwen3-1.7B in thinking mode via an OpenAI-compatible vLLM server:

  **Setting 1 (English)** — English system prompt, thinking mode enabled,
      no thinking prefix.  The model reasons in English.
  **Setting 2 (Native)** — Native-language system prompt + a thinking prefix
      that seeds ``<think>`` in the target language.

Output is written to ``data/translate_dpo/raw/{LANG}_responses.jsonl``.

Usage::

    python data/generate_translate_dpo_responses.py --language JA_JP
    python data/generate_translate_dpo_responses.py              # all languages
    python data/generate_translate_dpo_responses.py --language JA_JP --max_samples 50
"""

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
import yaml
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.translate_dpo_prompts import get_translate_dpo_prompts

# Number of parallel worker processes (IO-bound — HTTP calls to vLLM).
N_WORKERS = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"\b([A-D])\b")


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_final_answer(text: str) -> str:
    """Return the last standalone A/B/C/D letter found in *text*."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else ""


def _format_mmmlu_question(row: dict) -> str:
    """Format an MMMLU row (openai/MMMLU format) into a question string."""
    q = row["Question"]
    choices = "\n".join(
        f"{letter}. {row[letter]}" for letter in ("A", "B", "C", "D")
    )
    return f"{q}\n\n{choices}"


def _format_mmlu_question(row: dict) -> str:
    """Format an MMLU row (cais/mmlu format) into a question string."""
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


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------


def generate_response_api(
    client: openai.OpenAI,
    model_name: str,
    messages: list,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    thinking_prefix: str = "",
) -> Tuple[str, str]:
    """Call a vLLM server and return ``(full_response, thinking_trace)``.

    When *thinking_prefix* is given it is injected as a partial assistant
    message so the model continues CoT in the target language.
    """
    extra: dict = {"chat_template_kwargs": {"enable_thinking": True}}

    if thinking_prefix:
        messages = list(messages) + [
            {"role": "assistant", "content": f"<think>\n{thinking_prefix}\n"}
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


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _generation_worker(
    rank: int,
    row_pairs: List[Tuple[dict, dict]],
    language_code: str,
    config: dict,
    tmp_path: str,
    prompts: dict,
    log_dir: str,
    port: int,
) -> None:
    """Subprocess worker: generates Setting-1 + Setting-2 responses for a
    slice of the MMMLU dataset.

    Each element in *row_pairs* is ``(english_row, native_row)`` — the same
    question in English (cais/mmlu format) and the native language
    (openai/MMMLU format).
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(
        name=f"gen_tdpo.{language_code}.w{rank}",
        log_dir=Path(log_dir),
        filename=f"gen_tdpo_{language_code}_w{rank}_{ts}.log",
    )
    logger.info("Worker %d starting: %d rows → %s", rank, len(row_pairs), tmp_path)

    model_name = config["models"]["generation_model"]
    gen_cfg = config.get("generation", {})
    max_tokens = gen_cfg.get("max_tokens", 8192)
    temperature = gen_cfg.get("temperature", 0.6)
    top_p = gen_cfg.get("top_p", 0.95)

    client = openai.OpenAI(
        base_url=f"http://localhost:{port}/v1", api_key="EMPTY"
    )

    english_sp = prompts["english_system_prompt"]
    native_sp = prompts["native_system_prompt"]
    thinking_prefix = prompts["thinking_prefix"]

    with (
        open(tmp_path, "w", encoding="utf-8") as fout,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        for idx, (en_row, native_row) in enumerate(tqdm(
            row_pairs,
            desc=f"[{language_code}|w{rank}]",
            unit="ex",
            position=rank,
            leave=True,
        )):
            # English question for Setting 1, native question for Setting 2
            english_question = _format_mmlu_question(en_row)
            native_question = _format_mmmlu_question(native_row)

            # --- Setting 1: English system prompt + English question --------
            msgs_en = build_messages(english_sp, english_question)
            fut_en: Future = pool.submit(
                generate_response_api,
                client, model_name, msgs_en,
                max_tokens, temperature, top_p,
                "",  # no prefix
            )

            # --- Setting 2: Native system prompt + native question ----------
            msgs_native = build_messages(native_sp, native_question)
            fut_native: Future = pool.submit(
                generate_response_api,
                client, model_name, msgs_native,
                max_tokens, temperature, top_p,
                thinking_prefix,
            )

            english_resp, english_thinking = fut_en.result()
            native_resp, native_thinking = fut_native.result()

            # Extract final answers
            en_final = english_resp.split("</think>")[-1] if "</think>" in english_resp else english_resp
            na_final = native_resp.split("</think>")[-1] if "</think>" in native_resp else native_resp
            english_answer = _extract_final_answer(en_final)
            native_answer = _extract_final_answer(na_final)

            # Gold answer: prefer letter (MMMLU), fall back to index (MMLU)
            gold = native_row.get("Answer", "")
            if not gold and "answer" in en_row:
                gold = chr(65 + en_row["answer"])  # 0→A, 1→B, …

            logger.debug(
                "idx=%d | en_ans=%s | na_ans=%s | gold=%s",
                idx, english_answer, native_answer, gold,
            )

            # Log full responses every 10 questions
            if idx % 10 == 0:
                logger.info(
                    "\n========== SAMPLE idx=%d ==========\n"
                    "[ENGLISH QUESTION (Setting 1)]\n%s\n\n"
                    "[NATIVE QUESTION (Setting 2)]\n%s\n\n"
                    "[ENGLISH RESPONSE (Setting 1)]\n%s\n\n"
                    "[NATIVE RESPONSE (Setting 2)]\n%s\n"
                    "====================================",
                    idx, english_question, native_question,
                    english_resp, native_resp,
                )

            record = {
                "english_question": english_question,
                "native_question": native_question,
                "english_response": english_resp,
                "english_thinking": english_thinking,
                "english_answer": english_answer,
                "native_response": native_resp,
                "native_thinking": native_thinking,
                "native_answer": native_answer,
                "gold_answer": gold,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("[w%d] Done: %d records → %s", rank, len(row_pairs), tmp_path)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def generate_for_language(
    language_code: str,
    config: dict,
    output_dir: Path,
    port: int,
    max_samples: int | None = None,
) -> None:
    """Generate Setting-1 + Setting-2 responses for one language."""
    log_dir = PROJECT_ROOT / "outputs-translate_dpo" / "logs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(
        name=f"gen_tdpo.{language_code}",
        log_dir=log_dir,
        filename=f"gen_tdpo_{language_code}_coord_{ts}.log",
    )

    prompts = get_translate_dpo_prompts(language_code)
    logger.info("English SP : %s", prompts["english_system_prompt"])
    logger.info("Native SP  : %s", prompts["native_system_prompt"])
    logger.info("Prefix     : %s", prompts["thinking_prefix"])

    # --- Load MMMLU (native) + MMLU (English) --------------------------------
    logger.info("Loading MMMLU (native) split for %s …", language_code)
    ds_native = load_dataset(
        config["datasets"]["preference_source"],
        language_code,
        split="test",
    )
    english_source = config["datasets"].get(
        "english_source", "cais/mmlu"
    )
    logger.info("Loading English MMLU from %s …", english_source)
    ds_english = load_dataset(english_source, "all", split="test")

    assert len(ds_native) == len(ds_english), (
        f"Dataset length mismatch: native={len(ds_native)} vs "
        f"english={len(ds_english)}. Cannot pair rows."
    )

    if max_samples is not None:
        n = min(max_samples, len(ds_native))
        ds_native = ds_native.select(range(n))
        ds_english = ds_english.select(range(n))
    total = len(ds_native)
    logger.info("Loaded %d paired examples → %d workers.", total, N_WORKERS)

    all_pairs: List[Tuple[dict, dict]] = [
        (dict(ds_english[i]), dict(ds_native[i])) for i in range(total)
    ]

    # --- Split across workers -----------------------------------------------
    chunk_size = math.ceil(total / N_WORKERS)
    chunks = [
        all_pairs[i * chunk_size : (i + 1) * chunk_size]
        for i in range(N_WORKERS)
        if i * chunk_size < total
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths = [
        str(output_dir / f"{language_code}_part{r}.jsonl")
        for r in range(len(chunks))
    ]

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (chunk, tmp_path) in enumerate(zip(chunks, tmp_paths)):
        p = ctx.Process(
            target=_generation_worker,
            args=(
                rank, chunk, language_code, config,
                tmp_path, prompts, str(log_dir), port,
            ),
            name=f"tdpo-gen-{language_code}-w{rank}",
        )
        p.start()
        procs.append(p)
        logger.info("Launched worker %d (pid=%d) for %d pairs.", rank, p.pid, len(chunk))

    failed = []
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            logger.error("Worker %d exited with code %d.", rank, p.exitcode)
            failed.append(rank)

    if failed:
        raise RuntimeError(
            f"[{language_code}] Workers {failed} failed. Check logs."
        )

    # --- Merge temp files ---------------------------------------------------
    out_path = output_dir / f"{language_code}_responses.jsonl"
    total_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for rank, tmp_path in enumerate(tmp_paths):
            tmp = Path(tmp_path)
            if not tmp.exists():
                raise FileNotFoundError(f"Missing temp file from worker {rank}: {tmp_path}")
            with open(tmp, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total_written += 1
            tmp.unlink()

    logger.info("Merge complete: %d records → %s", total_written, out_path)
    print(f"[{language_code}] {total_written} records → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Setting-1 (English) + Setting-2 (Native) "
                    "responses for the Translate-DPO experiment."
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (e.g. JA_JP). If omitted, run all.",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "translate_dpo_config.yaml"),
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="vLLM server port for Qwen3-1.7B (overrides config).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max examples per language (for debugging).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = PROJECT_ROOT / "data" / "translate_dpo" / "raw"
    port = args.port or config["vllm_ports"]["generation"]

    languages = (
        [args.language] if args.language
        else [l["code"] for l in config["languages"]]
    )
    for lang in languages:
        generate_for_language(lang, config, output_dir, port, args.max_samples)


if __name__ == "__main__":
    main()
