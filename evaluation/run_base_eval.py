#!/usr/bin/env python3
"""
Evaluate the base Qwen3-8B model (no fine-tuning) on MGSM.

Runs all three thinking modes (natural, english_only, native_only) sequentially
for a single language.  Designed to be launched once per language so that the
caller (e.g. run_base_eval.sh) can assign one GPU per language.

Usage::

    CUDA_VISIBLE_DEVICES=5 python evaluation/run_base_eval.py --language JA_JP
    CUDA_VISIBLE_DEVICES=6 python evaluation/run_base_eval.py --language BN_BD
    CUDA_VISIBLE_DEVICES=7 python evaluation/run_base_eval.py --language SW_KE

Optional flags::

    --config        path to config.yaml (default: configs/config.yaml)
    --max_samples   cap number of MGSM examples (for quick debugging)
    --output_dir    override output directory (default: outputs/)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import csv

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.language_utils import analyze_thinking_language
from prompts.system_prompts import get_prompts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THINKING_MODES = ["natural", "english_only", "native_only"]

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _load_mgsm_local(
    config: dict, mgsm_split: str, max_samples: int | None = None
) -> list[dict]:
    """Load MGSM examples from a local TSV file.

    The TSV has two tab-separated columns: ``question`` and ``answer_number``.
    """
    local_dir = config["datasets"]["mgsm_local_dir"]
    tsv_path = Path(local_dir) / f"mgsm_{mgsm_split}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"MGSM local file not found: {tsv_path}")
    rows: list[dict] = []
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if len(line) < 2:
                continue
            rows.append({"question": line[0], "answer_number": line[1].strip()})
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


# ---------------------------------------------------------------------------
# Helpers (copied / adapted from run_mgsm_eval.py)
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_numerical_answer(response_text: str) -> str | None:
    """Return the last number found after the </think> block (if any)."""
    match = _THINK_RE.search(response_text)
    final_part = response_text[match.end():] if match else response_text
    numbers = _NUMBER_RE.findall(final_part)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def _normalise_number(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 2048,
) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Core evaluation loop for one (language, thinking_mode) pair
# ---------------------------------------------------------------------------


def evaluate_one_mode(
    *,
    language_code: str,
    thinking_mode: str,
    config: dict,
    model,
    tokenizer,
    mgsm_split: str,
    max_samples: int | None = None,
) -> dict:
    """Evaluate a loaded model on MGSM for a single thinking mode."""
    print(f"\n[{language_code}] thinking_mode={thinking_mode}")
    ds = _load_mgsm_local(config, mgsm_split, max_samples=max_samples)
    print(f"  {len(ds)} evaluation examples.")

    prompts_info = get_prompts(language_code, thinking_mode=thinking_mode)
    system_prompt = prompts_info["system_prompt"]
    cot_instruction = prompts_info["cot_instruction"]

    per_example: list[dict] = []
    num_correct = 0

    # Print a full response sample every SAMPLE_EVERY examples
    SAMPLE_EVERY = 5

    pbar = tqdm(
        ds,
        total=len(ds),
        desc=f"{language_code}/{thinking_mode}",
        unit="ex",
        dynamic_ncols=True,
    )
    for idx, row in enumerate(pbar):
        question = row["question"]
        gold = row["answer_number"]

        user_content = f"{cot_instruction}\n\n{question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = _generate_response(model, tokenizer, messages)
        predicted_str = _extract_numerical_answer(response)
        predicted = _normalise_number(predicted_str)
        gold_num = _normalise_number(gold)

        correct = (
            predicted is not None
            and gold_num is not None
            and abs(predicted - gold_num) < 1e-3
        )
        if correct:
            num_correct += 1

        lang_analysis = analyze_thinking_language(response, language_code)

        per_example.append(
            {
                "idx": idx,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": predicted_str,
                "correct": correct,
                "response": response,
                **lang_analysis,
            }
        )

        # Update progress bar with live accuracy
        acc_so_far = num_correct / (idx + 1)
        pbar.set_postfix(acc=f"{acc_so_far:.2%}", correct=num_correct)

        # Print a sample response periodically
        if idx % SAMPLE_EVERY == 0:
            sep = "-" * 60
            print(
                f"\n{sep}"
                f"\n[SAMPLE {idx}] {language_code} | {thinking_mode}"
                f"\nQuestion : {question}"
                f"\nGold     : {gold}"
                f"\nPredicted: {predicted_str}  ({'✓' if correct else '✗'})"
                f"\n--- Response ---\n{response[:1000]}"
                + ("…[truncated]" if len(response) > 1000 else "")
                + f"\n{sep}\n",
                flush=True,
            )

    accuracy = num_correct / len(ds) if len(ds) > 0 else 0.0
    mean_target = (
        sum(e["target_ratio"] for e in per_example) / len(per_example)
        if per_example
        else 0.0
    )
    mean_english = (
        sum(e["english_ratio"] for e in per_example) / len(per_example)
        if per_example
        else 0.0
    )

    return {
        "language": language_code,
        "model_type": "base",
        "model_name": config["models"]["base_model"],
        "thinking_mode": thinking_mode,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": len(ds),
        "mean_target_ratio": mean_target,
        "mean_english_ratio": mean_english,
        "per_example_results": per_example,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate base Qwen3-8B on MGSM (all thinking modes)."
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["JA_JP", "BN_BD", "SW_KE"],
        help="Language code to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of MGSM examples (for debugging).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <project_root>/outputs/).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    lang_info = next(
        (l for l in config["languages"] if l["code"] == args.language), None
    )
    if lang_info is None:
        raise ValueError(f"Language {args.language} not found in config.")
    mgsm_split = lang_info["mgsm_split"]

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model once; reuse across all thinking modes.
    model_name = config["models"]["base_model"]
    print(f"\n{'='*60}")
    print(f"Loading base model: {model_name}")
    print(f"Language: {args.language}  |  MGSM split: {mgsm_split}")
    print(f"Thinking modes: {THINKING_MODES}")
    print(f"{'='*60}\n")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results: dict[str, dict] = {}

    for thinking_mode in THINKING_MODES:
        results = evaluate_one_mode(
            language_code=args.language,
            thinking_mode=thinking_mode,
            config=config,
            model=model,
            tokenizer=tokenizer,
            mgsm_split=mgsm_split,
            max_samples=args.max_samples,
        )
        all_results[thinking_mode] = results

        # Save per-mode result file (mirrors naming in run_mgsm_eval.py)
        out_path = output_dir / f"eval_{args.language}_base_{thinking_mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(
            f"  Saved: {out_path}  "
            f"(acc={results['accuracy']:.2%}, "
            f"{results['num_correct']}/{results['num_total']})"
        )

    # Also write a combined summary for this language
    summary_path = output_dir / f"eval_{args.language}_base_summary.json"
    summary = {
        "language": args.language,
        "model_type": "base",
        "model_name": model_name,
        "results_by_mode": {
            tm: {
                k: v
                for k, v in r.items()
                if k != "per_example_results"
            }
            for tm, r in all_results.items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Base Qwen3-8B — MGSM Results — {args.language}")
    print(f"{'='*60}")
    print(f"  {'Thinking Mode':<20} {'Accuracy':>10}  {'Correct/Total':>14}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*14}")
    for tm, r in all_results.items():
        print(
            f"  {tm:<20} {r['accuracy']:>10.2%}  "
            f"{r['num_correct']:>6}/{r['num_total']:<6}"
        )
    print(f"{'='*60}")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
