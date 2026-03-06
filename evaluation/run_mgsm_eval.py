#!/usr/bin/env python3
"""
Evaluate reasoning accuracy on MGSM with language-fidelity analysis.

For each question in the MGSM split for a given language, this script:
  1. Formats the prompt using the native system prompt with the requested
     thinking mode (natural / english_only / native_only).
  2. Runs inference with either the base model or a DPO-finetuned checkpoint.
  3. Extracts the numerical answer from the final-answer portion of the
     response and compares against the gold answer.
  4. Analyses the thinking trace for language composition using
     ``data.language_utils.analyze_thinking_language``.
  5. Saves per-example and aggregate results as JSON.

Usage::

    python evaluation/run_mgsm_eval.py \\
        --language JA_JP --model_type base --thinking_mode natural

    python evaluation/run_mgsm_eval.py \\
        --language BN_BD --model_type dpo --dpo_mode 2 --thinking_mode native_only
"""

import argparse
import json
import re
import sys
from pathlib import Path

import csv

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.language_utils import analyze_thinking_language
from prompts.system_prompts import get_prompts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_numerical_answer(response_text: str) -> str | None:
    """Extract the final numerical answer from the response.

    Looks at the text **after** the ``</think>`` closing tag (if present) and
    returns the last number found.  Returns ``None`` if no number is detected.
    """
    match = _THINK_RE.search(response_text)
    final_part = response_text[match.end():] if match else response_text

    numbers = _NUMBER_RE.findall(final_part)
    if not numbers:
        return None
    # Take the last number found in the final-answer section
    return numbers[-1].replace(",", "")


def _normalise_number(value) -> float | None:
    """Try to parse a value as a float for comparison."""
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
    """Greedy-decode a response from *model*."""
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
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(
    language_code: str,
    model_type: str,
    thinking_mode: str,
    config: dict,
    dpo_mode: int | None = None,
    max_samples: int | None = None,
) -> dict:
    """Run MGSM evaluation and return results dict.

    Args:
        language_code: E.g. ``"JA_JP"``.
        model_type: ``"base"`` or ``"dpo"``.
        thinking_mode: ``"natural"``, ``"english_only"``, or ``"native_only"``.
        config: Parsed config dict.
        dpo_mode: Required when *model_type* is ``"dpo"`` — the filtering
            mode (1 or 2) used during DPO training.
        max_samples: Optional cap for debugging.

    Returns:
        Results dict with keys ``accuracy``, ``num_correct``, ``num_total``,
        ``mean_target_ratio``, ``mean_english_ratio``, ``per_example_results``.
    """
    # Resolve MGSM split name
    lang_info = next(
        (l for l in config["languages"] if l["code"] == language_code), None
    )
    if lang_info is None:
        raise ValueError(f"Language {language_code} not found in config.")
    mgsm_split = lang_info["mgsm_split"]

    # Load MGSM dataset from local TSV
    print(f"[{language_code}] Loading MGSM split '{mgsm_split}' from local files …")
    ds = _load_mgsm_local(config, mgsm_split, max_samples=max_samples)
    print(f"[{language_code}] {len(ds)} evaluation examples.")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if model_type == "base":
        model_name = config["models"]["base_model"]
        print(f"Loading base model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
    elif model_type == "dpo":
        if dpo_mode is None:
            raise ValueError("--dpo_mode is required when model_type is 'dpo'.")
        base_name = config["models"]["base_model"]
        adapter_dir = (
            PROJECT_ROOT / "outputs" / f"dpo_{language_code}_mode{dpo_mode}"
        )
        if not adapter_dir.exists():
            raise FileNotFoundError(f"DPO checkpoint not found: {adapter_dir}")
        print(f"Loading base model {base_name} + LoRA adapter from {adapter_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_dir), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.merge_and_unload()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompts
    prompts_info = get_prompts(language_code, thinking_mode=thinking_mode)
    system_prompt = prompts_info["system_prompt"]
    cot_instruction = prompts_info["cot_instruction"]

    # Evaluate
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
        gold = row["answer_number"]  # MGSM provides numerical gold answers

        user_content = f"{cot_instruction}\n\n{question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = _generate_response(model, tokenizer, messages)
        predicted_str = _extract_numerical_answer(response)
        predicted = _normalise_number(predicted_str)
        gold_num = _normalise_number(gold)

        correct = (predicted is not None and gold_num is not None
                    and abs(predicted - gold_num) < 1e-3)
        if correct:
            num_correct += 1

        # Language analysis on thinking trace
        lang_analysis = analyze_thinking_language(response, language_code)

        per_example.append({
            "idx": idx,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted_str,
            "correct": correct,
            "response": response,
            **lang_analysis,
        })

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

    accuracy = num_correct / len(ds) if ds else 0.0
    mean_target = (
        sum(e["target_ratio"] for e in per_example) / len(per_example)
        if per_example else 0.0
    )
    mean_english = (
        sum(e["english_ratio"] for e in per_example) / len(per_example)
        if per_example else 0.0
    )

    results = {
        "language": language_code,
        "model_type": model_type,
        "thinking_mode": thinking_mode,
        "dpo_mode": dpo_mode,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": len(ds),
        "mean_target_ratio": mean_target,
        "mean_english_ratio": mean_english,
        "per_example_results": per_example,
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run MGSM evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate on MGSM.")
    parser.add_argument(
        "--language", type=str, required=True, help="Language code."
    )
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["base", "dpo"],
        help="Which model to evaluate.",
    )
    parser.add_argument(
        "--thinking_mode", type=str, required=True,
        choices=["natural", "english_only", "native_only"],
        help="Thinking-mode constraint.",
    )
    parser.add_argument(
        "--dpo_mode", type=int, default=None, choices=[1, 2],
        help="DPO filtering mode (required if model_type=dpo).",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max examples (for debugging).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    results = evaluate(
        language_code=args.language,
        model_type=args.model_type,
        thinking_mode=args.thinking_mode,
        config=config,
        dpo_mode=args.dpo_mode,
        max_samples=args.max_samples,
    )

    # Determine output filename
    model_label = args.model_type
    if args.model_type == "dpo" and args.dpo_mode:
        model_label = f"dpo_mode{args.dpo_mode}"
    out_path = (
        PROJECT_ROOT / "outputs"
        / f"eval_{args.language}_{model_label}_{args.thinking_mode}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Language:        {args.language}")
    print(f"Model:           {model_label}")
    print(f"Thinking mode:   {args.thinking_mode}")
    print(f"Accuracy:        {results['accuracy']:.2%} "
          f"({results['num_correct']}/{results['num_total']})")
    print(f"Mean target ratio: {results['mean_target_ratio']:.3f}")
    print(f"Mean english ratio: {results['mean_english_ratio']:.3f}")
    print(f"Results saved to:  {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
