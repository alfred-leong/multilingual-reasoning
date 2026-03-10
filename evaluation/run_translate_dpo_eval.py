#!/usr/bin/env python3
"""
Evaluate the Translate-DPO model vs the base Qwen3-1.7B model on MGSM.

Uses the **same settings as Setting 2** (native system prompt + thinking
prefix) to evaluate both the base model and the DPO-trained model.

Metrics:
  - **Accuracy**: fraction of correctly answered questions.
  - **Native language ratio**: fraction of native language in the full
    response (thinking trace + final answer).

Usage::

    python evaluation/run_translate_dpo_eval.py \\
        --language JA_JP --model_type base

    python evaluation/run_translate_dpo_eval.py \\
        --language JA_JP --model_type dpo

    python evaluation/run_translate_dpo_eval.py   # all languages, both models
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.language_utils import analyze_thinking_language, detect_language_ratio
from prompts.translate_dpo_prompts import get_translate_dpo_prompts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_mgsm_local(
    config: dict, mgsm_split: str, max_samples: int | None = None,
) -> list[dict]:
    """Load MGSM examples from a local TSV file."""
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


def _extract_numerical_answer(response_text: str) -> str | None:
    """Extract the final numerical answer from the response.

    Looks at text **after** ``</think>`` and returns the last number found.
    """
    think_end = response_text.rfind("</think>")
    final_part = response_text[think_end + len("</think>"):] if think_end != -1 else response_text

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


def _detect_full_response_language_ratio(
    response_text: str, language_code: str,
) -> dict:
    """Detect the native language ratio of the **full** response
    (thinking + answer), not just the thinking trace.
    """
    # Strip <think>/<\think> tags to get plain text
    clean = re.sub(r"</?think>", "", response_text).strip()
    return detect_language_ratio(clean, language_code)


# ---------------------------------------------------------------------------
# Generation with thinking prefix
# ---------------------------------------------------------------------------


def _generate_response(
    model,
    tokenizer,
    messages: list[dict],
    thinking_prefix: str = "",
    max_new_tokens: int = 2048,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate a response with optional thinking prefix.

    When *thinking_prefix* is given, an incomplete assistant message starting
    with ``<think>\\n{prefix}\\n`` is appended and the model continues from
    that point using ``continue_final_message=True``.  The prefix is
    prepended to the output so the returned string contains the full response.
    """
    if thinking_prefix:
        gen_messages = list(messages) + [
            {"role": "assistant", "content": f"<think>\n{thinking_prefix}\n"},
        ]
        text = tokenizer.apply_chat_template(
            gen_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        # Thinking mode without prefix: start the assistant message with <think>
        gen_messages = list(messages) + [
            {"role": "assistant", "content": "<think>\n"},
        ]
        text = tokenizer.apply_chat_template(
            gen_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature if temperature is not None else 0.6
        gen_kwargs["top_p"] = top_p if top_p is not None else 0.95
    else:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Reconstruct the full response including the prefix
    if thinking_prefix:
        full = f"<think>\n{thinking_prefix}\n{generated_text}"
    else:
        full = f"<think>\n{generated_text}"

    return full


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(
    language_code: str,
    model_type: str,
    config: dict,
    max_samples: int | None = None,
    adapter_dir_override: str | None = None,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    full_finetune: bool = False,
) -> dict:
    """Run MGSM evaluation and return results dict.

    Args:
        language_code: E.g. ``"JA_JP"``.
        model_type: ``"base"`` or ``"dpo"``.
        config: Parsed config dict.
        max_samples: Optional cap for debugging.
        adapter_dir_override: Path to adapter (LoRA) or full model directory.
        full_finetune: If True and model_type is ``"dpo"``, load the full
            fine-tuned model directly instead of using LoRA adapters.

    Returns:
        Results dict with accuracy, native language ratio, and per-example
        results.
    """
    # Resolve MGSM split
    lang_info = next(
        (l for l in config["languages"] if l["code"] == language_code), None
    )
    if lang_info is None:
        raise ValueError(f"Language {language_code} not found in config.")
    mgsm_split = lang_info["mgsm_split"]

    # Load MGSM
    print(f"[{language_code}] Loading MGSM split '{mgsm_split}' …")
    ds = _load_mgsm_local(config, mgsm_split, max_samples=max_samples)
    print(f"[{language_code}] {len(ds)} evaluation examples.")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model_name = config["models"]["base_model"]

    if model_type == "base":
        print(f"Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    elif model_type == "dpo":
        if adapter_dir_override is not None:
            adapter_dir = Path(adapter_dir_override)
        else:
            adapter_dir = (
                PROJECT_ROOT / "outputs-translate_dpo" / f"dpo_{language_code}"
            )
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {adapter_dir}. "
                "Run training first."
            )

        if full_finetune:
            # Full fine-tuned model: load directly from checkpoint directory
            print(f"Loading full fine-tuned model from {adapter_dir}")
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir), trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(adapter_dir),
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # LoRA adapter: load base model + merge adapter
            print(f"Loading {base_model_name} + LoRA adapter from {adapter_dir}")
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir), trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            model = model.merge_and_unload()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get prompts — same as Setting 2 (native system prompt + prefix)
    prompts = get_translate_dpo_prompts(language_code)
    system_prompt = prompts["native_system_prompt"]
    thinking_prefix = prompts["thinking_prefix"]

    # --- Evaluate -----------------------------------------------------------
    per_example: list[dict] = []
    num_correct = 0
    SAMPLE_EVERY = 5

    pbar = tqdm(
        ds,
        total=len(ds),
        desc=f"{language_code}/{model_type}",
        unit="ex",
        dynamic_ncols=True,
    )
    for idx, row in enumerate(pbar):
        question = row["question"]
        gold = row["answer_number"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = _generate_response(
            model, tokenizer, messages,
            thinking_prefix=thinking_prefix,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

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

        # Language analysis on the FULL response (thinking + answer)
        full_lang = _detect_full_response_language_ratio(response, language_code)
        # Language analysis on thinking trace only
        think_lang = analyze_thinking_language(response, language_code)

        per_example.append({
            "idx": idx,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted_str,
            "correct": correct,
            "response": response,
            "full_target_ratio": full_lang["target"],
            "full_english_ratio": full_lang["english"],
            "think_target_ratio": think_lang["target_ratio"],
            "think_english_ratio": think_lang["english_ratio"],
            "thinking_length": think_lang["thinking_length"],
        })

        acc_so_far = num_correct / (idx + 1)
        pbar.set_postfix(acc=f"{acc_so_far:.2%}", correct=num_correct)

        if idx % SAMPLE_EVERY == 0:
            sep = "-" * 60
            print(
                f"\n{sep}"
                f"\n[SAMPLE {idx}] {language_code} | {model_type}"
                f"\nQuestion : {question}"
                f"\nGold     : {gold}"
                f"\nPredicted: {predicted_str}  ({'✓' if correct else '✗'})"
                f"\n--- Response ---\n{response[:1000]}"
                + ("…[truncated]" if len(response) > 1000 else "")
                + f"\n{sep}\n",
                flush=True,
            )

    accuracy = num_correct / len(ds) if ds else 0.0
    mean_full_target = (
        sum(e["full_target_ratio"] for e in per_example) / len(per_example)
        if per_example else 0.0
    )
    mean_full_english = (
        sum(e["full_english_ratio"] for e in per_example) / len(per_example)
        if per_example else 0.0
    )
    mean_think_target = (
        sum(e["think_target_ratio"] for e in per_example) / len(per_example)
        if per_example else 0.0
    )
    mean_think_english = (
        sum(e["think_english_ratio"] for e in per_example) / len(per_example)
        if per_example else 0.0
    )

    results = {
        "experiment": "translate_dpo",
        "language": language_code,
        "model_type": model_type,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": len(ds),
        "mean_full_target_ratio": mean_full_target,
        "mean_full_english_ratio": mean_full_english,
        "mean_think_target_ratio": mean_think_target,
        "mean_think_english_ratio": mean_think_english,
        "per_example_results": per_example,
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Translate-DPO model vs base on MGSM."
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (e.g. JA_JP). If omitted, evaluate all.",
    )
    parser.add_argument(
        "--model_type", type=str, default=None,
        choices=["base", "dpo"],
        help="Model to evaluate. If omitted, evaluate both.",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "translate_dpo_config.yaml"),
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max examples (for debugging).",
    )
    parser.add_argument(
        "--adapter_dir", type=str, default=None,
        help="Path to a LoRA adapter directory (overrides the default "
             "outputs-translate_dpo/dpo_{LANG} path). Only used when "
             "model_type is 'dpo'.",
    )
    parser.add_argument(
        "--output_suffix", type=str, default="",
        help="Suffix appended to the output JSON filename, e.g. "
             "'_removethink+filter' → eval_JA_JP_dpo_removethink+filter.json",
    )
    parser.add_argument(
        "--full_finetune", action="store_true",
        help="Load a full fine-tuned model (no LoRA) from --adapter_dir. "
             "Only used when model_type is 'dpo'.",
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="Enable sampling (default: greedy decoding).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (only used when --do_sample is set).",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p nucleus sampling (only used when --do_sample is set).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    languages = (
        [args.language] if args.language
        else [l["code"] for l in config["languages"]]
    )
    model_types = [args.model_type] if args.model_type else ["base", "dpo"]

    output_base = PROJECT_ROOT / "outputs-translate_dpo"
    output_base.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        for mt in model_types:
            print(f"\n{'='*60}")
            print(f"  Evaluating: {lang} / {mt}")
            print(f"{'='*60}")

            results = evaluate(
                language_code=lang,
                model_type=mt,
                config=config,
                max_samples=args.max_samples,
                adapter_dir_override=args.adapter_dir,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                full_finetune=args.full_finetune,
            )

            suffix = args.output_suffix
            out_path = output_base / f"eval_{lang}_{mt}{suffix}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*60}")
            print(f"  Language:               {lang}")
            print(f"  Model:                  {mt}")
            print(f"  Accuracy:               {results['accuracy']:.2%} "
                  f"({results['num_correct']}/{results['num_total']})")
            print(f"  Full native ratio:      {results['mean_full_target_ratio']:.3f}")
            print(f"  Full english ratio:     {results['mean_full_english_ratio']:.3f}")
            print(f"  Thinking native ratio:  {results['mean_think_target_ratio']:.3f}")
            print(f"  Thinking english ratio: {results['mean_think_english_ratio']:.3f}")
            print(f"  Results saved:          {out_path}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
