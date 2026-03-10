#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) with LoRA on the Translate-DPO translated data.

Trains a Qwen3-1.7B model with LoRA using standard SFT on
translated English reasoning traces.

Two modes:
  - **Mode 1** – Remove think tags and SFT on *all* data samples (~14k per lang).
  - **Mode 2** – Remove think tags, then filter to keep only samples where the
                 English response was correct (english_answer == gold_answer).

Usage::

    python training/run_translate_sft_training.py --language JA_JP --mode 1
    python training/run_translate_sft_training.py --language BN_BD --mode 2
    python training/run_translate_sft_training.py --mode 1   # all languages
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.translate_dpo_prompts import get_translate_dpo_prompts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>`` and ``</think>`` tags, returning plain text."""
    return re.sub(r"</?think>", "", text).strip()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_translate_sft(
    language_code: str,
    mode: int,
    config: dict,
    output_base: Path,
    full_finetune: bool = False,
) -> None:
    """Run SFT training for one language and mode.

    Args:
        language_code: E.g. ``"JA_JP"``.
        mode: 1 = all data (think tags removed),
              2 = correct-only filtered (think tags removed).
        config: Parsed translate-DPO config dict.
        output_base: Root directory for model checkpoints.
        full_finetune: If True, train all parameters (no LoRA).
    """
    # --- Load translated data -----------------------------------------------
    translated_path = (
        PROJECT_ROOT / "data" / "translate_dpo" / "translated"
        / f"{language_code}_translated.jsonl"
    )
    if not translated_path.exists():
        raise FileNotFoundError(
            f"Translated data not found: {translated_path}. "
            "Run translate_with_gemma.py first."
        )

    records = _load_jsonl(translated_path)
    print(f"[{language_code}] Loaded {len(records)} translated records.")

    # --- Build SFT dataset --------------------------------------------------
    prompts = get_translate_dpo_prompts(language_code)
    native_sp = prompts["native_system_prompt"]

    # Load tokenizer to format prompts via chat template
    model_name = config["models"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_records = []
    skipped_empty = 0
    skipped_correctness = 0

    for r in records:
        translated = r.get("translated_response", "")
        if not translated.strip():
            skipped_empty += 1
            continue

        # --- Mode 2: correctness filter -------------------------------------
        if mode == 2:
            english_answer = r.get("english_answer", "")
            gold_answer = r.get("gold_answer", "")
            if not (english_answer and english_answer == gold_answer):
                skipped_correctness += 1
                continue

        # --- Strip <think> tags ---------------------------------------------
        response = _strip_think_tags(translated)
        if not response:
            skipped_empty += 1
            continue

        # Format the full conversation using chat template
        messages = [
            {"role": "system", "content": native_sp},
            {"role": "user", "content": r["native_question"]},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        sft_records.append({"text": text})

    if skipped_empty:
        print(f"[{language_code}] Skipped {skipped_empty} records (empty response).")
    if skipped_correctness:
        print(f"[{language_code}] Filtered out {skipped_correctness} records "
              f"(correctness filter: kept only EN-correct).")
    print(f"[{language_code}] mode={mode}: {len(sft_records)} SFT examples prepared.")

    dataset = Dataset.from_list(sft_records)

    # --- Save SFT data for reproducibility ----------------------------------
    mode_label = f"mode{mode}"
    pairs_dir = PROJECT_ROOT / "data" / "translate_sft" / f"mode_{mode}"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pairs_dir / f"{language_code}_sft_{mode_label}.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{language_code}] SFT data saved → {pairs_path}")

    # --- Model --------------------------------------------------------------
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model: {model_name} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if full_finetune:
        # Full fine-tuning: train all parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning: {trainable}/{total} parameters trainable "
              f"({100 * trainable / total:.2f}%)")
    else:
        # LoRA for memory efficiency
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # --- SFT config ---------------------------------------------------------
    dpo_params = config["dpo"]  # reuse same hyperparam section
    ft_suffix = "_full" if full_finetune else ""
    output_dir = output_base / f"sft_{language_code}_{mode_label}{ft_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=float(dpo_params["learning_rate"]),
        num_train_epochs=int(dpo_params["num_train_epochs"]),
        per_device_train_batch_size=int(dpo_params["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(dpo_params["gradient_accumulation_steps"]),
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
        dataset_text_field="text",
    )

    # --- Train --------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"[{language_code}] Starting SFT training (mode={mode}) …")
    train_result = trainer.train()

    # Save model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[{language_code}] Model saved → {output_dir}")

    # Save training log
    log_path = output_dir / "training_log.json"
    log_data = {
        "experiment": "translate_sft",
        "language": language_code,
        "mode": mode,
        "full_finetune": full_finetune,
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics,
        "num_examples": len(sft_records),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"[{language_code}] Training log saved → {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Translate-SFT training (LoRA or full fine-tuning)."
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (e.g. JA_JP). If omitted, train all.",
    )
    parser.add_argument(
        "--mode", type=int, required=True, choices=[1, 2],
        help="Mode 1 = all data, Mode 2 = correct-only filtered.",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "translate_dpo_config.yaml"),
    )
    parser.add_argument(
        "--full_finetune", action="store_true",
        help="Full fine-tuning (no LoRA). Trains all model parameters.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_base = PROJECT_ROOT / "outputs-translate_sft"

    languages = (
        [args.language] if args.language
        else [l["code"] for l in config["languages"]]
    )
    for lang in languages:
        train_translate_sft(lang, args.mode, config, output_base,
                            full_finetune=args.full_finetune)


if __name__ == "__main__":
    main()
