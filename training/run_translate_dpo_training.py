#!/usr/bin/env python3
"""
DPO training for the Translate-DPO experiment.

Trains a Qwen3-1.7B model with LoRA using Direct Preference Optimisation.
For each example:
  - **Chosen** = English response translated to native language
  - **Rejected** = Directly-generated native-language response

The intuition is that models reason better in English, so translated English
reasoning traces should be higher quality than direct native-language traces.

Usage::

    python training/run_translate_dpo_training.py --language JA_JP
    python training/run_translate_dpo_training.py              # all languages
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_translate_dpo(
    language_code: str,
    config: dict,
    output_base: Path,
) -> None:
    """Run DPO training for one language.

    Args:
        language_code: E.g. ``"JA_JP"``.
        config: Parsed translate-DPO config dict.
        output_base: Root directory for model checkpoints.
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
    print(f"[{language_code}] Loaded {len(records)} translated pairs.")

    # --- Build DPO dataset --------------------------------------------------
    # Use the native system prompt as the DPO conditioning context, matching
    # the deployment setting (Setting 2).
    prompts = get_translate_dpo_prompts(language_code)
    native_sp = prompts["native_system_prompt"]

    # Load tokenizer first to format prompts via chat template
    model_name = config["models"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_records = []
    skipped = 0
    for r in records:
        translated = r.get("translated_response", "")
        native = r.get("native_response", "")
        if not translated.strip() or not native.strip():
            skipped += 1
            continue

        # Format the prompt (system + user) using the chat template.
        # add_generation_prompt=True appends the assistant header so the
        # chosen/rejected text is treated as the model's response.
        prompt_messages = [
            {"role": "system", "content": native_sp},
            {"role": "user", "content": r["native_question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        dpo_records.append({
            "prompt": prompt_text,
            "chosen": translated,
            "rejected": native,
        })

    if skipped:
        print(f"[{language_code}] Skipped {skipped} records (empty response).")
    print(f"[{language_code}] {len(dpo_records)} DPO pairs prepared.")

    dataset = Dataset.from_list(dpo_records)

    # --- Save DPO pairs for reproducibility ---------------------------------
    pairs_dir = PROJECT_ROOT / "data" / "translate_dpo" / "dpo_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pairs_dir / f"{language_code}_dpo_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{language_code}] DPO pairs saved → {pairs_path}")

    # --- Model --------------------------------------------------------------
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model: {model_name} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

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

    # --- DPO config ---------------------------------------------------------
    dpo_params = config["dpo"]
    output_dir = output_base / f"dpo_{language_code}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(dpo_params["learning_rate"]),
        beta=float(dpo_params["beta"]),
        num_train_epochs=int(dpo_params["num_train_epochs"]),
        per_device_train_batch_size=int(dpo_params["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(dpo_params["gradient_accumulation_steps"]),
        max_length=int(dpo_params["max_length"]),
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )

    # --- Train --------------------------------------------------------------
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"[{language_code}] Starting DPO training …")
    train_result = trainer.train()

    # Save model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[{language_code}] Model saved → {output_dir}")

    # Save training log
    log_path = output_dir / "training_log.json"
    log_data = {
        "experiment": "translate_dpo",
        "language": language_code,
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics,
        "num_examples": len(dpo_records),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"[{language_code}] Training log saved → {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Translate-DPO training."
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (e.g. JA_JP). If omitted, train all.",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "translate_dpo_config.yaml"),
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_base = PROJECT_ROOT / "outputs-translate_dpo"

    languages = (
        [args.language] if args.language
        else [l["code"] for l in config["languages"]]
    )
    for lang in languages:
        train_translate_dpo(lang, config, output_base)


if __name__ == "__main__":
    main()
