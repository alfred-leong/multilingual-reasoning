#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser("Run SFT Training for Exp_v2")
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["ja", "bn", "sw"], required=True)
    parser.add_argument("--filter", choices=["filter", "no-filter"], required=True)
    parser.add_argument("--full_finetune", action="store_true")
    args = parser.parse_args()

    model_name = "Qwen/Qwen3-1.7B"
    data_path = PROJECT_ROOT / "data" / "exp_v2" / "qwen3-1_7b" / args.dataset / "sft" / f"train_{args.filter}_{args.language}.jsonl"

    if not data_path.exists():
        print(f"Dataset {data_path} not found.")
        return

    # Load data
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_records = []
    for r in records:
        msgs = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["response"]}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        sft_records.append({"text": text})

    dataset = Dataset.from_list(sft_records)

    # Output details
    prefix = f"sft_{args.filter}_full" if args.full_finetune else f"sft_{args.filter}_lora"
    run_name = f"{prefix}_{args.dataset}_{args.language}"
    output_dir = Path("/external1/alfred/models/ml-reasoning-exp_v2") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="ml-reasoning", name=run_name, group=args.dataset, job_type="sft")

    # Load model
    # device_map="auto" conflicts with DeepSpeed/accelerate multi-GPU management;
    # only use it for single-GPU LoRA runs.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if not args.full_finetune else None
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)

    peft_config = None
    if not args.full_finetune:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="wandb",
        dataset_text_field="text",
        max_length=12288,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    wandb.finish()
    print(f"Saved SFT model to {output_dir}")

if __name__ == "__main__":
    main()
