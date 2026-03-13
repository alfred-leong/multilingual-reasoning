#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser("Run DPO Training for Exp_v2")
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["ja", "bn", "sw"], required=True)
    parser.add_argument("--filter", choices=["filter", "no-filter"], required=True)
    parser.add_argument("--full_finetune", action="store_true")
    args = parser.parse_args()

    model_name = "Qwen/Qwen3-1.7B"
    data_path = PROJECT_ROOT / "data" / "exp_v2" / "qwen3-1_7b" / args.dataset / "dpo" / f"train_{args.filter}_{args.language}.jsonl"

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

    dpo_records = {"prompt": [], "chosen": [], "rejected": []}
    for r in records:
        msgs = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["question"]},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Assuming Qwen's eos token format requires the sequence to end with it to compute valid logprobs
        chosen = r["chosen"] + tokenizer.eos_token
        rejected = r["rejected"] + tokenizer.eos_token
        
        dpo_records["prompt"].append(prompt)
        dpo_records["chosen"].append(chosen)
        dpo_records["rejected"].append(rejected)

    dataset = Dataset.from_dict(dpo_records)

    # Output details
    prefix = f"dpo_{args.filter}_full" if args.full_finetune else f"dpo_{args.filter}_lora"
    run_name = f"{prefix}_{args.dataset}_{args.language}"
    output_dir = Path("/external1/alfred/models/ml-reasoning-exp_v2") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="ml-reasoning", name=run_name, group=args.dataset, job_type="dpo")

    # Load model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True) if args.full_finetune else None

    if not args.full_finetune:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="wandb",
        max_length=4096,
        max_prompt_length=2048,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    wandb.finish()
    print(f"Saved DPO model to {output_dir}")

if __name__ == "__main__":
    main()
