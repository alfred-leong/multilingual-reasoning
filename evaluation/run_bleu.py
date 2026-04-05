#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import torch
import evaluate
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.translate_dpo_prompts import get_translate_dpo_prompts

_ANSWER_RE = re.compile(r"\b([A-D])\b")
_BOXED_RE = re.compile(r"\\boxed{([^}]*)}")

LANGUAGE_CODES = {
    "bn": "BN_BD",
    "ja": "JA_JP",
    "sw": "SW_KE",
    "en": "EN"
}

import torch
import numpy as np
import random
def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _extract_mmmlu_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else ""

def _extract_mgsm_answer(text: str) -> str:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""

# def _format_mmmlu_question(row: dict) -> str:
#     q = row["Question"]
#     choices = "\n".join(f"{letter}. {row[letter]}" for letter in ("A", "B", "C", "D"))
#     return f"{q}\n\n{choices}"

def _format_mmmlu_question(row: dict) -> str:
    q = row["Question"]  # CAIS MMLU has 'question', OpenAI MMMLU has 'Question'
    choices = "\n".join(
        f"{letter}. {row[letter]}" for letter in ("A", "B", "C", "D")
    )
    return f"{q}\n\n{choices}"

def generate_response(model, tokenizer, messages, thinking_prefix=""):
    if thinking_prefix:
        msgs = list(messages) + [{"role": "assistant", "content": f"<think>\n{thinking_prefix}"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    else:
        msgs = list(messages) + [{"role": "assistant", "content": "<think>\n"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    
    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    if thinking_prefix:
        return f"<think>\n{thinking_prefix}\n{gen_text}"
    else:
        return f"<think>\n{gen_text}"

def main():
    parser = argparse.ArgumentParser("Evaluate trained models on 20% test set.")
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["ja", "bn", "sw", "en"], required=True)
    parser.add_argument("--subset", type=str, default=None, help="Subset suffix, e.g. '100' loads test_{lang}_100.jsonl")
    parser.add_argument("--model_cfg_name", type=str, default="base")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Only used for logging purposes here since we're evaluating fixed outputs.")
    args = parser.parse_args()
    setup_seed(args.seed)

    # Load test data
    # suffix = f"_{args.model_cfg_name}" if args.model_cfg_name != "base" else ""
    suffix = f"_{args.model_cfg_name}" if args.model_cfg_name else ""
    suffix += f"_{args.subset}" if args.subset else ""
    suffix += f"_seed{args.seed}"
    if args.language != "en":
        test_file = PROJECT_ROOT / "data" / "exp_v3" / args.dataset / f"test_{args.language}-en{suffix}" / f"test_{args.language}-en.jsonl"
    else:
        test_file = PROJECT_ROOT / "data" / "exp_v3" / args.dataset / f"test_{args.language}{suffix}.jsonl"

    print(f"Loading test data from {test_file}...")
    if not test_file.exists():
        print(f"Test file {test_file} not found.")
        return

    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    predictions, references = [], []
    for x in records:
        predictions.append(x["self_translated_english_question"])
        references.append(x["expert_translated_english_question"])
        print("Prediction:", x["self_translated_english_question"])
        print("Reference:", x["expert_translated_english_question"])
        print()

    print("Ready to calculate BLEU scores")
    bleu_calculator = evaluate.load("bleu")
    bleu_1gram = bleu_calculator.compute(predictions=predictions, references=references, max_order=1)
    bleu_2gram = bleu_calculator.compute(predictions=predictions, references=references, max_order=2)
    bleu_4gram = bleu_calculator.compute(predictions=predictions, references=references, max_order=4)
    print(f"BLEU-1: {bleu_1gram['bleu']:.4f}")
    print(f"BLEU-2: {bleu_2gram['bleu']:.4f}")
    print(f"BLEU-4: {bleu_4gram['bleu']:.4f}")
    output_file = PROJECT_ROOT / "outputs-self_translate_questions" / f"bleu_scores_{args.dataset}_{args.language}-en{suffix}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "language": args.language,
            "bleu_1gram": bleu_1gram["bleu"],
            "bleu_2gram": bleu_2gram["bleu"],
            "bleu_4gram": bleu_4gram["bleu"]
        }, f, indent=2)
    print(f"Saved BLEU scores to {output_file}")


if __name__ == "__main__":
    main()
