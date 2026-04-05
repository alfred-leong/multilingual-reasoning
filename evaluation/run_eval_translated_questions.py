#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import torch
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
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--subset", type=str, default=None, help="Subset suffix, e.g. '100' loads test_{lang}_100.jsonl")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model. If not provided, uses the base model without fine-tuning.")
    parser.add_argument("--model_cfg_name", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    args = parser.parse_args()
    setup_seed(args.seed)

    # Load test data
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

    # Load prompts
    # lang_code = LANGUAGE_CODES[args.language]
    lang_code = "EN"    # evaluate from English-translated prompts
    prompts = get_translate_dpo_prompts(lang_code)
    system_prompt = prompts[f"native_system_prompt_{args.dataset}"]
    thinking_prefix = prompts["thinking_prefix"]
    print(f"System prompt:\n{system_prompt}\n")
    print(f"Thinking prefix:\n{thinking_prefix}\n")

    # Load model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_name = "Qwen/Qwen3-1.7B"

    if args.model_cfg_name != "base":
        print(f"Loading model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if "lora" in args.model_cfg_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype, device_map="auto", trust_remote_code=True)
            model = PeftModel.from_pretrained(base_model, args.model_path)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype, device_map="auto", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype, device_map="auto", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = len(records)
    results = []

    for idx, row in enumerate(tqdm(records, desc=f"Evaluating {args.dataset} ({args.language}-en questions)")):
        if args.language != "en":
            question = row["self_translated_english_question"]
            gold = row["ground_truth"]
        else:
            if args.dataset == "mmmlu":
                question = _format_mmmlu_question(row)
                gold = row.get("Answer", "").strip()
                if not gold and "answer" in row: # default to english fallback if missing
                    # some datasets have answer instead of Answer
                    ans_val = row["answer"]
                    if isinstance(ans_val, int):
                        gold = chr(65 + ans_val)
                    else:
                        gold = str(ans_val)
            else:
                question = row["question"]
                gold = row["answer"]
        gold = gold.strip()

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        print(msgs)
        response = generate_response(model, tokenizer, msgs, thinking_prefix)
        final_text = response.split("</think>")[-1] if "</think>" in response else response

        pred = _extract_mgsm_answer(final_text) if args.dataset == "mgsm" else _extract_mmmlu_answer(final_text)
        is_correct = (pred == gold)
        print(f"Prediction: {pred}, Gold: {gold}, Match: {is_correct}")
        print()

        if is_correct:
            correct += 1

        results.append({
            "question_index": row.get("question_index", idx),
            "message": msgs,
            "response": response,
            "pred": pred,
            "gold": gold,
            "correct": is_correct
        })
    
    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc:.2%} ({correct}/{total})")

    out_file = PROJECT_ROOT / "outputs-self_translate_questions" / f"eval_{args.dataset}_{args.language}-en{suffix}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "language": args.language,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "results": results
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
