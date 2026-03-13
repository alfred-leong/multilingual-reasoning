#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MMMLU_LANGS = {
    "en": {"path": "cais/mmlu", "name": "all"},
    "bn": {"path": "openai/MMMLU", "name": "BN_BD"},
    "ja": {"path": "openai/MMMLU", "name": "JA_JP"},
    "sw": {"path": "openai/MMMLU", "name": "SW_KE"},
}
MGSM_LANGS = {
    "en": {"path": "juletxara/mgsm", "name": "en"},
    "bn": {"path": "juletxara/mgsm", "name": "bn"},
    "ja": {"path": "juletxara/mgsm", "name": "ja"},
    "sw": {"path": "juletxara/mgsm", "name": "sw"},
}

def set_seed(seed=42):
    random.seed(seed)

def load_data(dataset_name):
    langs_dict = MMMLU_LANGS if dataset_name == "mmmlu" else MGSM_LANGS
    
    data_by_lang = {}
    total_len = None
    
    for lang, info in langs_dict.items():
        print(f"Loading {dataset_name} for lang {lang}...")
        
        if dataset_name == "mgsm":
            # Load local TSV
            filepath = f"/tier1/home/lweilun/multilingual-reasoning/data/mgsm/mgsm_{info['name']}.tsv"
            list_data = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        list_data.append({"question": parts[0], "answer": parts[1]})
        else:
            ds = load_dataset(info["path"], info["name"], trust_remote_code=True)
            if "test" in ds:
                ds = ds["test"]
            elif "train" in ds:
                ds = ds["train"]
            list_data = [dict(row) for row in ds]
        if total_len is None:
            total_len = len(list_data)
        else:
            if dataset_name == "mmmlu" and len(list_data) != total_len:
                print(f"Warning: {lang} length {len(list_data)} mismatch {total_len}")
                # We can truncate or take min just in case
                total_len = min(total_len, len(list_data))
                
        data_by_lang[lang] = list_data

    # Trim to exact same length just in case
    for lang in langs_dict.keys():
        data_by_lang[lang] = data_by_lang[lang][:total_len]
        
    return data_by_lang, total_len

def create_and_save_splits(dataset_name, data_by_lang, total_len, split_ratio=0.8):
    indices = list(range(total_len))
    random.shuffle(indices)
    
    train_size = int(total_len * split_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    out_dir = PROJECT_ROOT / "data" / "exp_v2" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train and test with question_index injected
    for lang in data_by_lang.keys():
        train_file = out_dir / f"train_{lang}.jsonl"
        test_file = out_dir / f"test_{lang}.jsonl"
        
        # We also inject a 'question_index' into the dictionary. 
        # Wait, MMMLU MMLU schemas differ. We can just add "question_index" directly.
        
        with open(train_file, "w", encoding="utf-8") as f:
            for i in train_indices:
                record = data_by_lang[lang][i].copy()
                record["question_index"] = i
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        with open(test_file, "w", encoding="utf-8") as f:
            for i in test_indices:
                record = data_by_lang[lang][i].copy()
                record["question_index"] = i
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print(f"Saved {lang} -> train: {train_size}, test: {total_len - train_size}")

def main():
    parser = argparse.ArgumentParser("Split dataset 80/20 per language aligned.")
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_by_lang, total_len = load_data(args.dataset)
    print(f"Loaded {args.dataset}. Total items per lang: {total_len}")
    
    create_and_save_splits(args.dataset, data_by_lang, total_len)

if __name__ == "__main__":
    main()
