#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.translate_dpo_prompts import get_translate_dpo_prompts

LANGUAGE_CODES = {
    "bn": "BN_BD",
    "ja": "JA_JP",
    "sw": "SW_KE"
}

def main():
    parser = argparse.ArgumentParser("Prepare SFT and DPO training datasets.")
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["ja", "bn", "sw"], required=True)
    args = parser.parse_args()

    qwen_dir = PROJECT_ROOT / "data" / "exp_v2" / "qwen3-1_7b" / args.dataset
    
    trans_file = qwen_dir / "translated" / f"translated_{args.language}_responses.jsonl"
    native_file = qwen_dir / "native" / f"native_{args.language}_responses.jsonl"

    if not trans_file.exists() or not native_file.exists():
        print(f"Error: Missing one of the required response files for {args.language} in {args.dataset}.")
        return

    trans_data = {}
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            trans_data[d["question_index"]] = d

    native_data = {}
    with open(native_file, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            native_data[d["question_index"]] = d

    # load the system prompts
    lang_code = LANGUAGE_CODES[args.language]
    prompts = get_translate_dpo_prompts(lang_code)
    system_prompt = prompts[f"native_system_prompt_{args.dataset}"]

    # output directories
    sft_dir = qwen_dir / "sft"
    dpo_dir = qwen_dir / "dpo"
    sft_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)

    sft_no_filter_path = sft_dir / f"train_no-filter_{args.language}.jsonl"
    sft_filter_path = sft_dir / f"train_filter_{args.language}.jsonl"
    dpo_no_filter_path = dpo_dir / f"train_no-filter_{args.language}.jsonl"
    dpo_filter_path = dpo_dir / f"train_filter_{args.language}.jsonl"

    count_sft_nofilt = 0
    count_sft_filt = 0
    count_dpo_nofilt = 0
    count_dpo_filt = 0

    with open(sft_no_filter_path, "w", encoding="utf-8") as f_sft_nofilt, \
         open(sft_filter_path, "w", encoding="utf-8") as f_sft_filt, \
         open(dpo_no_filter_path, "w", encoding="utf-8") as f_dpo_nofilt, \
         open(dpo_filter_path, "w", encoding="utf-8") as f_dpo_filt:
         
        for qid, t_d in trans_data.items():
            if qid not in native_data:
                continue
            n_d = native_data[qid]
            
            ground_truth = str(t_d.get("ground_truth", "")).strip()
            t_ans = str(t_d.get("translated_answer", "")).strip()
            n_ans = str(n_d.get("native_answer", "")).strip()

            native_question = n_d["native_question"]
            trans_resp = t_d["translated_response"]
            native_resp = n_d["native_response"]

            # Common keys for training data point
            base_dpo = {
                "system": system_prompt,
                "question": native_question,
                "chosen": trans_resp,
                "rejected": native_resp
            }
            base_sft = {
                "system": system_prompt,
                "question": native_question,
                "response": trans_resp
            }

            # writes
            f_sft_nofilt.write(json.dumps(base_sft, ensure_ascii=False) + "\n")
            count_sft_nofilt += 1
            f_dpo_nofilt.write(json.dumps(base_dpo, ensure_ascii=False) + "\n")
            count_dpo_nofilt += 1

            if t_ans == ground_truth:
                f_sft_filt.write(json.dumps(base_sft, ensure_ascii=False) + "\n")
                count_sft_filt += 1
                if n_ans != ground_truth:
                    f_dpo_filt.write(json.dumps(base_dpo, ensure_ascii=False) + "\n")
                    count_dpo_filt += 1

    print(f"[{args.dataset} - {args.language}] Done creating SFT and DPO splits")
    print(f"  - SFT (no filter): {count_sft_nofilt}")
    print(f"  - SFT (filter):    {count_sft_filt}")
    print(f"  - DPO (no filter): {count_dpo_nofilt}")
    print(f"  - DPO (filter):    {count_dpo_filt}")

if __name__ == "__main__":
    main()
