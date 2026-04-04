#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["bn", "ja", "sw"], required=True)
    parser.add_argument("--model", type=str, default="qwen3-1_7b",
                        help="Short name of the source model (e.g. qwen3-8b)")
    args = parser.parse_args()

    input_file = PROJECT_ROOT / "data" / "exp_v2" / args.model / args.dataset / "translated_gemma-27b" / f"translated_{args.language}_responses.jsonl"

    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    match_count = 0
    total = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            eng_ans = record.get("english_answer", "").strip()
            trans_ans = record.get("translated_answer", "").strip()

            if eng_ans == trans_ans:
                match_count += 1

    print(f"--- Dataset: {args.dataset} | Language: {args.language} ---")
    print(f"Total translated responses: {total}")
    if total > 0:
        print(f"Matches (english_answer == translated_answer): {match_count} ({match_count/total*100:.2f}%)")

if __name__ == "__main__":
    main()
