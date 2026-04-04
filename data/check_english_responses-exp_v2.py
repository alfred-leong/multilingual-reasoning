#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--model", type=str, default="qwen3-1_7b",
                        help="Short name of the source model (e.g. qwen3-8b)")
    args = parser.parse_args()

    input_file = PROJECT_ROOT / "data" / "exp_v2" / args.model / args.dataset / "english" / "english_responses.jsonl"

    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    correct = 0
    empty = 0
    total = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            ans = record.get("english_answer", "").strip()
            gt = record.get("ground_truth", "").strip()

            if not ans:
                empty += 1
            elif ans == gt:
                correct += 1

    print(f"--- Dataset: {args.dataset} ---")
    print(f"Total English responses: {total}")
    print(f"Empty responses: {empty} ({empty/total*100:.2f}%)")
    print(f"Correct responses: {correct} ({correct/total*100:.2f}%)")

if __name__ == "__main__":
    main()
