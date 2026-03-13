import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser("Check native responses against ground truth.")
    parser.add_argument("--dataset", type=str, choices=["mmmlu", "mgsm"], required=True)
    parser.add_argument("--language", type=str, choices=["ja", "bn", "sw"], required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "data" / "exp_v2" / "qwen3-1_7b" / args.dataset / "native" / f"native_{args.language}_responses.jsonl"

    if not input_file.exists():
        print(f"Error: {input_file} does not exist.")
        return

    correct = 0
    empty = 0
    total = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            data = json.loads(line)
            native_answer = data.get("native_answer", "").strip()
            gold = data.get("ground_truth", "").strip()

            if not native_answer:
                empty += 1
            elif native_answer == gold:
                correct += 1

    print(f"[{args.dataset.upper()} - {args.language.upper()}] Native Generations:")
    print(f"  Total    : {total}")
    print(f"  Correct  : {correct} ({correct/total*100:.1f}%)")
    print(f"  Empty    : {empty} ({empty/total*100:.1f}%)")

if __name__ == "__main__":
    main()
