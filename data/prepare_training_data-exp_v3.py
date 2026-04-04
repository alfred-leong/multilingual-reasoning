#!/usr/bin/env python3
"""Prepare SFT training data for qwen3-8b / mmmlu from translated_gemma-27b responses.

Analogous to prepare_training_data-exp_v2.py but adapted for qwen3-8b where only
translated responses exist (no native responses).  Native-language questions are
looked up from the mmmlu train split by question_index.
"""
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
    "sw": "SW_KE",
}


def _format_mmmlu_question(row: dict) -> str:
    q = row["Question"]
    choices = "\n".join(f"{letter}. {row[letter]}" for letter in ("A", "B", "C", "D"))
    return f"{q}\n\n{choices}"


def main():
    parser = argparse.ArgumentParser("Prepare SFT training data for qwen3-8b / mmmlu.")
    parser.add_argument("--language", type=str, choices=["ja", "bn", "sw"], required=True)
    args = parser.parse_args()

    # Paths
    translated_file = (
        PROJECT_ROOT / "data" / "exp_v2" / "qwen3-8b" / "mmmlu"
        / "translated_gemma-27b" / f"translated_{args.language}_responses.jsonl"
    )
    train_file = PROJECT_ROOT / "data" / "exp_v2" / "mmmlu" / f"train_{args.language}.jsonl"

    if not translated_file.exists():
        print(f"Error: Translated responses not found: {translated_file}")
        return
    if not train_file.exists():
        print(f"Error: Train split not found: {train_file}")
        return

    # Load native-language questions from the mmmlu train split, keyed by question_index
    native_questions = {}
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            native_questions[row["question_index"]] = _format_mmmlu_question(row)

    # Load translated responses
    trans_data = []
    with open(translated_file, "r", encoding="utf-8") as f:
        for line in f:
            trans_data.append(json.loads(line))

    # System prompt
    lang_code = LANGUAGE_CODES[args.language]
    prompts = get_translate_dpo_prompts(lang_code)
    system_prompt = prompts["native_system_prompt_mmmlu"]

    # Output directory (same location the training script expects)
    sft_dir = PROJECT_ROOT / "data" / "exp_v2" / "qwen3-8b" / "mmmlu" / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    sft_no_filter_path = sft_dir / f"train_no-filter_{args.language}.jsonl"
    sft_filter_path = sft_dir / f"train_filter_{args.language}.jsonl"

    count_nofilt = 0
    count_filt = 0
    skipped = 0

    with open(sft_no_filter_path, "w", encoding="utf-8") as f_nofilt, \
         open(sft_filter_path, "w", encoding="utf-8") as f_filt:

        for t_d in trans_data:
            qid = t_d["question_index"]
            if qid not in native_questions:
                skipped += 1
                continue

            question = native_questions[qid]
            response = t_d["translated_response"]
            ground_truth = str(t_d.get("ground_truth", "")).strip()
            translated_answer = str(t_d.get("translated_answer", "")).strip()

            record = {
                "system": system_prompt,
                "question": question,
                "response": response,
            }

            # no-filter: all translated responses
            f_nofilt.write(json.dumps(record, ensure_ascii=False) + "\n")
            count_nofilt += 1

            # filter: only responses where the translated answer matches ground truth
            if translated_answer == ground_truth:
                f_filt.write(json.dumps(record, ensure_ascii=False) + "\n")
                count_filt += 1

    print(f"[mmmlu - {args.language}] Done creating SFT splits")
    print(f"  - SFT (no filter): {count_nofilt}")
    print(f"  - SFT (filter):    {count_filt}")
    if skipped:
        print(f"  - Skipped (no matching question_index in train): {skipped}")


if __name__ == "__main__":
    main()
