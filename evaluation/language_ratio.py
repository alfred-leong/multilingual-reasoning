#!/usr/bin/env python3
"""
Batch language-ratio analysis for evaluation results.

Reads one or more evaluation JSON files produced by ``run_mgsm_eval.py``
and produces a consolidated language-ratio report (CSV + terminal output).

Usage::

    # Analyse a single evaluation result:
    python evaluation/language_ratio.py \\
        --input outputs/eval_JA_JP_base_natural.json

    # Analyse all evaluation results in the outputs/ directory:
    python evaluation/language_ratio.py --all
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_eval_results(path: Path) -> dict:
    """Load a single evaluation result JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyse_file(path: Path) -> dict:
    """Compute aggregate language ratios from a single eval file.

    Args:
        path: Path to an evaluation JSON file.

    Returns:
        A summary dict with language, model_type, thinking_mode, accuracy,
        mean_target_ratio, mean_english_ratio, and mean_other_ratio.
    """
    data = _load_eval_results(path)
    examples = data.get("per_example_results", [])

    if not examples:
        return {
            "file": path.name,
            "language": data.get("language"),
            "model_type": data.get("model_type"),
            "thinking_mode": data.get("thinking_mode"),
            "dpo_mode": data.get("dpo_mode"),
            "accuracy": data.get("accuracy", 0.0),
            "mean_target_ratio": 0.0,
            "mean_english_ratio": 0.0,
            "mean_other_ratio": 0.0,
            "num_examples": 0,
        }

    target_ratios = [e.get("target_ratio", 0.0) for e in examples]
    english_ratios = [e.get("english_ratio", 0.0) for e in examples]
    other_ratios = [e.get("other_ratio", 0.0) for e in examples]

    return {
        "file": path.name,
        "language": data.get("language"),
        "model_type": data.get("model_type"),
        "thinking_mode": data.get("thinking_mode"),
        "dpo_mode": data.get("dpo_mode"),
        "accuracy": data.get("accuracy", 0.0),
        "mean_target_ratio": sum(target_ratios) / len(target_ratios),
        "mean_english_ratio": sum(english_ratios) / len(english_ratios),
        "mean_other_ratio": sum(other_ratios) / len(other_ratios),
        "num_examples": len(examples),
    }


def main() -> None:
    """Parse arguments and produce language-ratio report."""
    parser = argparse.ArgumentParser(
        description="Aggregate language-ratio stats from evaluation results."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a single eval JSON file.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Analyse all eval_*.json files in outputs/.",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Optional path to save a CSV summary.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.input:
        paths.append(Path(args.input))
    elif args.all:
        pattern = str(PROJECT_ROOT / "outputs" / "eval_*.json")
        paths = sorted(Path(p) for p in glob.glob(pattern))
    else:
        parser.print_help()
        sys.exit(1)

    if not paths:
        print("No evaluation files found.")
        sys.exit(0)

    rows = [analyse_file(p) for p in paths]
    df = pd.DataFrame(rows)

    # Print summary
    print("\n" + "=" * 90)
    print("LANGUAGE-RATIO REPORT")
    print("=" * 90)
    display_cols = [
        "language", "model_type", "dpo_mode", "thinking_mode",
        "accuracy", "mean_target_ratio", "mean_english_ratio",
        "mean_other_ratio", "num_examples",
    ]
    print(df[display_cols].to_string(index=False, float_format="%.3f"))
    print("=" * 90)

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
