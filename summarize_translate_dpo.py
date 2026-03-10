#!/usr/bin/env python3
"""
Summarise accuracy results from outputs-translate_dpo evaluation JSON files.

Only considers files that do NOT end with _run-{i}.json (i.e. the primary
evaluation files, not the per-run replicas).
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR     = Path(__file__).parent
OUTPUT_DIR   = ROOT_DIR / "outputs-translate_dpo"
SFT_DIR      = ROOT_DIR / "outputs-translate_sft"

LANGUAGES = ["BN", "JA", "SW"]

# Language short-code → full country-code folder suffix used in directory names
LANG_TO_CODE = {"BN": "BN_BD", "JA": "JA_JP", "SW": "SW_KE"}

# Mapping from eval file-name suffix → display run name, in desired display order
SUFFIX_TO_RUN = {
    "base":                         "base",
    "dpo":                          "dpo lora",
    "dpo_removethink+filter":       "dpo filtered lora",
    "dpo_removethink+filter_full":  "dpo filtered full",
    "dpo_sft_mode1":                "sft lora",
    "dpo_sft_mode1_full":           "sft full",
    "dpo_sft_mode2":                "sft filtered lora",
    "dpo_sft_mode2_full":           "sft filtered full",
}

RUN_ORDER = list(SUFFIX_TO_RUN.values())

# Map each run name → function that returns the training_log.json Path for a given lang code
# base has no training log.
def _dpo_log(lang_code: str, subdir: str) -> Path:
    return OUTPUT_DIR / subdir.format(lang_code=lang_code) / "training_log.json"

def _sft_log(lang_code: str, subdir: str) -> Path:
    return SFT_DIR / subdir.format(lang_code=lang_code) / "training_log.json"

RUN_TRAINING_LOG: dict[str, callable] = {
    "base":               lambda lc: None,
    "dpo lora":           lambda lc: _dpo_log(lc, "dpo_{lang_code}"),
    "dpo filtered lora":  lambda lc: _dpo_log(lc, "dpo_{lang_code}_removethink+filter"),
    "dpo filtered full":  lambda lc: _dpo_log(lc, "dpo_{lang_code}_removethink+filter_full"),
    "sft lora":           lambda lc: _sft_log(lc, "sft_{lang_code}_mode1"),
    "sft full":           lambda lc: _sft_log(lc, "sft_{lang_code}_mode1_full"),
    "sft filtered lora":  lambda lc: _sft_log(lc, "sft_{lang_code}_mode2"),
    "sft filtered full":  lambda lc: _sft_log(lc, "sft_{lang_code}_mode2_full"),
}

# Regex to detect per-run files we want to SKIP  (e.g. _run-1.json)
RUN_SUFFIX_RE = re.compile(r"_run-\d+\.json$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename(filename: str):
    """
    Parse an eval filename like  eval_BN_BD_dpo_removethink+filter_full.json
    and return (language_short, suffix) or None if it cannot be parsed.

    language_short is one of BN / JA / SW.
    suffix is the part after the language+country code, e.g. "dpo_removethink+filter_full".
    """
    if not filename.startswith("eval_") or not filename.endswith(".json"):
        return None
    inner = filename[len("eval_"):-len(".json")]   # e.g. BN_BD_dpo_removethink+filter_full
    parts = inner.split("_", 2)                    # ['BN', 'BD', 'dpo_removethink+filter_full']
    if len(parts) < 2:
        return None

    lang_short = parts[0]
    if lang_short not in LANGUAGES:
        return None

    suffix = parts[2] if len(parts) > 2 else "base"
    return lang_short, suffix


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_num_examples(run_name: str, lang_code: str) -> int | None:
    """Return training sample count for a run, or None for base."""
    log_fn = RUN_TRAINING_LOG.get(run_name)
    if log_fn is None:
        return None
    log_path = log_fn(lang_code)
    if log_path is None or not log_path.exists():
        return None
    data = load_json(log_path)
    return data.get("num_examples")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Collect results:  {lang: {run_name: accuracy}}
    results: dict[str, dict[str, float]] = {lang: {} for lang in LANGUAGES}

    for filepath in sorted(OUTPUT_DIR.iterdir()):
        filename = filepath.name
        if not filename.endswith(".json"):
            continue
        if RUN_SUFFIX_RE.search(filename):
            continue

        parsed = parse_filename(filename)
        if parsed is None:
            continue

        lang, suffix = parsed
        if suffix not in SUFFIX_TO_RUN:
            print(f"  [WARN] Unrecognised suffix '{suffix}' in {filename} — skipping")
            continue

        run_name = SUFFIX_TO_RUN[suffix]
        results[lang][run_name] = float(load_json(filepath)["accuracy"])

    # ---------------------------------------------------------------------------
    # Determine top-1 / top-2 per language (by delta, excluding base)
    # ---------------------------------------------------------------------------
    top_ranks: dict[str, dict[str, int]] = {}   # lang -> {run_name -> rank (1 or 2)}
    for lang in LANGUAGES:
        lang_results = results[lang]
        base_acc = lang_results.get("base", 0.0)
        deltas = [
            (run, lang_results[run] - base_acc)
            for run in RUN_ORDER
            if run != "base" and run in lang_results
        ]
        deltas.sort(key=lambda x: x[1], reverse=True)
        top_ranks[lang] = {}
        if len(deltas) >= 1:
            top_ranks[lang][deltas[0][0]] = 1
        if len(deltas) >= 2:
            top_ranks[lang][deltas[1][0]] = 2

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------

    col_widths = {
        "lang":     10,
        "run":      22,
        "samples":  18,
        "acc":      10,
        "delta":    15,
        "rank":      6,
    }

    header = (
        f"{'Language':<{col_widths['lang']}}"
        f"{'Run Name':<{col_widths['run']}}"
        f"{'Train Samples':>{col_widths['samples']}}"
        f"{'Accuracy':>{col_widths['acc']}}"
        f"{'Delta vs Base':>{col_widths['delta']}}"
        f"{'Best':>{col_widths['rank']}}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for lang in LANGUAGES:
        lang_results = results[lang]
        lang_code    = LANG_TO_CODE[lang]
        base_acc     = lang_results.get("base")
        ranks        = top_ranks[lang]

        first = True
        for run_name in RUN_ORDER:
            acc = lang_results.get(run_name)
            if acc is None:
                continue

            lang_label   = lang if first else ""
            first        = False

            # Delta
            delta_str = "—"
            if base_acc is not None and run_name != "base":
                delta = acc - base_acc
                sign  = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta:.2%}"

            # Training samples
            num_ex = get_num_examples(run_name, lang_code)
            samples_str = f"{num_ex:,}" if num_ex is not None else "—"

            # Rank badge
            rank = ranks.get(run_name)
            badge = "✔✔" if rank == 1 else ("✔" if rank == 2 else "")

            print(
                f"{lang_label:<{col_widths['lang']}}"
                f"{run_name:<{col_widths['run']}}"
                f"{samples_str:>{col_widths['samples']}}"
                f"{acc:>{col_widths['acc']}.2%}"
                f"{delta_str:>{col_widths['delta']}}"
                f"{badge:>{col_widths['rank']}}"
            )

        print(separator)


if __name__ == "__main__":
    main()
