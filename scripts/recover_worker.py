#!/usr/bin/env python3
"""Recover a single failed generation worker and optionally merge all parts.

The vLLM servers must be running before calling this script. Start them with::

    bash scripts/start_vllm_servers.sh [GPU_SMALL] [GPU_LARGE]

Usage examples::

    # Re-run worker 1 for SW_KE (rows 4681–9361):
    python scripts/recover_worker.py \\
        --language SW_KE --rank 1 --start_row 4681 --end_row 9362

    # Re-run and then immediately merge all parts into the final JSONL:
    python scripts/recover_worker.py \\
        --language SW_KE --rank 1 --start_row 4681 --end_row 9362 --merge

    # Just merge existing part files without re-generation:
    python scripts/recover_worker.py --language SW_KE --merge_only
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from datasets import load_dataset

from data.generate_preference_data import (
    VLLM_PORTS,
    _generation_worker,
)
from prompts.system_prompts import get_prompts


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_worker(
    language_code: str,
    rank: int,
    start_row: int,
    end_row: int,
    config: dict,
) -> None:
    """Re-run a single generation worker for the given row range."""
    output_dir = PROJECT_ROOT / "data" / "raw"
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if language_code not in VLLM_PORTS:
        raise ValueError(f"Unknown language code {language_code!r}. Known: {list(VLLM_PORTS)}")
    ports = VLLM_PORTS[language_code]

    prompts_info = get_prompts(language_code, thinking_mode="native_only")
    system_prompt = prompts_info["system_prompt"]
    cot_instruction = prompts_info["cot_instruction"]

    print(f"Loading {language_code} dataset …")
    ds = load_dataset(
        config["datasets"]["preference_source"],
        language_code,
        split="test",
    )
    total = len(ds)
    end_row = min(end_row, total)
    print(f"Dataset has {total} rows; slicing [{start_row}:{end_row}] ({end_row - start_row} rows)")

    rows = [dict(ds[i]) for i in range(start_row, end_row)]

    tmp_path = str(output_dir / f"{language_code}_part{rank}.jsonl")
    print(f"Writing to {tmp_path}")
    print(f"Using vLLM ports: small={ports['small']}  large={ports['large']}")

    # Run directly in this process (no extra spawning needed)
    _generation_worker(
        rank=rank,
        rows=rows,
        language_code=language_code,
        config=config,
        tmp_path=tmp_path,
        system_prompt=system_prompt,
        cot_instruction=cot_instruction,
        log_dir=str(log_dir),
        small_port=ports["small"],
        large_port=ports["large"],
    )
    print(f"Done. Wrote {tmp_path}")


def merge_parts(language_code: str, output_dir: Path, n_workers: int = 3) -> None:
    """Merge all part files in rank order into the final pairs JSONL.

    Deletes temp files after a successful merge.
    """
    out_path = output_dir / f"{language_code}_pairs.jsonl"
    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for rank in range(n_workers):
            part = output_dir / f"{language_code}_part{rank}.jsonl"
            if not part.exists():
                raise FileNotFoundError(
                    f"Part file missing: {part}. Run recovery for rank {rank} first."
                )
            lines = 0
            with open(part, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    lines += 1
                    total += 1
            print(f"  rank {rank}: {lines} lines from {part}")
            part.unlink()
    print(f"Merged {total} pairs → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover a single failed generation worker (API-based)."
    )
    parser.add_argument("--language", required=True, help="Language code, e.g. SW_KE")
    parser.add_argument("--rank", type=int, default=None,
                        help="Worker rank to re-run (0, 1, or 2)")
    parser.add_argument("--start_row", type=int, default=None,
                        help="Inclusive start row index in the full dataset")
    parser.add_argument("--end_row", type=int, default=None,
                        help="Exclusive end row index in the full dataset")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--n_workers", type=int, default=3,
                        help="Total number of workers used for this language (default 3)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all parts into final JSONL after re-generation")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip re-generation; just merge existing part files")
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = PROJECT_ROOT / "data" / "raw"

    if not args.merge_only:
        if args.rank is None or args.start_row is None or args.end_row is None:
            parser.error("--rank, --start_row, and --end_row are required unless --merge_only")
        run_worker(args.language, args.rank, args.start_row, args.end_row, config)

    if args.merge or args.merge_only:
        print(f"\nMerging {args.n_workers} parts for {args.language} …")
        merge_parts(output_dir=output_dir, language_code=args.language,
                    n_workers=args.n_workers)


if __name__ == "__main__":
    main()
