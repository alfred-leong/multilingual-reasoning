#!/usr/bin/env python3
"""
Prepare a local copy of the TranslateGemma model config for vLLM compatibility.

vLLM 0.16.0 has a bug in its Transformers v4 backward-compatibility code
path: when ``text_config`` has both ``rope_theta`` (a scalar) and
``rope_parameters`` (a nested dict keyed by attention layer type), vLLM
injects ``rope_theta`` as a flat key into the nested ``rope_parameters``
dict.  This breaks ``is_rope_parameters_nested()`` — which checks that
*all* keys belong to ``ALLOWED_ATTENTION_LAYER_TYPES`` — causing vLLM to
treat the dict as flat and then fail because there is no top-level
``rope_type`` key.

**Fix:** This script creates a local model directory with a patched
``config.json`` where ``rope_theta`` (and ``rope_local_base_freq``) are
moved *into* each nested ``rope_parameters`` entry, and removed from the
``text_config`` top level.  All other model files are symlinked from the
HuggingFace cache to avoid duplication.

Usage::

    python scripts/prepare_translate_gemma.py
    python scripts/prepare_translate_gemma.py --model google/translategemma-4b-it
    python scripts/prepare_translate_gemma.py --output models/translategemma-patched
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def prepare_patched_model(
    model_id: str,
    output_dir: Path,
) -> Path:
    """Create a local patched model directory.

    Args:
        model_id: HuggingFace model ID (e.g. ``google/translategemma-4b-it``).
        output_dir: Where to create the patched model directory.

    Returns:
        The path to the patched model directory.
    """
    # Ensure the model is downloaded and get its cache path
    cache_dir = Path(snapshot_download(model_id))
    print(f"Cache directory: {cache_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Symlink all files except config.json
    for src_file in cache_dir.iterdir():
        dst_file = output_dir / src_file.name
        if src_file.name == "config.json":
            continue
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        dst_file.symlink_to(src_file.resolve())

    # Patch config.json
    config_src = cache_dir / "config.json"
    with open(config_src, "r", encoding="utf-8") as f:
        config = json.load(f)

    text_config = config.get("text_config", {})
    rope_params = text_config.get("rope_parameters")

    if rope_params and isinstance(rope_params, dict):
        # Collect scalar RoPE fields that vLLM would inject
        rope_theta = text_config.get("rope_theta")
        rope_local_base_freq = text_config.get("rope_local_base_freq")

        # Check if rope_parameters is nested (keyed by attention type)
        attention_types = {
            "full_attention", "sliding_attention",
            "chunked_attention", "linear_attention",
        }
        nested_keys = set(rope_params.keys())
        is_nested = bool(nested_keys) and nested_keys.issubset(attention_types)

        if is_nested:
            print("Detected nested rope_parameters — patching …")

            for layer_type, entry in rope_params.items():
                if not isinstance(entry, dict):
                    continue

                # Assign the appropriate rope_theta per layer type:
                # - sliding_attention → rope_local_base_freq (short-range)
                # - full_attention    → rope_theta (long-range)
                if layer_type == "sliding_attention" and rope_local_base_freq is not None:
                    entry.setdefault("rope_theta", rope_local_base_freq)
                elif rope_theta is not None:
                    entry.setdefault("rope_theta", rope_theta)

            # Remove these from text_config top level so vLLM doesn't
            # re-inject them and break the nested detection
            text_config.pop("rope_theta", None)
            text_config.pop("rope_local_base_freq", None)

            print(f"  Patched rope_parameters: {json.dumps(rope_params, indent=2)}")
            print(f"  Removed text_config keys: rope_theta, rope_local_base_freq")
        else:
            print("rope_parameters is flat — no patching needed.")
    else:
        print("No rope_parameters found — no patching needed.")

    # Write patched config
    config_dst = output_dir / "config.json"
    with open(config_dst, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Patched config written to {config_dst}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Patch TranslateGemma config.json for vLLM 0.16.0 compatibility."
    )
    parser.add_argument(
        "--model", type=str, default="google/translategemma-4b-it",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "models" / "translategemma-patched"),
        help="Output path for the patched model directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    patched_path = prepare_patched_model(args.model, output_dir)
    print(f"\nReady. Use this model path with vLLM:\n  {patched_path}")


if __name__ == "__main__":
    main()
