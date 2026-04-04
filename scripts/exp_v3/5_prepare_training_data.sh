#!/bin/bash
set -e

# Prepare SFT training data for qwen3-8b / mmmlu from translated_gemma-27b responses.
#
# Unlike exp_v2 (which had both translated/ and native/ response files),
# qwen3-8b only has translated responses.  The native-language questions are
# sourced directly from the mmmlu train split (matched by question_index).
# This means we produce SFT data only (no DPO — there are no native responses
# to use as rejected examples).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_SETUP="${CONDA_SETUP:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SETUP"
conda activate "${CONDA_ENV:-ml-delta}"

LANGS=("ja" "bn" "sw")

echo "Preparing SFT training data for qwen3-8b / mmmlu (translated_gemma-27b)"
for lang in "${LANGS[@]}"; do
    python3 "${PROJECT_ROOT}/data/prepare_training_data-exp_v3.py" \
        --language "$lang"
done

echo "Data preparation complete."
