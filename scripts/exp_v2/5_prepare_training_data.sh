#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET=${1:-"mgsm"} # default to mmmlu, but accept an argument

source /tier1/home/lweilun/miniconda3/etc/profile.d/conda.sh
conda activate ml-delta

LANGS=("ja" "bn" "sw")

echo "Preparing training data (SFT and DPO splits) for dataset: $DATASET"
for lang in "${LANGS[@]}"; do
    python3 "$PROJECT_ROOT/data/prepare_training_data-exp_v2.py" --dataset "$DATASET" --language "$lang"
done

echo "Data preparation complete."
