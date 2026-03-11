#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET=${1:-"mgsm"} # default to mgsm, but accept an argument (mgsm or mmmlu)

source /home/alfred/miniconda3/etc/profile.d/conda.sh
conda activate ml-delta

echo "Splitting dataset: $DATASET"
python3 "$PROJECT_ROOT/data/split_datasets-exp_v2.py" --dataset "$DATASET"
echo "Dataset splitting complete"
