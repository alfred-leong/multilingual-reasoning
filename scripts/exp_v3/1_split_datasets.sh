#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET=${1:-"mmmlu"} # default to mmmlu

CONDA_SETUP="${CONDA_SETUP:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SETUP"
conda activate "${CONDA_ENV:-ml-delta}"

echo "Splitting dataset: $DATASET"
python3 "$PROJECT_ROOT/data/split_datasets-exp_v2.py" --dataset "$DATASET"
echo "Dataset splitting complete"
