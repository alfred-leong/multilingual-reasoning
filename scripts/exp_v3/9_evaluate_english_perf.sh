#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_SETUP="${CONDA_SETUP:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SETUP"
conda activate "${CONDA_ENV:-ml-delta}"

MODEL_STORE="${MODEL_STORE:-/external1/alfred/models/ml-reasoning-exp_v3}"
OUTPUT_DIR="outputs-exp_v3"
LOG_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}/logs"
GPUS=(5 6 7)

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Collect all models
# ---------------------------------------------------------------------------
mapfile -t ALL_MODELS < <(find "$MODEL_STORE" -maxdepth 1 -mindepth 1 -type d -name "*mmmlu*" | sort)
echo "Found ${#ALL_MODELS[@]} MMMLU models to evaluate on English."

# ---------------------------------------------------------------------------
# Phase 1: Base model evaluation (Qwen3-1.7B and Qwen3-8B on English)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 1: Base model English evaluation ==="

BASE_PIDS=()
base_idx=0
for base_model in "Qwen/Qwen3-1.7B" "Qwen/Qwen3-8B"; do
    base_label="${base_model##*/}"
    gpu="${GPUS[$((base_idx % ${#GPUS[@]}))]}"

    for dataset in mgsm mmmlu; do
        if [ "$dataset" = "mmmlu" ]; then
            SUBSET_ARGS="--subset 100"
            SUBSET_TAG="_100"
        else
            SUBSET_ARGS=""
            SUBSET_TAG=""
        fi

        out_file="${PROJECT_ROOT}/${OUTPUT_DIR}/eval_${dataset}${SUBSET_TAG}_en_${base_label}.json"
        if [ -f "$out_file" ]; then
            echo "  [base] ${base_label} ${dataset} already evaluated. Skipping."
            continue
        fi

        log_file="${LOG_DIR}/eval_en_${dataset}${SUBSET_TAG}_${base_label}.log"
        echo "  [base] Evaluating ${base_label} on ${dataset} (English) GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="$gpu" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
            --dataset "$dataset" --language en --model_dir "$base_model" --full_finetune \
            $SUBSET_ARGS --output_dir "$OUTPUT_DIR" \
            > "$log_file" 2>&1 &
        BASE_PIDS+=($!)

        # Use next GPU for the next job
        base_idx=$((base_idx + 1))
    done
done

for pid in "${BASE_PIDS[@]}"; do
    wait "$pid" || echo "WARNING: base eval PID $pid failed"
done
echo "Base model English evaluation complete."

# ---------------------------------------------------------------------------
# Phase 2: Trained model evaluation on English (MGSM + MMMLU-100)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: Trained model English evaluation ==="

# Build a queue of (model_dir, dataset) pairs
declare -a EVAL_QUEUE=()
for model_dir in "${ALL_MODELS[@]}"; do
    model_name="$(basename "$model_dir")"
    for dataset in mgsm mmmlu; do
        if [ "$dataset" = "mmmlu" ]; then
            SUBSET_TAG="_100"
        else
            SUBSET_TAG=""
        fi
        out_file="${PROJECT_ROOT}/${OUTPUT_DIR}/eval_${dataset}${SUBSET_TAG}_en_${model_name}.json"
        if [ -f "$out_file" ]; then
            echo "  [skip] ${model_name} ${dataset} already evaluated."
            continue
        fi
        EVAL_QUEUE+=("${model_dir}|${dataset}")
    done
done

echo "  ${#EVAL_QUEUE[@]} evaluations to run."

# Run evaluations, 3 at a time across GPUs
idx=0
while [ $idx -lt ${#EVAL_QUEUE[@]} ]; do
    PIDS=()
    for gpu_slot in 0 1 2; do
        job_idx=$((idx + gpu_slot))
        if [ $job_idx -ge ${#EVAL_QUEUE[@]} ]; then
            break
        fi

        IFS='|' read -r model_dir dataset <<< "${EVAL_QUEUE[$job_idx]}"
        model_name="$(basename "$model_dir")"
        gpu="${GPUS[$gpu_slot]}"

        if [ "$dataset" = "mmmlu" ]; then
            SUBSET_ARGS="--subset 100"
            SUBSET_TAG="_100"
        else
            SUBSET_ARGS=""
            SUBSET_TAG=""
        fi

        log_file="${LOG_DIR}/eval_en_${dataset}${SUBSET_TAG}_${model_name}.log"
        echo "  [GPU ${gpu}] ${model_name} -> ${dataset} (English)"
        CUDA_VISIBLE_DEVICES="$gpu" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
            --dataset "$dataset" --language en --model_dir "$model_dir" \
            --full_finetune $SUBSET_ARGS --output_dir "$OUTPUT_DIR" \
            > "$log_file" 2>&1 &
        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "WARNING: eval PID $pid failed"
    done

    idx=$((idx + 3))
done

echo "All trained model English evaluations complete."

# ---------------------------------------------------------------------------
# Phase 3: Report results
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 3: Results Summary ==="

python3 - <<'PYEOF'
import json
import os
from collections import defaultdict
from pathlib import Path

output_dir = Path(os.environ.get("PROJECT_ROOT", ".")) / "outputs-exp_v3"

# Collect all English eval results
results = []
for f in sorted(output_dir.glob("eval_*_en_*.json")):
    with open(f) as fh:
        data = json.load(fh)
    results.append({"file": f.name, **data})

if not results:
    print("No English evaluation results found.")
    exit()

# -------------------------------------------------------------------------
# Group by (dataset, model_group) where model_group strips seed
# -------------------------------------------------------------------------
def get_group_key(model_name):
    """Remove seed suffix to group models."""
    import re
    return re.sub(r"_seed\d+$", "", model_name)

def get_seed(model_name):
    import re
    m = re.search(r"_seed(\d+)$", model_name)
    return m.group(1) if m else "N/A"

# Organize: {dataset: {group: [(seed, accuracy)]}}
grouped = defaultdict(lambda: defaultdict(list))
for r in results:
    dataset = r["dataset"]
    model_name = Path(r.get("model_dir", "")).name
    if not model_name:
        model_name = r["file"].split("_en_")[-1].replace(".json", "")

    subset = ""
    if "total" in r and r["total"] == 100:
        subset = " (100)"

    group = get_group_key(model_name)
    seed = get_seed(model_name)
    grouped[dataset][group].append((seed, r["accuracy"], r["correct"], r["total"]))

# Print results
for dataset in ["mgsm", "mmmlu"]:
    if dataset not in grouped:
        continue
    print(f"\n{'='*80}")
    print(f"  DATASET: {dataset.upper()} — English Performance")
    print(f"{'='*80}")
    print(f"{'Model Group':<65} {'Seeds':>5} {'Avg Acc':>8} {'Per-Seed Accuracies'}")
    print(f"{'-'*65} {'-'*5} {'-'*8} {'-'*30}")

    for group in sorted(grouped[dataset].keys()):
        entries = sorted(grouped[dataset][group], key=lambda x: x[0])
        accs = [e[1] for e in entries]
        avg_acc = sum(accs) / len(accs)
        seed_strs = [f"{e[0]}:{e[1]:.2%}" for e in entries]
        print(f"{group:<65} {len(entries):>5} {avg_acc:>8.2%} {', '.join(seed_strs)}")

    print()
PYEOF

echo ""
echo "Done. All results saved in ${OUTPUT_DIR}/."
