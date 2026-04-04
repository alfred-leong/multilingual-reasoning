#!/bin/bash
set -e

# SFT training + evaluation for Qwen3-1.7B on mmmlu, all languages (ja, bn, sw).
# For each language: training (3 seeds x 2 filter modes) → base eval → trained-model eval.
# Languages are processed sequentially; seeds are evaluated in parallel.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_SETUP="${CONDA_SETUP:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SETUP" 2>/dev/null
conda activate "${CONDA_ENV:-ml-delta}"

# ── Configuration ────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-1.7B"
MODEL_KEY="qwen3-1_7b"
DATASET="mmmlu"
SEEDS=(42 123 456)
FILTERS=("no-filter" "filter")
LANGS=("ja" "bn" "sw")
GPU_IDS="5,6,7"
EVAL_GPUS=(5 6 7)
MODEL_STORE="${MODEL_STORE:-/external1/alfred/models/ml-reasoning-exp_v3}"
OUTPUT_DIR="outputs-exp_v3"
LR="2e-7"
LOG_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ── Helper: evaluate one trained model on mgsm + mmmlu ──────────────────────
run_model_evals() {
    local gpu_id=$1
    local run_name=$2
    local model_dir=$3
    local lang=$4

    for eval_dataset in mgsm mmmlu; do
        SUBSET_ARGS=""
        SUBSET_TAG=""
        if [ "$eval_dataset" == "mmmlu" ]; then
            SUBSET_ARGS="--subset 100"
            SUBSET_TAG="_100"
        fi

        out_file="${PROJECT_ROOT}/${OUTPUT_DIR}/eval_${eval_dataset}${SUBSET_TAG}_${lang}_${run_name}.json"
        if [ -f "$out_file" ]; then
            echo "  [${run_name}] ${eval_dataset} already evaluated. Skipping."
            continue
        fi

        log_file="${LOG_DIR}/eval_${run_name}_${eval_dataset}.log"
        echo "  [${run_name}] Evaluating on ${eval_dataset} (GPU ${gpu_id})"

        CUDA_VISIBLE_DEVICES="$gpu_id" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
            --dataset "$eval_dataset" --language "$lang" \
            --model_dir "$model_dir" --full_finetune \
            --run_name "$run_name" \
            $SUBSET_ARGS --output_dir "$OUTPUT_DIR" \
            > "$log_file" 2>&1
    done
}

# ── Helper: evaluate trained models for a language (3 seeds in parallel) ─────
eval_trained_models() {
    local lang=$1

    for seed_idx in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$seed_idx]}
        GPU=${EVAL_GPUS[$seed_idx]}

        (
            for FILTER in "${FILTERS[@]}"; do
                model_dir_name="sft_${FILTER}_full_${DATASET}_${MODEL_KEY}_${lang}_seed${SEED}"
                run_name="sft_${FILTER}_full_${DATASET}_${MODEL_KEY}_${lang}_lr${LR}_seed${SEED}"
                model_dir="${MODEL_STORE}/${model_dir_name}"
                run_model_evals "$GPU" "$run_name" "$model_dir" "$lang"
            done
        ) &
    done
    wait
}

# ── Helper: verify all eval outputs exist for a language ─────────────────────
verify_outputs() {
    local lang=$1
    for FILTER in "${FILTERS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_name="sft_${FILTER}_full_${DATASET}_${MODEL_KEY}_${lang}_lr${LR}_seed${SEED}"
            for eval_dataset in mgsm mmmlu; do
                SUBSET_TAG=""
                if [ "$eval_dataset" == "mmmlu" ]; then
                    SUBSET_TAG="_100"
                fi
                out_file="${PROJECT_ROOT}/${OUTPUT_DIR}/eval_${eval_dataset}${SUBSET_TAG}_${lang}_${run_name}.json"
                if [ ! -f "$out_file" ]; then
                    echo "  ERROR: missing ${out_file}"
                    return 1
                fi
            done
        done
    done
}

# ═════��═══════════════════════════════════���════════════════════════════════════
# Run full pipeline for each language sequentially
# ════════════════════��═══════════════════════════════��═════════════════════════
for LANG in "${LANGS[@]}"; do
    LANG_UPPER=$(echo "$LANG" | tr '[:lower:]' '[:upper:]')
    echo "========================================================================"
    echo "[${LANG_UPPER}] Full pipeline (GPUs ${GPU_IDS})"
    echo "========================================================================"

    # ── Phase 1: Training ────────────────────────────────────────────��───────
    echo "  Phase 1: SFT Training (${LANG})"
    echo "  --------------------------------------------------------------------"

    for FILTER in "${FILTERS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            model_dir_name="sft_${FILTER}_full_${DATASET}_${MODEL_KEY}_${LANG}_seed${SEED}"
            run_name="sft_${FILTER}_full_${DATASET}_${MODEL_KEY}_${LANG}_lr${LR}_seed${SEED}"
            model_dir="${MODEL_STORE}/${model_dir_name}"

            if [ -f "${model_dir}/config.json" ]; then
                echo "  [${FILTER} seed${SEED}] Already exists: ${model_dir}. Skipping."
                continue
            fi

            log_file="${LOG_DIR}/${run_name}.log"
            echo "  [${FILTER} seed${SEED}] Training on GPUs ${GPU_IDS} -> ${log_file}"

            CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch --num_processes 3 \
                --use_deepspeed --deepspeed_config_file "${PROJECT_ROOT}/configs/deepspeed_zero3.json" \
                "${PROJECT_ROOT}/training/run_sft-exp_v2.py" \
                --dataset "$DATASET" --language "$LANG" --filter "$FILTER" \
                --model "$MODEL" --full_finetune --seed "$SEED" \
                --model_store "$MODEL_STORE" \
                > "$log_file" 2>&1

            if [ $? -ne 0 ]; then
                echo "  Training failed: ${run_name}. Check ${log_file}"
                exit 1
            fi

            echo "  [${FILTER} seed${SEED}] Done."
        done
    done

    echo "  All training runs complete for ${LANG}."

    # ── Phase 2: Base model evaluation ───────────────────────────────────────
    echo "  Phase 2: Base model evaluation (${LANG})"
    echo "  --------------------------------------------------------------------"

    base_mgsm_out="${PROJECT_ROOT}/${OUTPUT_DIR}/eval_mgsm_${LANG}_Qwen3-1.7B.json"
    if [ -f "$base_mgsm_out" ]; then
        echo "  [base] MGSM already evaluated. Skipping."
    else
        echo "  [base] Evaluating on MGSM (GPU ${EVAL_GPUS[0]})"
        CUDA_VISIBLE_DEVICES="${EVAL_GPUS[0]}" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
            --dataset mgsm --language "$LANG" --model_dir "$MODEL" --full_finetune \
            --output_dir "$OUTPUT_DIR" \
            > "${LOG_DIR}/eval_base_mgsm_${LANG}.log" 2>&1
    fi

    base_mmmlu_out="${PROJECT_ROOT}/${OUTPUT_DIR}/eval_mmmlu_100_${LANG}_Qwen3-1.7B.json"
    if [ -f "$base_mmmlu_out" ]; then
        echo "  [base] MMMLU already evaluated. Skipping."
    else
        echo "  [base] Evaluating on MMMLU-100 (GPU ${EVAL_GPUS[0]})"
        CUDA_VISIBLE_DEVICES="${EVAL_GPUS[0]}" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
            --dataset mmmlu --language "$LANG" --model_dir "$MODEL" --full_finetune \
            --subset 100 --output_dir "$OUTPUT_DIR" \
            > "${LOG_DIR}/eval_base_mmmlu_${LANG}.log" 2>&1
    fi

    # ── Phase 3: Trained model evaluation ──────────────────────────────���─────
    echo "  Phase 3: Trained model evaluation (${LANG})"
    echo "  --------------------------------------------------------------------"

    eval_trained_models "$LANG"
    verify_outputs "$LANG"

    echo "[${LANG_UPPER}] All phases complete."
done

# ── Aggregate Results ────────────────────────────────────���───────────────────
echo "========================================================================"
echo "Results Summary"
echo "========================================================================"

python3 - "$PROJECT_ROOT" "$OUTPUT_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
output_dir = project_root / sys.argv[2]

seeds = [42, 123, 456]
filters = ["no-filter", "filter"]
model_key = "qwen3-1_7b"
langs = ["ja", "bn", "sw"]

def load_acc(filepath):
    return json.loads(filepath.read_text())["accuracy"]

for lang in langs:
    print()
    print(f"=== {lang.upper()} ===")
    print(f"{'Setting':<40} {'MGSM (50)':>12} {'MMMLU (100)':>12}")
    print("-" * 66)

    base_mgsm = load_acc(output_dir / f"eval_mgsm_{lang}_Qwen3-1.7B.json")
    base_mmmlu = load_acc(output_dir / f"eval_mmmlu_100_{lang}_Qwen3-1.7B.json")
    print(f"{'Base Qwen3-1.7B':<40} {base_mgsm:>11.2%} {base_mmmlu:>12.2%}")

    for filt in filters:
        mgsm_accs = []
        mmmlu_accs = []
        for seed in seeds:
            run_name = f"sft_{filt}_full_mmmlu_{model_key}_{lang}_lr2e-7_seed{seed}"
            mgsm_accs.append(load_acc(output_dir / f"eval_mgsm_{lang}_{run_name}.json"))
            mmmlu_accs.append(load_acc(output_dir / f"eval_mmmlu_100_{lang}_{run_name}.json"))

        avg_mgsm = sum(mgsm_accs) / len(mgsm_accs)
        avg_mmmlu = sum(mmmlu_accs) / len(mmmlu_accs)
        label = f"SFT full, {filt} (avg 3 seeds)"
        print(f"{label:<40} {avg_mgsm:>11.2%} {avg_mmmlu:>12.2%}")

        for i, seed in enumerate(seeds):
            sublabel = f"  seed {seed}"
            print(f"{sublabel:<40} {mgsm_accs[i]:>11.2%} {mmmlu_accs[i]:>12.2%}")

    print("-" * 66)

print()
PYEOF
