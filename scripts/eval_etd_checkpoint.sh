#!/bin/bash
# Usage: bash scripts/eval_etd_checkpoint.sh <step> <k> <run_dir> [task]
# Example: bash scripts/eval_etd_checkpoint.sh 5 2 running/sanity/ETD_k2_per_step
#          bash scripts/eval_etd_checkpoint.sh 5 2 running/train/ETD_k2 arc_challenge::olmes

set -e

STEP=$1
K=$2
RUN_DIR=$3
TASK=${4:-arc_challenge::olmes}
GPU=${CUDA_VISIBLE_DEVICES:-6}

OLMO_ROOT=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo
OLMES_ROOT=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/inference_inter_olmo2_1B_midtrain/olmes
OLMES_BIN=${OLMES_ROOT}/.venv/bin/olmes

# Ensure the venv's python is used for subprocesses launched by olmes
export PATH=${OLMES_ROOT}/.venv/bin:$PATH

# Force HuggingFace to always re-read custom model code instead of using stale cache
export HF_MODULES_CACHE=/tmp/hf_modules_etd_eval

if [ -z "$STEP" ] || [ -z "$K" ] || [ -z "$RUN_DIR" ]; then
    echo "Usage: bash scripts/eval_etd_checkpoint.sh <step> <k> <run_dir> [task]"
    exit 1
fi

MODEL_PATH=${OLMO_ROOT}/${RUN_DIR}/step${STEP}-hf
TASK_NAME="${TASK%%::*}"
OUTPUT_DIR=${OLMO_ROOT}/eval_results/${RUN_DIR}/step${STEP}/${TASK_NAME}

# Sanity checks
if [ ! -f "${OLMES_BIN}" ]; then
    echo "ERROR: olmes binary not found: ${OLMES_BIN}"
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: HF checkpoint not found: ${MODEL_PATH}"
    echo "  Run convert first: bash scripts/convert_etd_checkpoint.sh ${STEP} ${K} ${RUN_DIR}"
    exit 1
fi
if [ ! -f "${MODEL_PATH}/modeling_olmo.py" ]; then
    echo "ERROR: ${MODEL_PATH}/modeling_olmo.py missing — conversion was incomplete."
    echo "  Delete ${MODEL_PATH} and re-run: bash scripts/convert_etd_checkpoint.sh ${STEP} ${K} ${RUN_DIR}"
    exit 1
fi

echo "Evaluating ETD-k${K} step ${STEP} on task '${TASK}'"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU} "${OLMES_BIN}" \
    --model etd-k${K}-step${STEP} \
    --model-type hf \
    --model-args "model_path=${MODEL_PATH},trust_remote_code=True" \
    --task "${TASK}" \
    --output-dir "${OUTPUT_DIR}"

echo "Done: results saved to ${OUTPUT_DIR}"
