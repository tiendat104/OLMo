#!/bin/bash
# Usage: bash scripts/eval_etd_checkpoint.sh <step> <k> [run_dir] [task]
# Example: bash scripts/eval_etd_checkpoint.sh 5 2
#          bash scripts/eval_etd_checkpoint.sh 10 1 running/sanity/ETD_k1_per_step arc_challenge::olmes

set -e

STEP=$1
K=$2
RUN_DIR=${3:-running/sanity/ETD_k${K}_per_step}
TASK=${4:-arc_challenge::olmes}
GPU=${CUDA_VISIBLE_DEVICES:-6}

OLMO_ROOT=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo
OLMES_ROOT=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/inference_inter_olmo2_1B_midtrain/olmes
OLMES_BIN=${OLMES_ROOT}/.venv/bin/olmes

# Ensure the venv's python is used for subprocesses launched by olmes
export PATH=${OLMES_ROOT}/.venv/bin:$PATH

if [ -z "$STEP" ] || [ -z "$K" ]; then
    echo "Usage: bash scripts/eval_etd_checkpoint.sh <step> <k> [run_dir] [task]"
    exit 1
fi

MODEL_PATH=${OLMO_ROOT}/${RUN_DIR}/step${STEP}-hf
OUTPUT_DIR=${OLMO_ROOT}/eval_results/${RUN_DIR}/step${STEP}

# Sanity checks
if [ ! -f "${OLMES_BIN}" ]; then
    echo "ERROR: olmes binary not found: ${OLMES_BIN}"
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: HF checkpoint not found: ${MODEL_PATH}"
    echo "  Run convert first: bash scripts/convert_etd_checkpoint.sh ${STEP} ${K}"
    exit 1
fi
if [ ! -f "${MODEL_PATH}/modeling_olmo.py" ]; then
    echo "ERROR: ${MODEL_PATH}/modeling_olmo.py missing — conversion was incomplete."
    echo "  Delete ${MODEL_PATH} and re-run: bash scripts/convert_etd_checkpoint.sh ${STEP} ${K}"
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
