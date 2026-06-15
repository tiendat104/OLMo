#!/bin/bash
# Usage: bash scripts/eval_etd_checkpoint.sh <step> [run_dir] [task]
# Example: bash scripts/eval_etd_checkpoint.sh 5
#          bash scripts/eval_etd_checkpoint.sh 10 running/sanity/ETD_k2_per_step arc_challenge::olmes

set -e

STEP=$1
RUN_DIR=${2:-running/sanity/ETD_k2_per_step}
TASK=${3:-arc_challenge::olmes}
GPU=${CUDA_VISIBLE_DEVICES:-6}

OLMO_ROOT=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo
OLMES_BIN=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/inference_inter_olmo2_1B_midtrain/olmes/.venv/bin/olmes

if [ -z "$STEP" ]; then
    echo "Usage: bash scripts/eval_etd_checkpoint.sh <step> [run_dir] [task]"
    exit 1
fi

MODEL_PATH=${OLMO_ROOT}/${RUN_DIR}/step${STEP}-hf
OUTPUT_DIR=${OLMO_ROOT}/eval_results/${RUN_DIR}/step${STEP}

echo "Evaluating step ${STEP} on task '${TASK}'"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU} "${OLMES_BIN}" \
    --model etd-k2-step${STEP} \
    --model-type hf \
    --model-args "model_path=${MODEL_PATH},trust_remote_code=True" \
    --task "${TASK}" \
    --output-dir "${OUTPUT_DIR}"

echo "Done: results saved to ${OUTPUT_DIR}"
