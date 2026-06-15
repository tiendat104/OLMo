#!/bin/bash
# Evaluate an intermediate mid-training checkpoint of OLMo 2 1B baseline via olmes.
# Usage: bash scripts/eval_olmo2_baseline.sh <step> <revision> [task]
# Example: bash scripts/eval_olmo2_baseline.sh 1000 stage2-ingredient3-step1000-tokens3B
#          bash scripts/eval_olmo2_baseline.sh 5000 stage2-ingredient3-step5000-tokens11B arc_challenge::olmes

set -e

STEP=$1
REVISION=$2
TASK=${3:-gsm8k::olmes}
GPU=${CUDA_VISIBLE_DEVICES:-7}

OLMO_ROOT=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo
OLMES_ROOT=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/inference_inter_olmo2_1B_midtrain/olmes
OLMES_BIN=${OLMES_ROOT}/.venv/bin/olmes

# Ensure the venv's python is used for subprocesses launched by olmes
export PATH=${OLMES_ROOT}/.venv/bin:$PATH

if [ -z "$STEP" ] || [ -z "$REVISION" ]; then
    echo "Usage: bash scripts/eval_olmo2_baseline.sh <step> <revision> [task]"
    echo "  revision: HuggingFace revision string, e.g. stage2-ingredient3-step1000-tokens3B"
    exit 1
fi

OUTPUT_DIR=${OLMO_ROOT}/eval_results/baseline_olmo2_1B/step${STEP}

# Sanity checks
if [ ! -f "${OLMES_BIN}" ]; then
    echo "ERROR: olmes binary not found: ${OLMES_BIN}"
    exit 1
fi

echo "Evaluating OLMo 2 1B baseline"
echo "  Step:     ${STEP}"
echo "  Revision: ${REVISION}"
echo "  Task:     ${TASK}"
echo "  Output:   ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU} "${OLMES_BIN}" \
    --model allenai/OLMo-2-0425-1B \
    --model-args "{\"revision\": \"${REVISION}\", \"trust_remote_code\": true, \"gpu_memory_utilization\": 0.5}" \
    --model-type vllm \
    --task "${TASK}" \
    --output-dir "${OUTPUT_DIR}"

echo "Done: results saved to ${OUTPUT_DIR}"
