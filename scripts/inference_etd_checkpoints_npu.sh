#!/bin/bash
# NPU counterpart of inference_etd_checkpoints.sh.
# Batch convert + evaluate multiple intermediate ETD-k checkpoints on an
# Ascend NPU device.
#
# For each step in STEPS, this runs:
#   bash scripts/convert_etd_checkpoint.sh <step> <K> <RUN_DIR>
#   python scripts/eval_etd_checkpoint_multi_npu.py <step> <K> <RUN_DIR> --npu-device <NPU_DEVICE>
#
# Edit the variables below, then run:
#   bash scripts/inference_etd_checkpoints_npu.sh

# ── Configure here ──────────────────────────────────────────────────────────
# Steps to inference (one per line or space-separated)
STEPS=(
    10000
    12250
    12500
    12750
    13000
)

K=2                                       # number of ETD thinking iterations
RUN_DIR=running/replication/ETD_k2_npu    # training run directory
NPU_DEVICE=0                               # NPU device index to use
# ────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.." || exit 1    # run from repo root regardless of cwd

failed_steps=()

for step in "${STEPS[@]}"; do
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP ${step}: converting..."
    echo "============================================================"
    if ! bash scripts/convert_etd_checkpoint.sh "$step" "$K" "$RUN_DIR"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] convert FAILED for step ${step}, skipping eval."
        failed_steps+=("$step (convert)")
        continue
    fi

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP ${step}: evaluating on NPU device ${NPU_DEVICE}..."
    echo "============================================================"
    if ! python scripts/eval_etd_checkpoint_multi_npu.py "$step" "$K" "$RUN_DIR" --npu-device "$NPU_DEVICE"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval FAILED for step ${step}."
        failed_steps+=("$step (eval)")
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP ${step}: done."
done

echo "============================================================"
if [ ${#failed_steps[@]} -eq 0 ]; then
    echo "All ${#STEPS[@]} step(s) completed successfully."
else
    echo "Completed with ${#failed_steps[@]} failure(s):"
    for f in "${failed_steps[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
