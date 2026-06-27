#!/bin/bash
# Batch convert + evaluate multiple intermediate ETD-k checkpoints.
#
# For each step in STEPS, this runs:
#   bash scripts/convert_etd_checkpoint.sh <step> <K> <RUN_DIR>
#   python scripts/eval_etd_checkpoint_multi.py <step> <K> <RUN_DIR> --gpu <GPU>
#
# Edit the variables below, then run:
#   bash scripts/inference_etd_checkpoints.sh

# ── Configure here ──────────────────────────────────────────────────────────
# Steps to inference (one per line or space-separated)
STEPS=(
    10000
    12250
    12500
    12750
    13000
)

K=2                                  # number of ETD thinking iterations
RUN_DIR=running/replication/ETD_k2   # training run directory
GPU=4                                # GPU index to use
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP ${step}: evaluating on GPU ${GPU}..."
    echo "============================================================"
    if ! python scripts/eval_etd_checkpoint_multi.py "$step" "$K" "$RUN_DIR" --gpu "$GPU"; then
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
