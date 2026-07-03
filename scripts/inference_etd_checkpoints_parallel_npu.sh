#!/bin/bash
# Parallel NPU counterpart of inference_etd_checkpoints.sh.
#
# H100's inference scripts assume a single chip doing one checkpoint at a
# time, since that matches how GPUs are shared there. The NPU server has
# 8 dedicated chips, so this script instead evaluates up to 8 checkpoints
# *simultaneously*, one per NPU device.
#
# Given N steps in STEPS (N <= 8), checkpoint STEPS[i] is evaluated on NPU
# device i. Conversion (CPU/disk-bound) still happens sequentially first,
# to avoid N-way simultaneous large temp-copy I/O; only the actual NPU
# inference step runs in parallel.
#
# Edit the variables below, then run:
#   bash scripts/inference_etd_checkpoints_parallel_npu.sh

# ── Configure here ──────────────────────────────────────────────────────────
# Steps to inference -- one entry per NPU device (index 0, 1, 2, ...).
# Must have at most 8 entries (one per available NPU chip).
STEPS=(
    10000
    12250
    12500
    12750
    13000
)

K=2                                       # number of ETD thinking iterations
RUN_DIR=running/replication/ETD_k2_npu    # training run directory
# ────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.." || exit 1    # run from repo root regardless of cwd

N=${#STEPS[@]}
if [ "$N" -gt 8 ]; then
    echo "ERROR: ${N} steps requested, but only 8 NPU devices are available."
    echo "  Split STEPS into batches of at most 8 and run this script once per batch."
    exit 1
fi
if [ "$N" -eq 0 ]; then
    echo "ERROR: STEPS is empty."
    exit 1
fi

echo "============================================================"
echo "Phase 1/2: converting ${N} checkpoint(s) sequentially (CPU/disk-bound)"
echo "============================================================"
convert_failed_steps=()
for step in "${STEPS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Converting step ${step}..."
    if ! bash scripts/convert_etd_checkpoint.sh "$step" "$K" "$RUN_DIR"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] convert FAILED for step ${step}."
        convert_failed_steps+=("$step")
    fi
done

if [ ${#convert_failed_steps[@]} -gt 0 ]; then
    echo "Aborting: conversion failed for step(s): ${convert_failed_steps[*]}"
    exit 1
fi

echo "============================================================"
echo "Phase 2/2: evaluating ${N} checkpoint(s) in parallel across NPU devices 0-$((N - 1))"
echo "============================================================"

pids=()
for i in "${!STEPS[@]}"; do
    step="${STEPS[$i]}"
    log_file="/tmp/eval_npu${i}_step${step}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching step ${step} on NPU device ${i} (log: ${log_file})"
    (
        export HF_MODULES_CACHE="/tmp/hf_modules_etd_eval_npu${i}"
        python scripts/eval_etd_checkpoint_multi_npu.py "$step" "$K" "$RUN_DIR" --npu-device "$i"
    ) > "$log_file" 2>&1 &
    pids+=($!)
done

echo "All ${N} job(s) launched. Waiting for completion..."
echo "(tail -f any of the /tmp/eval_npu*.log files to watch progress live)"

failed_steps=()
for i in "${!STEPS[@]}"; do
    step="${STEPS[$i]}"
    if ! wait "${pids[$i]}"; then
        failed_steps+=("$step (NPU device ${i}, see /tmp/eval_npu${i}_step${step}.log)")
    fi
done

echo "============================================================"
if [ ${#failed_steps[@]} -eq 0 ]; then
    echo "All ${N} step(s) completed successfully."
else
    echo "Completed with ${#failed_steps[@]} failure(s):"
    for f in "${failed_steps[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
