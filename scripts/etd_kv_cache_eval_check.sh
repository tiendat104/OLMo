#!/bin/bash
# End-to-end check of the ETD KV-cache fix against a REAL trained checkpoint.
#
# The unit suite (scripts/etd_kv_cache_test.py) proves the cache is indexed correctly
# using random weights. This script closes the loop differently: it runs a real
# downstream benchmark on the actual replication checkpoint and compares the score
# against the number that checkpoint already produced.
#
# Why a generative task. Most of the 17 benchmarks are multiple-choice, scored by
# log-likelihood over a fixed sequence -- a single forward pass, no incremental
# decoding, so the KV cache is barely exercised. Only generative tasks (gsm8k,
# minerva_math, drop, bbh) decode token by token and actually test it. gsm8k is also
# the headline ETD metric.
#
# Isolation. Nothing is written into the replication checkout. The checkpoint is read
# from there; a fresh HF export, the HF module cache, and all results are written under
# this working copy.
#
# Why a fresh HF export is mandatory. convert_etd_checkpoint.sh copies modeling_olmo.py
# and configuration_olmo.py *into* the exported folder and registers them via auto_map.
# With trust_remote_code=True, olmes loads the model code from that folder rather than
# from the repo -- so the existing step<N>-hf folder still contains the pre-fix code and
# would silently test nothing.
#
# DO NOT use a multiple-choice task here. BoolQ, OpenBookQA, CommonsenseQA, SocialIQA,
# ARC, HellaSwag, MMLU and friends are scored by the log-likelihood of each option over
# a fixed sequence: one forward pass per option, no incremental decoding, so the KV
# cache is never touched. They would pass identically whether this fix is correct or
# catastrophically broken. Their speed is exactly because they do not generate.
#
# LIMIT bounds the number of examples. The decisive comparison here is cache-off versus
# cache-on under identical conditions, not against a published score, so a subset is
# fully valid as long as both runs see the same one (they do -- olmes takes the first N).
# Set LIMIT=0 for the full set, which additionally allows comparison against the stored
# replication result.
#
# Usage:
#   bash scripts/etd_kv_cache_eval_check.sh off          # baseline
#   bash scripts/etd_kv_cache_eval_check.sh on           # the actual test
#   LIMIT=0 TASK=triviaqa::olmes bash scripts/etd_kv_cache_eval_check.sh off
#
# Then diff the generated text per example, which is far sharper than comparing scores:
#   python scripts/etd_kv_cache_eval_diff.py
#
# Run 'off' first. It isolates the cache as the only variable in the 'on' run.

set -e

MODE=${1:?Usage: bash scripts/etd_kv_cache_eval_check.sh <on|off>}
case "$MODE" in
    on|off) ;;
    *) echo "ERROR: mode must be 'on' or 'off', got '$MODE'"; exit 1 ;;
esac

STEP=${STEP:-23852}
K=${K:-2}
RUN_DIR=${RUN_DIR:-running/ETD_k2_npu}
TASK=${TASK:-gsm8k::olmes}
TASK_NAME="${TASK%%::*}"
NPU_DEVICE_INDEX=${NPU_DEVICE_INDEX:-0}
LIMIT=${LIMIT:-150}

REPLICATION_ROOT=${REPLICATION_ROOT:-/home/n84449292/tiendat/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo}
WORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPORT_DIR=${WORK_ROOT}/kvcache_check/${RUN_DIR}/step${STEP}-hf
OUT_DIR=${WORK_ROOT}/kvcache_check/eval_results/${RUN_DIR}/step${STEP}/${TASK_NAME}-cache-${MODE}
REFERENCE=${REPLICATION_ROOT}/eval_results/${RUN_DIR}/step${STEP}/${TASK_NAME}/metrics.json

echo "=============================================================================="
echo "ETD KV-cache end-to-end check"
echo "  checkpoint : ${REPLICATION_ROOT}/${RUN_DIR}/step${STEP}-unsharded  (read-only)"
echo "  task       : ${TASK}"
echo "  examples   : $( [ "${LIMIT}" = "0" ] && echo "all" || echo "${LIMIT} (first N; same subset for both modes)" )"
echo "  KV cache   : ${MODE}"
echo "  export     : ${EXPORT_DIR}"
echo "  results    : ${OUT_DIR}"
echo "=============================================================================="

# ---------------------------------------------------------------------------------
# 1. Export the checkpoint to HF format using THIS checkout's (fixed) model code.
# ---------------------------------------------------------------------------------
if [ -d "${EXPORT_DIR}" ]; then
    echo "[1/3] HF export already present, reusing it."
else
    echo "[1/3] Exporting checkpoint with the current model code ..."
    mkdir -p "$(dirname "${EXPORT_DIR}")"
    SRC_ROOT="${REPLICATION_ROOT}" DEST_DIR="${EXPORT_DIR}" \
        bash "${WORK_ROOT}/scripts/convert_etd_checkpoint.sh" "${STEP}" "${K}" "${RUN_DIR}"
fi

# ---------------------------------------------------------------------------------
# 2. Set the cache switches in the exported config.
#
# Both are needed and they do different jobs:
#   use_cache      -- HF-level; also seeds generation_config, which is what
#                     generate() actually consults
#   etd_kv_cache   -- our opt-in switch; without it ETD k>1 refuses to cache
# ---------------------------------------------------------------------------------
echo "[2/3] Setting cache switches (mode=${MODE}) ..."
python3 - "$EXPORT_DIR" "$MODE" <<'PY'
import json, os, sys
export_dir, mode = sys.argv[1], sys.argv[2]
want = (mode == "on")

cfg_path = os.path.join(export_dir, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["use_cache"] = want
cfg["etd_kv_cache"] = want
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  config.json      use_cache={want}  etd_kv_cache={want}")

# generation_config.json, if present, overrides config.json inside generate().
gen_path = os.path.join(export_dir, "generation_config.json")
if os.path.exists(gen_path):
    with open(gen_path) as f:
        gen = json.load(f)
    gen["use_cache"] = want
    with open(gen_path, "w") as f:
        json.dump(gen, f, indent=2)
    print(f"  generation_config.json  use_cache={want}")
else:
    print("  generation_config.json  absent (config.json governs)")

k = cfg.get("etd_num_iterations", 1)
enc, think = cfg.get("etd_encoder_layers"), cfg.get("etd_thinking_layers")
n = cfg.get("n_layers")
if want and enc and think:
    print(f"  ETD {enc}-{think}*{k}-{n - enc - think}: expecting {enc + think * k + (n - enc - think)} cache entries per token")
PY

# ---------------------------------------------------------------------------------
# 3. Evaluate.
#
# A dedicated HF_MODULES_CACHE is essential: transformers caches trust_remote_code
# modules by content hash, and sharing the replication run's cache directory risks
# loading a stale pre-fix modeling_olmo.py.
# ---------------------------------------------------------------------------------
echo "[3/3] Running olmes ..."
mkdir -p "${OUT_DIR}"
EXTRA=""
[ "${LIMIT}" != "0" ] && EXTRA="--limit ${LIMIT}"
STARTED=$(date +%s)
OLMO_ROOT="${WORK_ROOT}" \
MODEL_PATH="${EXPORT_DIR}" \
OUTPUT_DIR="${OUT_DIR}" \
HF_MODULES_CACHE="${WORK_ROOT}/kvcache_check/hf_modules_${MODE}" \
NPU_DEVICE_INDEX="${NPU_DEVICE_INDEX}" \
OLMES_EXTRA_ARGS="${EXTRA}" \
    bash "${WORK_ROOT}/scripts/eval_etd_checkpoint_npu.sh" "${STEP}" "${K}" "${RUN_DIR}" "${TASK}"
ELAPSED=$(( $(date +%s) - STARTED ))

# ---------------------------------------------------------------------------------
# Report alongside the stored replication number.
# ---------------------------------------------------------------------------------
echo
echo "=============================================================================="
echo "RESULT  task=${TASK}  cache=${MODE}  wall-clock ${ELAPSED}s"
echo "=============================================================================="
echo "  (wall-clock is itself a signal: if cache=on is not markedly faster than"
echo "   cache=off, caching did not engage and the score means nothing.)"
python3 - "$OUT_DIR" "$REFERENCE" <<'PY'
import json, os, sys

def metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def show(label, data):
    if data is None:
        print(f"  {label:28s} (not found)")
        return
    if isinstance(data, list):
        data = data[0] if data else {}
    interesting = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    print(f"  {label:28s} {json.dumps(interesting, indent=None)[:200]}")

out_dir, reference = sys.argv[1], sys.argv[2]
found = None
for root, _, files in os.walk(out_dir):
    if "metrics.json" in files:
        found = os.path.join(root, "metrics.json")
        break
show("this run", metrics(found) if found else None)
show("stored replication result", metrics(reference))
print()
print("  Compare the primary accuracy field. See scripts/etd_kv_cache_eval_check.sh")
print("  header for how close is close enough.")
PY
