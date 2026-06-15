#!/bin/bash
# Usage: bash scripts/convert_etd_checkpoint.sh <step> [run_dir]
# Example: bash scripts/convert_etd_checkpoint.sh 5
#          bash scripts/convert_etd_checkpoint.sh 10 running/sanity/ETD_k2_per_step

set -e

STEP=$1
RUN_DIR=${2:-running/sanity/ETD_k2_per_step}
TOKENIZER=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/rep_olmo2_1B_midtrain/OLMo/olmo_data/tokenizers/allenai_dolma2.json

if [ -z "$STEP" ]; then
    echo "Usage: bash scripts/convert_etd_checkpoint.sh <step> [run_dir]"
    exit 1
fi

CHECKPOINT_DIR=${RUN_DIR}/step${STEP}-unsharded
DEST_DIR=${RUN_DIR}/step${STEP}-hf

echo "Converting step ${STEP}: ${CHECKPOINT_DIR} -> ${DEST_DIR}"

# Convert to HF format
python hf_olmo/convert_olmo_to_hf.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --destination-dir "${DEST_DIR}" \
    --tokenizer "${TOKENIZER}"

# Copy custom modeling files
cp hf_olmo/modeling_olmo.py "${DEST_DIR}/"
cp hf_olmo/configuration_olmo.py "${DEST_DIR}/"

# Add auto_map to config.json
python3 << EOF
import json
p = "${DEST_DIR}/config.json"
with open(p) as f: cfg = json.load(f)
cfg['auto_map'] = {
    'AutoConfig': 'configuration_olmo.OLMoConfig',
    'AutoModelForCausalLM': 'modeling_olmo.OLMoForCausalLM'
}
with open(p, 'w') as f: json.dump(cfg, f, indent=2)
print('auto_map added to config.json')
EOF

echo "Done: ${DEST_DIR}"
