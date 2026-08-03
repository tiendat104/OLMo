#!/bin/bash

SRC=/home/ubuntu/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo/running/replication/ETD_k2
DST=/mnt/data/tiendat/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo/running/replication/ETD_k2

# List the folders you want to move (relative to ETD_k2/)
FOLDERS=(
    step11500-hf
    step11750-hf
    step12000-hf
)

for folder in "${FOLDERS[@]}"; do
    if [ ! -d "$SRC/$folder" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP $folder (not found in source)"
        continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Moving $folder..."
    rsync -ah --progress "$SRC/$folder/" "$DST/$folder/" && rm -rf "$SRC/$folder"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done $folder"
done

echo "All done."
