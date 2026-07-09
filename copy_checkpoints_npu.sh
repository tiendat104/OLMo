#!/bin/bash

SRC=/home/n84449292/tiendat/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo/running/ETD_k2_npu
DST=/mnt/pipeline-data/canada_group_folder/tiendat/ETD_k2_npu

# List the folders you want to copy (relative to SRC/) — source is left untouched
FOLDERS=(
    step14400
)

for folder in "${FOLDERS[@]}"; do
    if [ ! -d "$SRC/$folder" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP $folder (not found in source)"
        continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Copying $folder..."
    sudo mkdir -p "$DST/$folder"
    sudo rsync -ah --progress "$SRC/$folder/" "$DST/$folder/"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done $folder"
done

echo "All done."