#!/bin/bash

skill_name="insert"
perception_dir="./dump/perception/Stow_insert/23-May-2023-12:35:57.698997_corner"

inspect_dir="./dump/perception/inspect/$skill_name"
mkdir -p "$inspect_dir"

for sub_dir in "$perception_dir"/*/; do
    file="$sub_dir/004_rgb_agentview.png"
    if [[ -f "$file" ]]; then
        echo "$sub_dir"
        vid_idx=$(basename -- "$sub_dir")
        cp "$file" "$inspect_dir/$vid_idx.png"
    fi
done

touch "$inspect_dir/inspect.txt"
