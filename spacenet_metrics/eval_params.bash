#!/bin/bash

base_dir="save"

echo "================================================="
echo "  Starting Evaluation Pipeline..."
echo "================================================="

# Iterate through folders inside ../save/
for full_path in ../save/exp_*; do
    # Check if it's a valid directory
    if [ -d "$full_path" ]; then
        # Extract just the folder name (e.g., exp_base_ext)
        exp_name=$(basename "$full_path")
        
        # Construct the clean relative path that apls.py expects
        target_dir="$base_dir/$exp_name"
        
        echo ""
        echo "========= Evaluating $exp_name ========="
        
        if [ ! -f "$full_path/results/apls.json" ]; then
            echo "    -> [RUNNING] Calculating APLS..."
            # Call the shell wrapper script in the current directory
            bash ./apls.bash "$target_dir"
        else
            echo "    -> [SKIPPED] APLS already exists."
        fi

        if [ ! -f "$full_path/results/topo.json" ]; then
            echo "    -> [RUNNING] Calculating TOPO..."
            bash ./topo.bash "$target_dir"
        else
            echo "    -> [SKIPPED] TOPO already exists."
        fi
    fi
done

echo ""
echo "================================================="
echo "  All evaluations finished!"
echo "  Please run 'python params_aggregate_rsts.py' in the root directory."
echo "================================================="
