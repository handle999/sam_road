#!/bin/bash
# ==========================================================
# SAM-Road Evaluation Pipeline
# Location: Must be inside spacenet_metrics/ folder
# ==========================================================

SAVE_DIR="../save"

echo "================================================="
echo "  Starting Evaluation Pipeline..."
echo "================================================="

for EXP_DIR in "$SAVE_DIR"/exp_*; do
    if [ -d "$EXP_DIR" ]; then
        EXP_ID=$(basename "$EXP_DIR")
        echo ""
        echo ">>> Evaluating: $EXP_ID"

        # ------------------------------------------------------
        # [Metric 1]: APLS
        # ------------------------------------------------------
        APLS_OUT="$EXP_DIR/apls_result.json"
        if [ ! -f "$APLS_OUT" ]; then
            echo "    -> [RUNNING] Calculating APLS..."
            python apls.py --dir "$EXP_DIR"
        else
            echo "    -> [SKIPPED] APLS already exists."
        fi

        # ------------------------------------------------------
        # [Metric 2]: TOPO
        # ------------------------------------------------------
        TOPO_OUT="$EXP_DIR/topo_result.json"
        if [ ! -f "$TOPO_OUT" ]; then
            echo "    -> [RUNNING] Calculating TOPO..."
            python topo.py -savedir "$EXP_DIR"
        else
            echo "    -> [SKIPPED] TOPO already exists."
        fi
    fi
done

echo ""
echo "================================================="
echo "  All evaluations finished!"
echo "  Triggering Result Aggregator..."
echo "================================================="

# Call the aggregator script located in the parent folder
python ../params_aggregate_rsts.py
