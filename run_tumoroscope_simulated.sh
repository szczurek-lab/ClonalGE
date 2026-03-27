#!/bin/bash
# Run Tumoroscope on 20 independent simulated datasets (3 conditions each).
# Results go to: /Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope/
#
# Usage: bash run_tumoroscope_simulated.sh
#
# Edit OUTPUT_DIR and PYTHON below if needed.

OUTPUT_DIR="/Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope/"
CONFIG_DIR="configs_tumoroscope"
NOISE=0
PYTHON="/Users/darvis01/anaconda3/envs/general/bin/python"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Output: $OUTPUT_DIR"
echo "Python: $PYTHON"
echo "Configs: $SCRIPT_DIR/$CONFIG_DIR"
echo "Runs: 1..20"
echo ""

for RUN in $(seq 1 20); do
    echo "========================================"
    echo " Run $RUN / 20"
    echo "========================================"
    "$PYTHON" "$SCRIPT_DIR/run_tumoroscope_simulated.py" \
        "$RUN" "$OUTPUT_DIR" "$SCRIPT_DIR/$CONFIG_DIR" "$NOISE"
    echo ""
done

echo "All 20 runs complete."
