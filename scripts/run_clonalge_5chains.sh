#!/bin/bash
# Run ClonalGE with 5 chains on 20 simulated datasets (3 conditions each).
# Results go to a separate directory so existing 1-chain results are untouched.
#
# Usage: bash run_clonalge_5chains.sh
#
OUTPUT_DIR="./Results_simulated_5chains/"
PYTHON="/home/shafighi/miniforge3/envs/clonalge/bin/python"

# 5 chains, 5 parallel workers
export CLONALGE_CHAINS=5
export CLONALGE_CORES=5

NOISE=0
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Output : $OUTPUT_DIR"
echo "Python : $PYTHON"
echo "Chains : $CLONALGE_CHAINS"
echo "Cores  : $CLONALGE_CORES"
echo ""

# cd into ClonalGE dir so config paths stay relative (main_diff_config.py
# uses re.split("/|.json") which breaks on absolute paths)
cd "$REPO_ROOT"

for RUN in $(seq 1 20); do
    echo "========================================"
    echo " Run $RUN / 20"
    echo "========================================"
    "$PYTHON" main_diff_config.py "$RUN" "$OUTPUT_DIR" configs/ "$NOISE"
    echo ""
done

echo "All 20 runs complete."
