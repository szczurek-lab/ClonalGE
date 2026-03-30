#!/bin/bash
# Run ClonalGE with 5 chains on 20 simulated datasets (3 conditions each).
# Results go to a separate directory so existing 1-chain results are untouched.
#
# Usage: bash run_clonalge_5chains.sh
#
# ── TODO: set these paths ────────────────────────────────────────────────────
OUTPUT_DIR="/path/to/Results_simulated_5chains/"
PYTHON="/path/to/conda/envs/clonalge/bin/python"
# ─────────────────────────────────────────────────────────────────────────────

# 5 chains, 5 parallel workers
export CLONALGE_CHAINS=5
export CLONALGE_CORES=5

CONFIG_DIR="configs"
NOISE=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Output : $OUTPUT_DIR"
echo "Python : $PYTHON"
echo "Chains : $CLONALGE_CHAINS"
echo "Cores  : $CLONALGE_CORES"
echo ""

for RUN in $(seq 1 20); do
    echo "========================================"
    echo " Run $RUN / 20"
    echo "========================================"
    "$PYTHON" "$SCRIPT_DIR/main_diff_config.py" \
        "$RUN" "$OUTPUT_DIR" "$SCRIPT_DIR/$CONFIG_DIR" "$NOISE"
    echo ""
done

echo "All 20 runs complete."
