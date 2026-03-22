#!/bin/bash
# run_figure2_simulations.sh
# Generates simulated data and runs ClonalGE for all 20 replicates (figure 2)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=20
CONFIG_DIR="configs/"
SIM_DIR="test_sim/"
RESULTS_DIR="Results_simulated_data/"
NOISE=0
LOG_DIR="logs_figure2"

mkdir -p "$LOG_DIR"

echo "=== Step 1: Generating simulated data (${RUNS} replicates) ==="
for i in $(seq 1 $RUNS); do
    echo "  Generating run $i / $RUNS ..."
    python generate_simulations.py $i "$SIM_DIR" "$CONFIG_DIR" $NOISE \
        > "${LOG_DIR}/generate_${i}.log" 2>&1
done
echo "  Done generating."

echo ""
echo "=== Step 2: Running ClonalGE on simulated data (${RUNS} replicates) ==="
for i in $(seq 1 $RUNS); do
    echo "  Running ClonalGE on run $i / $RUNS ..."
    python main_diff_config.py $i "$RESULTS_DIR" "$CONFIG_DIR" $NOISE \
        > "${LOG_DIR}/clonalge_${i}.log" 2>&1
done
echo "  Done."

echo ""
echo "=== All runs completed. Results in: ${RESULTS_DIR} ==="
