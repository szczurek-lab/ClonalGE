#!/bin/bash
# Run Tumoroscope on 20 simulated datasets (3 conditions each), no SLURM.
# Runs sequentially: 1 → 20.
#
# Usage: bash run_tumoroscope_simulated.sh
#
# ── TODO: set these paths ────────────────────────────────────────────────────
OUTPUT_DIR="/path/to/Results_simulated_tumoroscope/"
PYTHON="/path/to/conda/envs/clonalge/bin/python"

# Point to the test_sim/ directory produced by generate_simulations.sh
# Each run's simulation is at: $SIM_DIR/{run_number}/sample_{condition}
export SIM_DIR="/path/to/test_sim"
# ─────────────────────────────────────────────────────────────────────────────

NOISE=0
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs_tumoroscope"

echo "Output : $OUTPUT_DIR"
echo "Python : $PYTHON"
echo "Configs: $CONFIG_DIR"
echo "SIM_DIR: $SIM_DIR"
echo "Runs   : 1..20"
echo ""

for RUN in $(seq 1 20); do
    echo "========================================"
    echo " Run $RUN / 20"
    echo "========================================"
    "$PYTHON" "$SCRIPT_DIR/run_tumoroscope_simulated.py" \
        "$RUN" "$OUTPUT_DIR" "$CONFIG_DIR" "$NOISE"
    echo ""
done

echo "All 20 runs complete."
