#!/bin/bash
# Generate and run ClonalGE on assumption-violating simulations (20 replicates).
# Adjust SIM_DIR and RESULTS_DIR to match your storage paths.

SIM_DIR="/Volumes/LenovoPS8/ClonalGE/test_sim_violated"
RESULTS_NORMAL="/Volumes/LenovoPS8/ClonalGE/Results_violated_normal"
RESULTS_ZINB="/Volumes/LenovoPS8/ClonalGE/Results_violated_zinb"
RESULTS_BATCH="/Volumes/LenovoPS8/ClonalGE/Results_violated_batch"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

# Step 1: generate violated simulations
echo "=== Generating violated simulations ==="
for i in $(seq 1 20); do
    python generate_simulations_violated.py "$i" "$SIM_DIR"
done

# Step 2: run ClonalGE on each variant
echo "=== Running ClonalGE: normal ==="
for i in $(seq 1 20); do
    python main_violated.py "$i" "$RESULTS_NORMAL" configs/ 0 normal
done

echo "=== Running ClonalGE: zinb ==="
for i in $(seq 1 20); do
    python main_violated.py "$i" "$RESULTS_ZINB" configs/ 0 zinb
done

echo "=== Running ClonalGE: batch ==="
for i in $(seq 1 20); do
    python main_violated.py "$i" "$RESULTS_BATCH" configs/ 0 batch
done

echo "=== Done. Run plot_violated_results.py to generate figure. ==="
