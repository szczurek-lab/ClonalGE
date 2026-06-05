#!/bin/bash

cd "$(dirname "$0")/.."   # run from repo root
# Run ClonalGE on real prostate data to regenerate Figure 5 results.
#
# Usage:
#   bash run_figure5.sh [NUM_RUNS] [START_SEED] [N_JOBS] [SECTION]
#   NUM_RUNS   — number of independent runs (default: 10)
#   START_SEED — starting random seed (default: 1)
#   N_JOBS     — number of parallel jobs (default: number of CPU cores)
#   SECTION    — config section name for result filenames (default: P1.2)

set -e

NUM_RUNS=${1:-10}
START_SEED=${2:-1}
N_JOBS=${3:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}
SECTION=${4:-P1.2}
CONFIG="prostate_data_configs/config_selected_spots_any_mutations_prostate.json"
RESULT_PREFIX="Results_figure5"

# --- Check required data files ---
MISSING=0
for f in \
    prostate_data_configs/vardict2_st_calling/vardict2_1.2.ac \
    prostate_data_configs/vardict2_st_calling/vardict2_2.4.ac \
    prostate_data_configs/vardict2_st_calling/vardict2_3.3.ac \
    prostate_data_configs/STdata/P1.2.tsv \
    prostate_data_configs/STdata/P2.4.tsv \
    prostate_data_configs/STdata/P3.3.tsv \
    prostate_data_configs/C_tree_1.txt \
    prostate_data_configs/F_tree_1.txt \
    prostate_data_configs/ssm_data.txt \
    prostate_data_configs/prostate_cell_count_annotation_any_fixed_margin.txt \
    most_var_genes1000.txt; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "ERROR: Some required data files are missing. See above."
    exit 1
fi

echo "All input data files found."
echo "Running $NUM_RUNS independent runs with $N_JOBS parallel jobs..."
echo ""

# --- Run all jobs in parallel ---
END_SEED=$((START_SEED + NUM_RUNS - 1))
running=0

for i in $(seq $START_SEED $END_SEED); do
    RESULT_DIR="${RESULT_PREFIX}_${i}"
    echo "=== Starting Run $i  ->  $RESULT_DIR ==="
    # Save observed data and visualize only for the first run (non-blocking)
    if [ $i -eq $START_SEED ]; then
        python main_real_1000.py "$RESULT_DIR" "$CONFIG" True True True True False $i &
    else
        python main_real_1000.py "$RESULT_DIR" "$CONFIG" True False False True False $i &
    fi

    running=$((running + 1))
    if [ $running -ge $N_JOBS ]; then
        wait
        running=0
    fi
done

wait
echo ""
echo "All $NUM_RUNS runs complete. Results in ${RESULT_PREFIX}_*/"

echo "Running chain selection post-processing..."
python select_chains.py "$RESULT_PREFIX" "$NUM_RUNS" "$START_SEED" "$SECTION"
echo "Chain selection complete."
