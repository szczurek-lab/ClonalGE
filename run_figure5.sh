#!/bin/bash
# Run ClonalGE on real prostate data to regenerate Figure 5 results.
# Figure 5: Y_validation — comparison of reconstructed vs observed gene expression
# for ClonalGE and Tumoroscope+LR on prostate cancer data.
#
# Usage:
#   bash run_figure5.sh [NUM_RUNS] [START_SEED]
#   NUM_RUNS  — number of independent runs (default: 20, paper used 100)
#   START_SEED — starting random seed (default: 1)

set -e

NUM_RUNS=${1:-20}
START_SEED=${2:-1}
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
    echo ""
    echo "ERROR: Some required data files are missing (listed above)."
    echo "These are the raw prostate cancer data files from Berglund et al. 2018."
    echo "They are not in the git repository. Please obtain them and place them"
    echo "in the paths shown above, then re-run this script."
    exit 1
fi

echo "All input data files found."
echo "Running $NUM_RUNS independent runs (seeds $START_SEED to $((START_SEED + NUM_RUNS - 1)))..."
echo ""

# --- Run ClonalGE on real prostate data ---
# Args: result_dir  config  run_sampling  save_obs  vis_obs  vis_results  saved_inputs  seed
for i in $(seq $START_SEED $((START_SEED + NUM_RUNS - 1))); do
    RESULT_DIR="${RESULT_PREFIX}_${i}"
    echo "=== Run $i / $NUM_RUNS  ->  $RESULT_DIR ==="

    # First run: save observed data + visualize. Subsequent: reuse saved inputs.
    if [ $i -eq $START_SEED ]; then
        python main_real_1000.py "$RESULT_DIR" "$CONFIG" True True True True False $i
    else
        python main_real_1000.py "$RESULT_DIR" "$CONFIG" True False False True False $i
    fi

    echo "=== Run $i done ==="
    echo ""
done

echo "All $NUM_RUNS runs complete. Results in ${RESULT_PREFIX}_*/"
echo ""
echo "To select the best run (highest likelihood) and generate Figure 5,"
echo "compare log-likelihoods across runs and use the best chain's inferred"
echo "H, B, and N to compute Y_hat = t * N * H * B vs observed Y."
