#!/bin/bash
# run_figure2_simulations.sh
# Generates simulated data and runs ClonalGE for all 20 replicates (figure 2)
#
# Usage inside screen:
#   screen -S figure2
#   bash run_figure2_simulations.sh
#   Ctrl+A, D  to detach
#   screen -r figure2  to reattach

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNS=20
CONFIG_DIR="configs/"
SIM_DIR="test_sim/"
RESULTS_DIR="Results_simulated_data/"
NOISE=0
LOG_DIR="logs_figure2"

# Set to true to skip generation when data already exists on the server.
# Per-run: generation is also skipped automatically if all config sample
# files are found under ${SIM_DIR}/<run>/ regardless of this flag.
SKIP_GENERATION=true

mkdir -p "$LOG_DIR"

# Prevent OpenBLAS/MKL from spawning internal threads. When Python's
# multiprocessing uses fork(), child processes inherit corrupted BLAS thread
# state, causing "corrupted size vs. prev_size" heap crashes.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLAS_NUM_THREADS=1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Returns 0 (true) if all config sample files exist for run $1
sim_data_exists() {
    local run=$1
    for cfg in normal low_variance high_coverage; do
        if [ ! -f "${SIM_DIR}${run}/sample_${cfg}" ]; then
            return 1
        fi
    done
    return 0
}

echo "[$(timestamp)] === Step 1: Generating simulated data (${RUNS} replicates) ==="
if [ "$SKIP_GENERATION" = true ]; then
    echo "[$(timestamp)]   SKIP_GENERATION=true — checking existing files ..."
fi
for i in $(seq 1 $RUNS); do
    if sim_data_exists $i; then
        echo "[$(timestamp)]   Run $i — data already exists, skipping generation."
    elif [ "$SKIP_GENERATION" = true ]; then
        echo "[$(timestamp)]   Run $i — data missing but SKIP_GENERATION=true, skipping."
    else
        echo "[$(timestamp)]   Generating run $i / $RUNS ..."
        if python generate_simulations.py $i "$SIM_DIR" "$CONFIG_DIR" $NOISE \
            > "${LOG_DIR}/generate_${i}.log" 2>&1; then
            echo "[$(timestamp)]   Run $i generation OK"
        else
            echo "[$(timestamp)]   ERROR in generation run $i — see ${LOG_DIR}/generate_${i}.log"
        fi
    fi
done
echo "[$(timestamp)]   Done with Step 1."

echo ""
echo "[$(timestamp)] === Step 2: Running ClonalGE on simulated data (${RUNS} replicates) ==="
for i in $(seq 1 $RUNS); do
    echo "[$(timestamp)]   Running ClonalGE run $i / $RUNS ..."
    if python main_diff_config.py $i "$RESULTS_DIR" "$CONFIG_DIR" $NOISE \
        > "${LOG_DIR}/clonalge_${i}.log" 2>&1; then
        echo "[$(timestamp)]   Run $i ClonalGE OK"
    else
        echo "[$(timestamp)]   ERROR in ClonalGE run $i — see ${LOG_DIR}/clonalge_${i}.log"
    fi
done

echo ""
echo "[$(timestamp)] === Step 3: Running scalability / runtime analysis ==="
if python run_scalability.py \
    --iters 500 \
    --outdir results/scalability \
    > "${LOG_DIR}/scalability.log" 2>&1; then
    echo "[$(timestamp)]   Scalability analysis OK"
else
    echo "[$(timestamp)]   ERROR in scalability analysis — see ${LOG_DIR}/scalability.log"
fi

echo ""
echo "[$(timestamp)] === Step 4: Plotting runtime figure ==="
mkdir -p plots_paper
if python plot_runtime.py \
    --csv results/scalability/runtime_results.csv \
    --output plots_paper/runtime.png \
    > "${LOG_DIR}/plot_runtime.log" 2>&1; then
    echo "[$(timestamp)]   Runtime figure saved to plots_paper/runtime.png"
else
    echo "[$(timestamp)]   ERROR plotting runtime — see ${LOG_DIR}/plot_runtime.log"
fi

echo ""
echo "[$(timestamp)] === All steps completed. ==="
echo "[$(timestamp)]   Simulation results : ${RESULTS_DIR}"
echo "[$(timestamp)]   Scalability CSV    : results/scalability/runtime_results.csv"
echo "[$(timestamp)]   Runtime figure     : plots_paper/runtime.png"
