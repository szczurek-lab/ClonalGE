#!/bin/bash
# Run all 60 violated simulation tasks in parallel on a multi-core machine (no SLURM).
# Usage: bash run_violated_cluster.sh [max_parallel]
# Default: run all 60 at once. Pass a number to cap concurrency (e.g. 20).

cd "$(dirname "$0")/.."   # run from repo root

# ── Adjust these paths ────────────────────────────────────────────────────────
BASE=$(pwd)   # directory containing test_sim_violated/ and Results_violated_*/
CODE=$(pwd)   # directory containing main_violated.py
PYTHON=python
# ─────────────────────────────────────────────────────────────────────────────

MAX_PAR=${1:-60}
mkdir -p "$CODE/logs"
cd "$CODE"

run_one() {
    local IDX=$1
    if   [ $IDX -le 20 ]; then VARIANT=normal; RUN=$IDX
    elif [ $IDX -le 40 ]; then VARIANT=zinb;   RUN=$((IDX - 20))
    else                        VARIANT=batch;  RUN=$((IDX - 40))
    fi

    RDIR="${BASE}/Results_violated_${VARIANT}"
    RESULT_TXT="${RDIR}${RUN}/results_normal.txt"
    LOG="$CODE/logs/violated_${VARIANT}_${RUN}.log"

    if grep -q "Variant:" "$RESULT_TXT" 2>/dev/null; then
        echo "Task $IDX ($VARIANT run $RUN): already done, skipping"
        return
    fi

    echo "Starting $VARIANT run $RUN ..."
    CLONALGE_CHAINS=1 CLONALGE_CORES=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PYTHON main_violated.py "$RUN" "$RDIR" "${CODE}/configs/" 0 "$VARIANT" \
        > "$LOG" 2>&1
    echo "Done: $VARIANT run $RUN  (exit $?)"
}
export -f run_one
export BASE CODE PYTHON

running=0
for idx in $(seq 1 60); do
    run_one "$idx" &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_PAR" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done

wait
echo "All violated tasks done."
echo "Generate figure with:"
echo "  python plot_violated_results.py --base_dir $BASE --num_runs 20 --output plots_paper/violated_simulations.png"
