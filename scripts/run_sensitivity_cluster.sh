#!/bin/bash
# Run all 45 sensitivity tasks in parallel on a multi-core machine (no SLURM).
# Usage: bash run_sensitivity_cluster.sh [max_parallel]
# Default: run all 45 at once. Pass a number to cap concurrency (e.g. 20).

cd "$(dirname "$0")/.."   # run from repo root

# ── Adjust these paths ────────────────────────────────────────────────────────
BASE=$(pwd)   # directory containing test_sim/ and sensitivity_results/
CODE=$(pwd)   # directory containing run_sensitivity_one.py
PYTHON=python
# ─────────────────────────────────────────────────────────────────────────────

MAX_PAR=${1:-45}   # max parallel jobs
mkdir -p "$BASE/sensitivity_results" "$CODE/logs"
cd "$CODE"

running=0
for task_id in $(seq 1 45); do
    log="$CODE/logs/sensitivity_${task_id}.log"
    csv="$BASE/sensitivity_results/task$(printf '%03d' $task_id)_"*.csv 2>/dev/null
    # skip if already done
    if ls "$BASE/sensitivity_results/task$(printf '%03d' $task_id)"_*.csv 2>/dev/null | grep -q .; then
        echo "Task $task_id: already done, skipping"
        continue
    fi

    echo "Starting task $task_id ..."
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PYTHON run_sensitivity_one.py \
        --sim_dir "$BASE/test_sim" \
        --outdir  "$BASE/sensitivity_results" \
        --task_id "$task_id" \
        > "$log" 2>&1 &

    running=$((running + 1))
    if [ "$running" -ge "$MAX_PAR" ]; then
        wait -n 2>/dev/null || wait   # wait for one to finish, then continue
        running=$((running - 1))
    fi
done

wait
echo "All sensitivity tasks done."
echo "Generate figure with:"
echo "  python plot_sensitivity.py --dir $BASE/sensitivity_results --output plots_paper/sensitivity_analysis.png"
