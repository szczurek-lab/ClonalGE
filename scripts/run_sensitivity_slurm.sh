#!/bin/bash
#SBATCH --job-name=sensitivity
#SBATCH --output=logs/sensitivity_%A_%a.log
#SBATCH --array=1-45
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00

# 45 tasks = 3 reps × 3 params × 5 factors
# Task ID mapping: task = (rep-1)*15 + param_idx*5 + factor_idx + 1

# ── Adjust these paths ────────────────────────────────────────────────────────
BASE=$(pwd)   # directory containing test_sim/ and sensitivity_results/
CODE=$(pwd)   # directory containing run_sensitivity_one.py, clonalGE.py, etc.
PYTHON=python
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$CODE/logs"
cd "$CODE"

$PYTHON run_sensitivity_one.py \
    --sim_dir  "${BASE}/test_sim" \
    --outdir   "${BASE}/sensitivity_results" \
    --task_id  "$SLURM_ARRAY_TASK_ID"
