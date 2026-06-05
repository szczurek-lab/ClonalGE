#!/bin/bash
# ============================================================
# SLURM array job: run Tumoroscope on 20 simulated datasets
#
# Submit with:
#   sbatch run_tumoroscope_simulated_slurm.sh
#
# Each array task (1..20) runs all 3 conditions for one replicate.
# ============================================================

#SBATCH --job-name=tum_sim
#SBATCH --array=1-20
#SBATCH --time=06:00:00          # 6 h per replicate (3 conditions × ~2 h each)
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/tum_%A_%a.out
#SBATCH --error=logs/tum_%A_%a.err

# ── TODO: set these paths for your cluster ──────────────────
CLONALGE_DIR="/path/to/ClonalGE"            # directory containing run_tumoroscope_simulated.py
TUMOROSCOPE_SRC="/path/to/Tumoroscope/src"  # directory containing tumoroscope/ package
RESULTS_DIR="/path/to/Results_simulated_tumoroscope/"
CONFIG_DIR="${CLONALGE_DIR}/configs_tumoroscope"
NOISE=0

# Optional: reuse existing ClonalGE simulation objects (paired comparison).
# Set to the directory where generate_simulations.sh stored test_sim/
# If the folder does not exist, fresh simulations will be generated instead.
export SIM_DIR="/path/to/test_sim"   # remove or leave blank to generate new sims

# ── TODO: activate your ClonalGE conda environment ─────────
# Common patterns — uncomment the one that matches your cluster:
#
# Option A — conda activate (if conda is already initialised in the shell)
# conda activate clonalge
#
# Option B — source activate (older conda)
# source activate clonalge
#
# Option C — module + conda (many HPC clusters)
# module load anaconda3
# source activate clonalge
#
# Option D — explicit python path (most reliable, no shell init needed)
# PYTHON="/path/to/conda/envs/clonalge/bin/python"

PYTHON="${PYTHON:-python}"   # falls back to whichever python is in PATH
# ────────────────────────────────────────────────────────────

export TUMOROSCOPE_SRC

mkdir -p "${RESULTS_DIR}" logs

echo "============================="
echo "SLURM array task : $SLURM_ARRAY_TASK_ID"
echo "ClonalGE dir     : $CLONALGE_DIR"
echo "Tumoroscope src  : $TUMOROSCOPE_SRC"
echo "Results dir      : $RESULTS_DIR"
echo "SIM_DIR          : ${SIM_DIR:-<generate new>}"
echo "Python           : $(which $PYTHON)"
echo "============================="

cd "$CLONALGE_DIR"

"$PYTHON" run_tumoroscope_simulated.py \
    "$SLURM_ARRAY_TASK_ID" \
    "$RESULTS_DIR" \
    "$CONFIG_DIR" \
    "$NOISE"

echo "Run $SLURM_ARRAY_TASK_ID complete."
