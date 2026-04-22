#!/bin/bash
#SBATCH --job-name=violated
#SBATCH --output=logs/violated_%A_%a.log
#SBATCH --array=1-60
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Map array index to (variant, run_num)
# Index 1-20  → normal,  run 1-20
# Index 21-40 → zinb,    run 1-20
# Index 41-60 → batch,   run 1-20

# ── Adjust these paths ────────────────────────────────────────────────────────
BASE=/path/to/ClonalGE/data          # directory containing test_sim_violated/ and Results_violated_*/
CODE=/path/to/ClonalGE               # directory containing main_violated.py, clonalGE.py, etc.
PYTHON=python                        # python executable
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$CODE/logs"
cd "$CODE"

IDX=$SLURM_ARRAY_TASK_ID
if   [ $IDX -le 20 ]; then VARIANT=normal; RUN=$IDX
elif [ $IDX -le 40 ]; then VARIANT=zinb;   RUN=$((IDX - 20))
else                        VARIANT=batch;  RUN=$((IDX - 40))
fi

RDIR="${BASE}/Results_violated_${VARIANT}"
RESULT_TXT="${RDIR}${RUN}/results_normal.txt"

# Skip if already done
if grep -q "Variant:" "$RESULT_TXT" 2>/dev/null; then
    echo "Skipping $VARIANT run $RUN (already done)"
    exit 0
fi

CLONALGE_CHAINS=1 CLONALGE_CORES=1 \
$PYTHON main_violated.py $RUN $RDIR "${CODE}/configs/" 0 $VARIANT
