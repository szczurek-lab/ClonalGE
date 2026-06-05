#!/bin/bash

cd "$(dirname "$0")/.."   # run from repo root
# Usage: bash run_main_real_1000.sh [N_JOBS]
# N_JOBS — number of parallel jobs (default: number of CPU cores)

N_JOBS=${1:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}
CONFIG="prostate_data_configs/config_selected_spots_any_mutations_prostate.json"

echo "Running 100 independent runs with $N_JOBS parallel jobs..."

running=0
for i in $(seq 1 100); do
    python main_real_1000.py Results_real_data_$i "$CONFIG" True True True True False $i &

    running=$((running + 1))
    if [ $running -ge $N_JOBS ]; then
        wait
        running=0
    fi
done

wait
echo "All 100 runs complete."
