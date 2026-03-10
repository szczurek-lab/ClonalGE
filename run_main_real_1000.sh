#!/bin/bash
# Usage: bash run_main_real_1000.sh [N_JOBS]
# N_JOBS — number of parallel jobs (default: number of CPU cores)

N_JOBS=${1:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}
CONFIG="prostate_data_configs/config_selected_spots_any_mutations_prostate.json"

echo "Running 100 independent runs with $N_JOBS parallel jobs..."

run_job() {
    local i=$1
    python main_real_1000.py Results_real_data_$i \
        prostate_data_configs/config_selected_spots_any_mutations_prostate.json \
        True True True True False $i
}
export -f run_job

seq 1 100 | xargs -P "$N_JOBS" -I{} bash -c 'run_job "$@"' _ {}

echo "All 100 runs complete."
