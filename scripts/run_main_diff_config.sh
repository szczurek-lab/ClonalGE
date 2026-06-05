#!/bin/bash

cd "$(dirname "$0")/.."   # run from repo root

for i in {1..20}
do
    python  main_diff_config.py  $i Results_simulated_data/ configs/  0
done
