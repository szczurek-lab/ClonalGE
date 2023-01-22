#!/bin/bash

for i in {1..20}
do
    python  main_diff_config.py  $i Results_simulated_data/ configs/  0
done
