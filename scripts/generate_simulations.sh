#!/bin/bash

cd "$(dirname "$0")/.."   # run from repo root

for i in {1..20}
do
    python  generate_simulations.py  $i 'test_sim/' configs/  0
done
