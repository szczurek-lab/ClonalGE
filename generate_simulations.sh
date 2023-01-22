#!/bin/bash

for i in {1..20}
do
    python  generate_simulations.py  $i 'test_sim/' configs/  0
done
