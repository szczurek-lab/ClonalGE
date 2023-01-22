#!/bin/bash
for i in {1..100}
do
  python main_real_1000.py Results_real_data_$i prostate_data_configs/config_selected_spots_any_mutations_prostate.json True True True True False $i
done
