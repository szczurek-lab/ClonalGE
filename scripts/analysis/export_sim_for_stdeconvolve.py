"""
Export simulated Y matrices and ground truth H, B for all runs/configs
so STdeconvolve can be run on them in R.

Outputs to: /Volumes/LenovoPS8/ClonalGE/stdeconvolve_inputs/{run}/{config}/
  - Y.csv        : spots x genes count matrix (integer, rounded)
  - H_true.csv   : spots x K ground truth clone proportions
  - B_true.csv   : K x genes ground truth expression (row-normalized to sum=1)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import pickle
import numpy as np
import pandas as pd
import os
import sys

SIM_DIR = "/Volumes/LenovoPS8/ClonalGE/test_sim"
OUT_DIR = "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_inputs"
RUNS = range(1, 21)
CONFIGS = ["normal", "low_variance", "high_coverage"]

os.makedirs(OUT_DIR, exist_ok=True)

for run in RUNS:
    for config in CONFIGS:
        pkl_path = f"{SIM_DIR}/{run}/sample_{config}"
        if not os.path.exists(pkl_path):
            print(f"MISSING: {pkl_path}")
            continue

        out_path = f"{OUT_DIR}/{run}/{config}"
        os.makedirs(out_path, exist_ok=True)

        sim = pickle.load(open(pkl_path, "rb"))

        # Y: (S, g) -> integer counts (STdeconvolve expects count matrix)
        Y = np.round(sim.Y).astype(int)
        Y = np.maximum(Y, 0)
        pd.DataFrame(Y).to_csv(f"{out_path}/Y.csv", index=False)

        # H_true: (S, K)
        pd.DataFrame(sim.H).to_csv(f"{out_path}/H_true.csv", index=False)

        # B_true: (K, g) — row-normalize for scale-free comparison
        B = sim.B.copy()
        row_sums = B.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        B_norm = B / row_sums
        pd.DataFrame(B_norm).to_csv(f"{out_path}/B_true.csv", index=False)

        print(f"Run {run} / {config}: Y{Y.shape}, H{sim.H.shape}, B{sim.B.shape}")

print("Done.")
