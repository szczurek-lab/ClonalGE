"""
Evaluate STdeconvolve vs ClonalGE (5-chain) on simulated data.

For each run/config:
  - Load STdeconvolve theta (S x K) and beta (K x g)
  - Load ground truth H_true (S x K) and B_true (K x g)
  - Apply Hungarian matching (minimize MAE between theta cols and H_true cols)
  - Compute H_MAE and B_MAE for STdeconvolve
  - Load ClonalGE H_SEE and B_SEE from best-chain pickles

Outputs:
  /Volumes/LenovoPS8/ClonalGE/stdeconvolve_comparison.csv
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import pickle
import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/Users/darvis01/Documents/ClonalGE")

SIM_DIR     = "/Volumes/LenovoPS8/ClonalGE/test_sim"
STD_IN_DIR  = "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_inputs"
STD_OUT_DIR = "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_results"
CL_DIR      = "/Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains"
OUT_CSV     = "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_comparison.csv"

RUNS    = range(1, 21)
CONFIGS = ["normal", "low_variance", "high_coverage"]


def hungarian_match(pred, true):
    """Reorder columns of pred (S x K) to best match true (S x K).
    Returns permuted pred and the permutation index."""
    K = true.shape[1]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.mean(np.abs(pred[:, i] - true[:, j]))
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = which ground-truth clone best matches pred topic i
    perm = np.argsort(col_ind)  # reorder pred topics to align with ground truth
    return pred[:, perm], perm


rows = []

for run in RUNS:
    for config in CONFIGS:
        std_path = f"{STD_OUT_DIR}/{run}/{config}"
        inp_path = f"{STD_IN_DIR}/{run}/{config}"
        cl_pkl   = f"{CL_DIR}/{run}/best_oo5_{config}"

        theta_f = f"{std_path}/theta.csv"
        beta_f  = f"{std_path}/beta.csv"
        h_true_f = f"{inp_path}/H_true.csv"
        b_true_f = f"{inp_path}/B_true.csv"

        if not all(os.path.exists(f) for f in [theta_f, beta_f, h_true_f, b_true_f]):
            print(f"MISSING STdeconvolve outputs: run={run} config={config}")
            continue
        if not os.path.exists(cl_pkl):
            print(f"MISSING ClonalGE pickle: {cl_pkl}")
            continue

        theta  = pd.read_csv(theta_f).values   # (S, K)
        beta   = pd.read_csv(beta_f).values     # (K, g)
        H_true = pd.read_csv(h_true_f).values   # (S, K)
        B_true = pd.read_csv(b_true_f).values   # (K, g)

        # Hungarian match on H (spots x clones)
        theta_matched, perm = hungarian_match(theta, H_true)
        beta_matched = beta[perm, :]

        h_mae_std = float(np.mean(np.abs(theta_matched - H_true)))

        # B comparison: both already row-normalized (B_true normalized at export)
        # Normalize beta rows as well (STdeconvolve beta already sums to 1, but confirm)
        beta_row_sums = beta_matched.sum(axis=1, keepdims=True)
        beta_row_sums[beta_row_sums == 0] = 1
        beta_norm = beta_matched / beta_row_sums
        b_mae_std = float(np.mean(np.abs(beta_norm - B_true)))

        # ClonalGE metrics
        cl = pickle.load(open(cl_pkl, "rb"))
        h_mae_cl = float(cl.H_SEE)
        b_mae_cl = float(cl.B_SEE)

        # Also compute ClonalGE B_SEE with same normalization for fair comparison
        B_cl = cl.inferred_B.copy()
        row_sums = B_cl.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        B_cl_norm = B_cl / row_sums
        b_mae_cl_norm = float(np.mean(np.abs(B_cl_norm - B_true)))

        rows.append({
            "run": run,
            "config": config,
            "STdeconvolve_H_MAE": h_mae_std,
            "ClonalGE_H_MAE": h_mae_cl,
            "STdeconvolve_B_MAE": b_mae_std,
            "ClonalGE_B_MAE_raw": b_mae_cl,
            "ClonalGE_B_MAE_norm": b_mae_cl_norm,
        })
        print(f"Run {run:2d} / {config:15s} | "
              f"H: STD={h_mae_std:.4f}  CL={h_mae_cl:.4f} | "
              f"B: STD={b_mae_std:.4f}  CL(norm)={b_mae_cl_norm:.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to {OUT_CSV}")

# Summary table
print("\n=== Summary (mean ± std across 20 runs) ===")
for config in CONFIGS:
    sub = df[df["config"] == config]
    print(f"\n{config}:")
    for col in ["STdeconvolve_H_MAE", "ClonalGE_H_MAE",
                "STdeconvolve_B_MAE", "ClonalGE_B_MAE_norm"]:
        print(f"  {col}: {sub[col].mean():.4f} ± {sub[col].std():.4f}")
