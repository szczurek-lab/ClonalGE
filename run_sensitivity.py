"""
Hyperparameter sensitivity analysis for ClonalGE.

For each perturbation factor in {0.50, 0.75, 1.00, 1.25, 1.50} applied
independently to alpha_g, beta, and p_g, runs ClonalGE on REPS replicates
of the 'normal' simulation config and reports H_SEE and B_SEE (rMAE).

Output:
    <outdir>/sensitivity_results.csv

Usage:
    python run_sensitivity.py \\
        --sim_dir   /Volumes/LenovoPS8/ClonalGE/test_sim \\
        --outdir    /Volumes/LenovoPS8/ClonalGE/sensitivity_results \\
        --reps      3
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import pickle
import random
import time
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from multiprocessing import Pool

from clonalGE import clonalGE
import tumoroscope as tum

# ── constants ──────────────────────────────────────────────────────────────────
FACTORS   = [0.50, 0.75, 1.00, 1.25, 1.50]
PARAMS    = ['alpha_g', 'beta', 'p_g']
CONFIG    = 'normal'
CHAINS    = 1
MIN_ITER  = 3000
MAX_ITER  = 10000
BURN_IN   = 500
BATCH     = 1000
EVERY_N   = 5
CHANGES_B = 500
OPT_RATE  = 0.4

# ── helpers ───────────────────────────────────────────────────────────────────

def calc_B(Y, H, N):
    K, G = H.shape[1], Y.shape[1]
    X = np.diag(N) @ H
    B = np.zeros((K, G))
    for g in range(G):
        if Y[:, g].sum() > 0:
            B[:, g] = lsq_linear(X, Y[:, g], bounds=(0, np.inf)).x
    return B


def see(inferred, true):
    return float(np.sqrt(np.mean((inferred - true) ** 2)))


def rmae(inferred, true):
    m = np.mean(np.abs(true))
    return float(np.mean(np.abs(inferred - true)) / m) if m > 0 else float('nan')


def run_chain(args):
    tum_obj, cl_obj, sim, min_iter, max_iter, batch, every_n = args
    tum_obj.gibbs_sampling(
        seed=random.randint(1, 10000), min_iter=int(min_iter / 1.5),
        max_iter=int(max_iter / 2), burn_in=BURN_IN // 2,
        batch=batch, simulated_data=sim, n_sampling=True,
        F_fraction=False, theta_variable=False, pi_2D=True,
        th=0.8, every_n_sample=every_n, changes_batch=CHANGES_B,
        var_calculation=int(min_iter * 0.9))
    B_init = calc_B(sim.Y, tum_obj.inferred_H, tum_obj.inferred_n)
    inits = (tum_obj.inferred_n, tum_obj.inferred_H, tum_obj.inferred_G,
             tum_obj.inferred_pi, tum_obj.inferred_phi, tum_obj.inferred_Z, B_init)
    cl_obj.gibbs_sampling(
        seed=random.randint(1, 10000), min_iter=min_iter, max_iter=max_iter,
        batch=batch, simulated_data=sim, n_sampling=True,
        F_fraction=False, pi_2D=True, th=0.8, every_n_sample=every_n,
        changes_batch=CHANGES_B)
    return cl_obj


def run_one(sim, b_alpha_factor, b_beta_factor, p_y_factor, run_id, outdir):
    """Run ClonalGE with perturbed hyperparameters; return (H_rMAE, B_rMAE)."""
    b_alpha = np.clip(sim.b_alpha * b_alpha_factor, 1e-6, None)
    b_beta  = float(sim.b_beta * b_beta_factor)
    p_y     = np.clip(sim.p_y * p_y_factor, 1e-6, 1 - 1e-6)

    config_file = 'configs/normal.json'
    with open(config_file) as f:
        cfg = json.load(f)

    K = sim.K; S = sim.S; I = sim.I; g = sim.g
    F_epsilon = np.tile(cfg['Gamma']['F_epsilon'], (K, 1))
    F         = np.tile(cfg['Gamma']['F'],         (K, 1))
    result_txt = os.path.join(outdir, f'run{run_id}_a{b_alpha_factor}_b{b_beta_factor}_p{p_y_factor}.txt')

    tum_obj = tum.tumoroscope(
        name=result_txt + '_tum', K=K, S=S, r=None, p=None, I=I,
        avarage_clone_in_spot=sim.avarage_clone_in_spot, F=F, C=sim.C,
        A=sim.A, D=sim.D, F_epsilon=F_epsilon, optimal_rate=OPT_RATE,
        n_lambda=sim.n, gamma=0.95, pi_2D=True, result_txt=result_txt,
        rp_est_method='my')

    cl_obj = clonalGE(
        name=result_txt + '_cl', K=K, S=S, g=g, r=None, q=None, I=I,
        avarage_clone_in_spot=sim.avarage_clone_in_spot, F=F, C=sim.C,
        A=sim.A, D=sim.D, F_epsilon=F_epsilon, optimal_rate=OPT_RATE,
        n_lambda=sim.n, pi_2D=True, result_txt=result_txt,
        Y=sim.Y, p_y=p_y, b_alpha=b_alpha, b_beta=b_beta)

    tum_obj.gibbs_sampling(
        seed=random.randint(1, 10000), min_iter=int(MIN_ITER / 1.5),
        max_iter=int(MAX_ITER / 2), burn_in=BURN_IN // 2, batch=BATCH,
        simulated_data=sim, n_sampling=True, F_fraction=False,
        theta_variable=False, pi_2D=True, th=0.8, every_n_sample=EVERY_N,
        changes_batch=CHANGES_B, var_calculation=int(MIN_ITER * 0.9))

    B_init = calc_B(sim.Y, tum_obj.inferred_H, tum_obj.inferred_n)
    inits = (tum_obj.inferred_n, tum_obj.inferred_H, tum_obj.inferred_G,
             tum_obj.inferred_pi, tum_obj.inferred_phi, tum_obj.inferred_Z, B_init)
    cl_obj.inits = inits  # propagate inits manually

    cl_obj.gibbs_sampling(
        seed=random.randint(1, 10000), min_iter=MIN_ITER, max_iter=MAX_ITER,
        batch=BATCH, simulated_data=sim, n_sampling=True, F_fraction=False,
        pi_2D=True, th=0.8, every_n_sample=EVERY_N, changes_batch=CHANGES_B)

    h_rmae = rmae(cl_obj.inferred_H, sim.H)
    b_rmae = rmae(cl_obj.inferred_B, sim.B)
    return h_rmae, b_rmae


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim_dir', required=True)
    ap.add_argument('--outdir',  required=True)
    ap.add_argument('--reps',    type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for rep in range(1, args.reps + 1):
        sim_path = os.path.join(args.sim_dir, str(rep), f'sample_{CONFIG}')
        if not os.path.exists(sim_path):
            print(f'Missing {sim_path}, skipping')
            continue
        sim = pickle.load(open(sim_path, 'rb'))
        print(f'Rep {rep}')

        for param in PARAMS:
            for factor in FACTORS:
                a_f = factor if param == 'alpha_g' else 1.0
                b_f = factor if param == 'beta'    else 1.0
                p_f = factor if param == 'p_g'     else 1.0

                print(f'  {param} × {factor}', end=' ', flush=True)
                t0 = time.time()
                try:
                    h_rmae, b_rmae = run_one(sim, a_f, b_f, p_f,
                                             run_id=rep, outdir=args.outdir)
                    elapsed = time.time() - t0
                    print(f'H_rMAE={h_rmae:.4f}  B_rMAE={b_rmae:.4f}  ({elapsed:.0f}s)')
                except Exception as e:
                    print(f'ERROR: {e}')
                    h_rmae = b_rmae = float('nan')

                rows.append(dict(rep=rep, param=param, factor=factor,
                                 H_rMAE=h_rmae, B_rMAE=b_rmae))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, 'sensitivity_results.csv')
    df.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}')


if __name__ == '__main__':
    main()
