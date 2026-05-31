#!/usr/bin/env python
"""
run_scalability.py
------------------
Wall-clock runtime analysis for ClonalGE.

Measures per-iteration time for:
  1. The three baseline simulation configs (normal, low_variance, high_coverage)
  2. Scalability sweeps: S, g, I varied independently (K=5 fixed)

Each ClonalGE experiment runs TIMING_ITERS Gibbs iterations (min_iter set
much larger so early-stopping never fires).  Workers run in parallel via
multiprocessing.Pool.

Usage:
    python run_scalability.py [--workers N] [--iters N] [--outdir DIR]

Outputs:
    <outdir>/runtime_results.csv
    <outdir>/tmp_logs/<experiment>.log   (per-worker stdout/stderr)
"""

import argparse
import csv
import multiprocessing
import os
import sys
import time

import numpy as np
import scipy.special as sc
from scipy.optimize import lsq_linear

import simulation as sim
import tumoroscope as tum
from clonalGE import clonalGE

# ── timing parameters ─────────────────────────────────────────────────────────
TIMING_ITERS  = 500    # ClonalGE iterations per timing experiment
TUM_MAX_ITER  = 400    # short Tumoroscope run for inits
TUM_BATCH     = 200
BATCH         = 200    # ClonalGE batch size (for internal bookkeeping only)
EVERY_N       = 5      # thinning factor

# ── model constants (from normal.json) ────────────────────────────────────────
K            = 5
AVG_CLONE    = 2.5
OPTIMAL_RATE = 0.4
TH           = 0.8
PI_2D        = True
N_LAMBDA     = 45
GAMMA        = 0.95
F_BASE       = [30, 1]
F_EPS_BASE   = [2,  1]
PHI_GAMMA    = [0.005, 1]
B_ALPHA_SHAPE = 1
B_BETA        = 3.5
P_MEAN        = 0.045
P_STD         = 0.02
B_ALPHA_VAR   = 0.25

# C template rows (6-mutation-pattern from normal.json, tiled to fill any I)
C_TEMPLATE = np.array([
    [0, 1, 0.5, 1,   1  ],
    [0, 1, 0,   1,   0  ],
    [0, 0, 1,   1,   0  ],
    [0, 1, 0,   0.5, 1  ],
    [0, 0.5, 0.5, 1, 0  ],
    [0, 1, 0,   0,   1  ],
], dtype=np.float64)

# ── scalability sweep grids ───────────────────────────────────────────────────
S_VALUES = [50, 100, 200, 300, 500, 750, 1000]
G_VALUES = [10, 25, 50, 100, 200, 500]
I_VALUES = [50, 100, 150, 200, 400, 600]
S_BASE, G_BASE, I_BASE = 300, 50, 200


# ── helpers ───────────────────────────────────────────────────────────────────

def trunc_norm_sampling_vector(mu, sigma):
    n = len(mu)
    U = np.random.mtrand._rand.uniform(size=n)
    y = mu + sigma * sc.ndtri(U + sc.ndtr(-mu / sigma) * (1 - U))
    return y


def build_C(I):
    """Tile C_TEMPLATE rows to fill I rows."""
    reps = int(np.ceil(I / len(C_TEMPLATE)))
    return np.tile(C_TEMPLATE, (reps, 1))[:I]


def calc_B(Y, H, N):
    K_loc = H.shape[1]
    g_loc = Y.shape[1]
    N_mat = N * np.eye(len(N))
    X = np.matmul(N_mat, H)
    B = np.zeros((K_loc, g_loc))
    for g_idx in range(g_loc):
        if Y[:, g_idx].sum() > 0:
            B[:, g_idx] = lsq_linear(X, Y[:, g_idx], bounds=(0, np.inf)).x
    return B


def make_simulation(K, S, g, I, seed):
    n_lambda = np.tile(N_LAMBDA, S)
    F        = np.tile(F_BASE,     (K, 1)).astype(float)
    F_eps    = np.tile(F_EPS_BASE, (K, 1)).astype(float)
    C        = build_C(I)
    b_alpha_scale = np.sqrt(B_ALPHA_VAR / B_ALPHA_SHAPE)
    np.random.seed(seed)
    p_y     = trunc_norm_sampling_vector(np.full(g, P_MEAN), P_STD)
    b_alpha = np.random.gamma(B_ALPHA_SHAPE, b_alpha_scale, g)
    return sim.simulation(
        K=K, S=S, g=g, r=PHI_GAMMA[0], q=PHI_GAMMA[1], I=I,
        F=F, D=None, A=None, C=C,
        avarage_clone_in_spot=AVG_CLONE,
        random_seed=seed,
        F_epsilon=F_eps, n=None,
        p_c_binom=None, theta=1, Z=None,
        n_lambda=n_lambda, F_fraction=False,
        pi_2D=PI_2D, Y=None, p_y=p_y,
        b_alpha=b_alpha, b_beta=B_BETA,
        b_alpha_shape=B_ALPHA_SHAPE, b_alpha_scale=b_alpha_scale,
    )


# ── worker ────────────────────────────────────────────────────────────────────

def time_one_experiment(args):
    """
    Builds simulation data in memory, runs short Tumoroscope for init,
    then times TIMING_ITERS ClonalGE Gibbs iterations.
    Returns a dict row for the CSV.
    """
    (label, param_name, param_value,
     K, S, g, I, timing_iters, tmp_dir, seed) = args

    log_path   = os.path.join(tmp_dir, f'{label}.log')
    wall_time  = float('nan')
    gen_time   = float('nan')

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    try:
        with open(log_path, 'w') as logf:
            sys.stdout = logf
            sys.stderr = logf

            n_lambda = np.tile(N_LAMBDA, S)
            F        = np.tile(F_BASE,     (K, 1)).astype(float)
            F_eps    = np.tile(F_EPS_BASE, (K, 1)).astype(float)

            # ── generate simulation data ──────────────────────────────────
            t0 = time.time()
            sample = make_simulation(K=K, S=S, g=g, I=I, seed=seed)
            gen_time = time.time() - t0

            result_base = os.path.join(tmp_dir, label)

            # ── short Tumoroscope run for initialization ──────────────────
            tum_obj = tum.tumoroscope(
                name=result_base + '_tum',
                K=K, S=S, r=None, p=None, I=I,
                avarage_clone_in_spot=AVG_CLONE, F=F, C=sample.C,
                A=sample.A, D=sample.D, F_epsilon=F_eps,
                optimal_rate=OPTIMAL_RATE, n_lambda=n_lambda,
                gamma=GAMMA, pi_2D=PI_2D,
                result_txt=result_base + '_tum_results',
                rp_est_method='my',
            )
            tum_result = tum_obj.gibbs_sampling(
                seed=seed,
                min_iter=TUM_MAX_ITER * 10,  # never early-stops
                max_iter=TUM_MAX_ITER,
                burn_in=50, batch=TUM_BATCH,
                simulated_data=sample, n_sampling=True,
                F_fraction=False, theta_variable=False,
                pi_2D=PI_2D, th=TH,
                every_n_sample=EVERY_N,
                changes_batch=TUM_BATCH,
                var_calculation=int(TUM_MAX_ITER * 0.9),
            )

            # ── ClonalGE initialization from Tumoroscope output ───────────
            B_init = calc_B(sample.Y, tum_result.inferred_H, tum_result.inferred_n)
            inits  = (
                tum_result.inferred_n, tum_result.inferred_H,
                tum_result.inferred_G, tum_result.inferred_pi,
                tum_result.inferred_phi, tum_result.inferred_Z,
                B_init,
            )

            cl_obj = clonalGE(
                name=result_base + '_cl',
                K=K, S=S, g=g, r=None, q=None, I=I,
                avarage_clone_in_spot=AVG_CLONE, F=F, C=sample.C,
                A=sample.A, D=sample.D, F_epsilon=F_eps,
                optimal_rate=OPTIMAL_RATE, n_lambda=n_lambda,
                pi_2D=PI_2D,
                result_txt=result_base + '_cl_results',
                Y=sample.Y, p_y=sample.p_y,
                b_alpha=sample.b_alpha, b_beta=sample.b_beta,
                inits=inits,
            )

            # ── time ClonalGE (min_iter >> max_iter → no early stopping) ──
            t0 = time.time()
            cl_obj.gibbs_sampling(
                seed=seed,
                min_iter=timing_iters * 10,   # never early-stops
                max_iter=timing_iters,
                batch=BATCH,
                simulated_data=None, n_sampling=True,
                F_fraction=False, pi_2D=PI_2D, th=TH,
                every_n_sample=EVERY_N,
                changes_batch=BATCH,
            )
            wall_time = time.time() - t0

    except Exception as exc:
        with open(log_path, 'a') as logf:
            import traceback
            logf.write(f'\nERROR: {exc}\n')
            traceback.print_exc(file=logf)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    sec_per_iter = wall_time / timing_iters if not np.isnan(wall_time) else float('nan')
    status = f'{wall_time:.1f}s ({sec_per_iter:.4f} s/iter)' if not np.isnan(wall_time) else 'ERROR'
    print(f'  [{label:40s}]  {status}', flush=True)

    return {
        'label':        label,
        'param_name':   param_name,
        'param_value':  str(param_value),
        'K':            K,
        'S':            S,
        'g':            g,
        'I':            I,
        'timing_iters': timing_iters,
        'wall_time_s':  round(wall_time, 4) if not np.isnan(wall_time) else 'nan',
        'sec_per_iter': round(sec_per_iter, 6) if not np.isnan(sec_per_iter) else 'nan',
        'gen_time_s':   round(gen_time, 4) if not np.isnan(gen_time) else 'nan',
    }


# ── experiment list ───────────────────────────────────────────────────────────

def build_experiments(timing_iters, tmp_dir):
    exps = []
    seed = 0

    # baseline configs (same model dimensions, different hyperparams)
    for cfg in ['normal', 'low_variance', 'high_coverage']:
        exps.append((
            f'baseline_{cfg}', 'config', cfg,
            K, S_BASE, G_BASE, I_BASE,
            timing_iters, tmp_dir, seed,
        ))
        seed += 1

    # scalability: vary S
    for s_val in S_VALUES:
        exps.append((
            f'scale_S_{s_val:04d}', 'S', s_val,
            K, s_val, G_BASE, I_BASE,
            timing_iters, tmp_dir, seed,
        ))
        seed += 1

    # scalability: vary g
    for g_val in G_VALUES:
        exps.append((
            f'scale_g_{g_val:04d}', 'g', g_val,
            K, S_BASE, g_val, I_BASE,
            timing_iters, tmp_dir, seed,
        ))
        seed += 1

    # scalability: vary I
    for i_val in I_VALUES:
        exps.append((
            f'scale_I_{i_val:04d}', 'I', i_val,
            K, S_BASE, G_BASE, i_val,
            timing_iters, tmp_dir, seed,
        ))
        seed += 1

    return exps


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='ClonalGE scalability runtime analysis')
    ap.add_argument('--workers', type=int, default=None,
                    help='Parallel workers (default: all available CPUs)')
    ap.add_argument('--iters',   type=int, default=TIMING_ITERS,
                    help=f'ClonalGE Gibbs iterations per experiment (default: {TIMING_ITERS})')
    ap.add_argument('--outdir',  default='results/scalability',
                    help='Output directory for CSV and logs (default: results/scalability)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tmp_dir = os.path.join(args.outdir, 'tmp_logs')
    os.makedirs(tmp_dir, exist_ok=True)

    experiments = build_experiments(args.iters, tmp_dir)
    n_workers   = args.workers or multiprocessing.cpu_count()

    print(f'Running {len(experiments)} experiments on {n_workers} parallel workers '
          f'({args.iters} ClonalGE iters each)')
    print(f'Logs  → {tmp_dir}/')
    print(f'CSV   → {args.outdir}/runtime_results.csv\n', flush=True)

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(time_one_experiment, experiments)

    # write CSV
    csv_path   = os.path.join(args.outdir, 'runtime_results.csv')
    fieldnames = [
        'label', 'param_name', 'param_value',
        'K', 'S', 'g', 'I',
        'timing_iters', 'wall_time_s', 'sec_per_iter', 'gen_time_s',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    errors = sum(1 for r in results if r['wall_time_s'] == 'nan')
    print(f'\nDone. {len(results) - errors}/{len(results)} experiments succeeded.')
    print(f'Results → {csv_path}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
