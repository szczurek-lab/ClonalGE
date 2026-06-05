"""
Compute Tumoroscope + LR results for Figure 2.

For each run/condition, this script:
  1. Loads the Tumoroscope results txt → parses inferred H and n
  2. Loads the ClonalGE simulation pickle from SIM_DIR → gets true Y, B, phi, n
  3. Computes B via non-negative least squares (same calc_B as ClonalGE uses)
  4. Appends to the results txt:
       Mean of true H / phi / n / B
       Standard Error of the Estimate B (Tumoroscope+LR)

Run ONCE on the cluster after Tumoroscope has already finished.

Usage:
    python compute_tumoroscope_lr.py \
        --tum_dir  /path/to/Results_simulated_tumoroscope \
        --sim_dir  /path/to/test_sim \
        --num_runs 20
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import os, sys, re, argparse
import numpy as np
from scipy.optimize import lsq_linear

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--tum_dir',  required=True,
                help='Base dir of Tumoroscope results (run number appended, e.g. /path/Results_simulated_tumoroscope)')
ap.add_argument('--sim_dir',  required=True,
                help='test_sim directory: {sim_dir}/{run}/sample_{cond}')
ap.add_argument('--num_runs', type=int, default=20)
args = ap.parse_args()

CONDITIONS = ['normal', 'low_variance', 'high_coverage']

# ── add ClonalGE to path so simulation class unpickles correctly ──────────────
clonalge_dir = os.path.dirname(os.path.abspath(__file__))
if clonalge_dir not in sys.path:
    sys.path.insert(0, clonalge_dir)
import pickle


def calc_B(Y, H, N):
    """Non-negative least squares B estimate — same as main_diff_config.py."""
    K = H.shape[1]
    G = Y.shape[1]
    N_diag = N * np.eye(len(N))
    X = np.matmul(N_diag, H)          # diag(N) @ H,  shape (S, K)
    B = np.zeros((K, G))
    for g in range(G):
        if Y[:, g].sum() > 0:
            B[:, g] = lsq_linear(X, Y[:, g], bounds=(0, np.inf)).x
    return B


def parse_matrix(txt, header):
    """Extract a float matrix from the block following 'header\\n' in txt."""
    pattern = re.escape(header) + r'\n((?:[\d.eE+\-nan ]+\n?)+)'
    m = re.search(pattern, txt)
    if m is None:
        return None
    lines = m.group(1).strip().split('\n')
    return np.array([[float(v) for v in l.split()] for l in lines if l.strip()])


def already_done(txt):
    return 'Standard Error of the Estimate B (Tumoroscope+LR)' in txt


# ── main loop ─────────────────────────────────────────────────────────────────
for run in range(1, args.num_runs + 1):
    sim_base = os.path.join(args.sim_dir, str(run))
    tum_base = args.tum_dir.rstrip('/') + str(run)

    for cond in CONDITIONS:
        result_txt = os.path.join(tum_base, f'results_{cond}.txt')
        sim_path   = os.path.join(sim_base, f'sample_{cond}')

        print(f'\nRun {run} / {cond}')

        if not os.path.exists(result_txt):
            print(f'  SKIP: {result_txt} not found')
            continue
        if not os.path.exists(sim_path):
            print(f'  SKIP: {sim_path} not found')
            continue

        txt = open(result_txt).read()
        if already_done(txt):
            print(f'  already done — skipping')
            continue

        # ── parse inferred H and n from results txt ───────────────────────────
        inferred_H = parse_matrix(txt, 'Inferred H:')
        inferred_n = parse_matrix(txt, 'Inferred n:')
        if inferred_H is None or inferred_n is None:
            print(f'  ERROR: could not parse Inferred H or n from {result_txt}')
            continue
        inferred_n = inferred_n.flatten()

        # ── load simulation object ────────────────────────────────────────────
        try:
            sim = pickle.load(open(sim_path, 'rb'))
        except Exception as e:
            print(f'  ERROR loading sim: {e}')
            continue

        Y      = sim.Y        # (S, G)
        B_true = sim.B        # (K, G)
        n_true = sim.n        # (S,)
        phi_true = sim.phi    # (I, K)
        H_true   = sim.H      # (S, K)

        # ── compute Tumoroscope + LR  B estimate ─────────────────────────────
        print(f'  Computing LR B  (S={Y.shape[0]}, G={Y.shape[1]}, K={inferred_H.shape[1]}) ...')
        B_lr  = calc_B(Y, inferred_H, inferred_n)
        B_SEE = float(np.mean(np.abs(B_true - B_lr)))
        print(f'  B_SEE (LR) = {B_SEE:.5f}')

        # ── mean of true values (for rMAE denominator) ────────────────────────
        mean_H   = float(np.mean(H_true))     # always 1/K for simplex
        mean_phi = float(np.mean(phi_true))
        mean_n   = float(np.mean(n_true))
        mean_B   = float(np.mean(B_true))

        # ── append to results txt ─────────────────────────────────────────────
        with open(result_txt, 'a') as f:
            f.write(f'\nMean of true H:\n{mean_H}\n')
            f.write(f'Mean of true phi:\n{mean_phi}\n')
            f.write(f'Mean of true n:\n{mean_n}\n')
            f.write(f'Mean of true B:\n{mean_B}\n')
            f.write(f'Standard Error of the Estimate B (Tumoroscope+LR):\n{B_SEE}\n')

        print(f'  Appended to {result_txt}')

print('\nDone.')
