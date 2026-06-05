"""
Compute Tumoroscope + NB results for Figure 2.

For each run/condition:
  1. Load Tumoroscope inferred H and n from results txt
  2. Load simulation pickle → true Y, B, p_y, b_alpha, b_beta
  3. Estimate B by maximising NB log-likelihood + Gamma prior (per gene, L-BFGS-B)
       r_sg = n_s * (H @ B)_sg * (p_y / (1-p_y))_g
       Y_sg ~ NB(r_sg, p_y_g)
       B_kg ~ Gamma(b_alpha_g, b_beta)   [same prior as ClonalGE]
  4. Append B_SEE (Tumoroscope+NB) to results txt

Run ONCE locally (or on cluster) after Tumoroscope has finished.

Usage:
    python compute_tumoroscope_nb.py \\
        --tum_dir  /Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope \\
        --sim_dir  /Volumes/LenovoPS8/ClonalGE/test_sim \\
        --num_runs 20
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import os, sys, re, argparse
import numpy as np
import scipy.stats
from scipy.optimize import minimize, lsq_linear

ap = argparse.ArgumentParser()
ap.add_argument('--tum_dir',  required=True)
ap.add_argument('--sim_dir',  required=True)
ap.add_argument('--num_runs', type=int, default=20)
args = ap.parse_args()

CONDITIONS = ['normal', 'low_variance', 'high_coverage']

clonalge_dir = os.path.dirname(os.path.abspath(__file__))
if clonalge_dir not in sys.path:
    sys.path.insert(0, clonalge_dir)
import pickle


# ── helpers ───────────────────────────────────────────────────────────────────

def calc_B_nb(Y, H, n, p_y, b_alpha, b_beta):
    """
    Estimate B (K x G) by maximising NB log-likelihood + Gamma prior.
    Uses LR solution as initial guess per gene.
    """
    K = H.shape[1]
    G = Y.shape[1]

    p_y       = np.clip(p_y, 1e-6, 1 - 1e-6)
    p_y_ratio = p_y / (1 - p_y)           # (G,)
    n_arr     = np.asarray(n, dtype=float) # (S,)

    # LR initial guess
    N_diag = n_arr * np.eye(len(n_arr))
    X      = N_diag @ H                   # (S, K)

    B = np.zeros((K, G))

    for g in range(G):
        y_g = Y[:, g]
        if y_g.sum() == 0:
            continue

        p_g       = float(p_y[g])
        p_ratio_g = float(p_y_ratio[g])
        alpha_g   = float(b_alpha[g]) if hasattr(b_alpha, '__len__') else float(b_alpha)
        beta_g    = float(b_beta)

        # initial guess from LR
        b0 = lsq_linear(X, y_g, bounds=(1e-6, np.inf)).x

        def neg_log_posterior(b_kg):
            r_sg = n_arr * (H @ b_kg) * p_ratio_g
            r_sg = np.maximum(r_sg, 1e-10)
            ll   = np.sum(scipy.stats.nbinom.logpmf(y_g, r_sg, p_g))
            lp   = np.sum(scipy.stats.gamma.logpdf(b_kg,
                                                    a=alpha_g, scale=beta_g))
            return -(ll + lp)

        bounds = [(1e-10, None)] * K
        res = minimize(neg_log_posterior, b0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-9})
        B[:, g] = res.x

    return B


def parse_matrix(txt, header):
    pattern = re.escape(header) + r'\n((?:[\d.eE+\-nan ]+\n?)+)'
    m = re.search(pattern, txt)
    if m is None:
        return None
    lines = m.group(1).strip().split('\n')
    return np.array([[float(v) for v in l.split()] for l in lines if l.strip()])


def already_done(txt):
    return 'Standard Error of the Estimate B (Tumoroscope+NB)' in txt


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

        inferred_H = parse_matrix(txt, 'Inferred H:')
        inferred_n = parse_matrix(txt, 'Inferred n:')
        if inferred_H is None or inferred_n is None:
            print(f'  ERROR: could not parse Inferred H or n')
            continue
        inferred_n = inferred_n.flatten()

        try:
            sim = pickle.load(open(sim_path, 'rb'))
        except Exception as e:
            print(f'  ERROR loading sim: {e}')
            continue

        Y      = sim.Y
        B_true = sim.B
        p_y    = sim.p_y
        b_alpha = sim.b_alpha
        b_beta  = sim.b_beta

        S, G = Y.shape
        K    = inferred_H.shape[1]
        print(f'  Computing NB B  (S={S}, G={G}, K={K}) ...')

        B_nb  = calc_B_nb(Y, inferred_H, inferred_n, p_y, b_alpha, b_beta)
        B_SEE = float(np.mean(np.abs(B_true - B_nb)))
        print(f'  B_SEE (NB) = {B_SEE:.5f}')

        with open(result_txt, 'a') as f:
            f.write(f'\nStandard Error of the Estimate B (Tumoroscope+NB):\n{B_SEE}\n')

        print(f'  Appended to {result_txt}')

print('\nDone.')
