"""
Generate assumption-violating simulation datasets for ClonalGE robustness testing.

Two violation types are applied on top of the standard 'normal' config:
  1. zinb   — Zero-Inflated Negative Binomial: each Y_sg is independently
              zeroed out with probability ZIP_RATE (structural zeros added
              on top of the NB-generated counts).
  2. batch  — Multiplicative section-level batch effects: spots are divided
              into J_BATCH equal groups; each group's expression matrix Y
              is multiplied by a factor drawn from Uniform(BATCH_LO, BATCH_HI)
              and rounded to integers.

Standard ('normal') simulations are also saved for direct comparison.

Usage:
    python generate_simulations_violated.py <run_number> <output_dir>

    e.g.  for i in $(seq 1 20); do
              python generate_simulations_violated.py $i \\
                  /Volumes/LenovoPS8/ClonalGE/test_sim_violated
          done
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import os
import sys
import json
import pickle
import random
import numpy as np
import scipy.special as sc

from clonalge import simulation as sim
from clonalge import constants

# ── violation parameters ──────────────────────────────────────────────────────
ZIP_RATE  = 0.10   # fraction of counts set to structural zeros (ZINB)
J_BATCH   = 5      # number of batch groups
BATCH_LO  = 0.5    # lower bound of batch factor Uniform
BATCH_HI  = 2.0    # upper bound of batch factor Uniform

CONFIGS = ['normal']   # only 'normal' config for violated experiments
CONFIG_DIR = 'configs'


def trunc_norm_vector(mu, sigma):
    n = len(mu)
    U = np.random.mtrand._rand.uniform(size=n)
    return mu + sigma * sc.ndtri(U + sc.ndtr(-mu / sigma) * (1 - U))


def apply_zinb(Y, zip_rate, rng):
    """Add structural zeros: zero out each entry independently with prob zip_rate."""
    mask = rng.binomial(1, 1 - zip_rate, size=Y.shape)
    return (Y * mask).astype(int)


def apply_batch(Y, n_groups, lo, hi, rng):
    """Apply multiplicative batch factors, one per group of spots."""
    S = Y.shape[0]
    group_size = S // n_groups
    Y_out = Y.astype(float).copy()
    factors = rng.uniform(lo, hi, size=n_groups)
    for j in range(n_groups):
        start = j * group_size
        end = (j + 1) * group_size if j < n_groups - 1 else S
        Y_out[start:end, :] *= factors[j]
    return np.round(Y_out).astype(int), factors


def load_config(path):
    with open(path) as f:
        return json.load(f)


def build_C(data):
    C_temp = data['C_variation']['C']
    repeat_temp = data['C_variation']['repeat']
    if C_temp is None:
        return None
    parts = [np.tile(C_temp[ct], (repeat_temp[ct], 1))
             for ct in range(len(C_temp))]
    return np.concatenate(parts)


def generate_base_sim(data, rng_seed):
    """Generate a standard simulation object from a config dict."""
    K  = data['structure']['K']
    S  = data['structure']['S']
    I  = data['structure']['I']
    g  = data['structure']['g']
    theta = data['structure']['theta']
    acs = data['Z_variation']['avarage_clone_in_spot']
    C   = build_C(data)
    F_epsilon = np.tile(data['Gamma']['F_epsilon'], (K, 1))
    F         = np.tile(data['Gamma']['F'],         (K, 1))
    n_lambda  = np.tile(data['n_variation']['n_lambda'], S)
    p_y   = trunc_norm_vector(
        np.array([data['Y_variation']['p_mean']] * g),
        data['Y_variation']['p_std'])
    b_alpha_shape = data['Y_variation']['b_alpha_shape']
    b_alpha_var   = data['Y_variation']['b_alpha_var']
    b_alpha_scale = np.sqrt(b_alpha_var / b_alpha_shape)
    b_alpha = np.random.gamma(b_alpha_shape, b_alpha_scale, g)
    b_beta  = data['Y_variation']['b_beta']
    phi_gamma = np.array(data['Gamma']['phi_gamma'])

    mean_read = data['Gamma'].get('mean_read', None)
    lb = np.floor(mean_read * 0.9) if mean_read else None
    ub = np.ceil(mean_read * 1.1)  if mean_read else None

    while True:
        s = sim.simulation(
            K=K, S=S, g=g, r=phi_gamma[0], q=phi_gamma[1], I=I,
            F=F, D=None, A=None, C=C, avarage_clone_in_spot=acs,
            random_seed=random.randint(1, 10000), F_epsilon=F_epsilon,
            n=None, p_c_binom=data['C_variation']['p_c_binom'],
            theta=theta, Z=None, n_lambda=n_lambda,
            F_fraction=data['Gamma']['F_fraction'],
            pi_2D=True, Y=None, p_y=p_y,
            b_alpha=b_alpha, b_beta=b_beta,
            b_alpha_shape=b_alpha_shape, b_alpha_scale=b_alpha_scale)
        if mean_read is None:
            break
        mean_d = np.mean(np.sum(s.D, axis=0))
        if lb <= mean_d <= ub:
            break
    return s


def main():
    if len(sys.argv) < 3:
        print('Usage: generate_simulations_violated.py <run_number> <output_dir>')
        sys.exit(1)

    run_number = str(sys.argv[1])
    out_root   = sys.argv[2]

    rng = np.random.default_rng(int(run_number) * 42)

    for config_name in CONFIGS:
        config_path = os.path.join(CONFIG_DIR, config_name + '.json')
        data = load_config(config_path)

        out_dir = os.path.join(out_root, run_number)
        os.makedirs(out_dir, exist_ok=True)

        print(f'Run {run_number} / {config_name} — generating base simulation ...')
        base = generate_base_sim(data, rng_seed=int(run_number))

        # ── 1. Standard (no violation) ─────────────────────────────────────
        path_normal = os.path.join(out_dir, f'sample_{config_name}')
        pickle.dump(base, open(path_normal, 'wb'))
        print(f'  saved: {path_normal}')

        # ── 2. ZINB violation ─────────────────────────────────────────────
        import copy
        zinb = copy.deepcopy(base)
        zinb.Y_original = zinb.Y.copy()   # keep ground-truth Y for reference
        zinb.Y = apply_zinb(zinb.Y, ZIP_RATE, rng)
        zinb.violation_type = 'zinb'
        zinb.zip_rate = ZIP_RATE
        path_zinb = os.path.join(out_dir, f'sample_{config_name}_zinb')
        pickle.dump(zinb, open(path_zinb, 'wb'))
        print(f'  saved: {path_zinb}  '
              f'(zero fraction: {(zinb.Y == 0).mean():.2%})')

        # ── 3. Batch-effect violation ─────────────────────────────────────
        batch = copy.deepcopy(base)
        batch.Y_original = batch.Y.copy()
        batch.Y, batch_factors = apply_batch(batch.Y, J_BATCH, BATCH_LO, BATCH_HI, rng)
        batch.violation_type = 'batch'
        batch.batch_factors  = batch_factors
        path_batch = os.path.join(out_dir, f'sample_{config_name}_batch')
        pickle.dump(batch, open(path_batch, 'wb'))
        print(f'  saved: {path_batch}  '
              f'(batch factors: {np.round(batch_factors, 2)})')


if __name__ == '__main__':
    main()
