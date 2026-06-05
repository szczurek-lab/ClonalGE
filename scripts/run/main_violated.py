"""
Run ClonalGE on one replicate of a (possibly assumption-violating) simulation.

Usage:
    python main_violated.py <run_num> <results_dir> <config_dir> <noise> <variant>

variant: normal | zinb | batch
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, json, pickle, random, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import lsq_linear
from clonalge import constants
from clonalge.model import clonalGE
from clonalge import tumoroscope as tum

# ── settings ──────────────────────────────────────────────────────────────────
constants.CHAINS = int(os.environ.get('CLONALGE_CHAINS', 1))
constants.CORES  = int(os.environ.get('CLONALGE_CORES',  min(constants.CHAINS, 10)))
every_n_sample = 5
changes_batch  = 500
optimal_rate   = 0.4
pi_2D = True
th    = 0.8

number     = str(sys.argv[1])
result_dir = sys.argv[2]
config_dir = sys.argv[3]
noise      = str(sys.argv[4])
variant    = sys.argv[5] if len(sys.argv) > 5 else 'normal'

# sim_dir is always the base violated sim directory, independent of variant
import re as _re
sim_dir    = _re.sub(r'Results_violated_\w+', 'test_sim_violated',
                     result_dir.rstrip('/'))
result_dir = result_dir + number
os.makedirs(result_dir, exist_ok=True)

result_txt = result_dir + '/results_'


def calc_B(Y, H, N):
    K, G = H.shape[1], Y.shape[1]
    X = np.diag(N) @ H
    B = np.zeros((K, G))
    for g in range(G):
        if Y[:, g].sum() > 0:
            B[:, g] = lsq_linear(X, Y[:, g], bounds=(0, np.inf)).x
    return B


def rmae(inferred, true):
    m = np.mean(np.abs(true))
    return float(np.mean(np.abs(inferred - true)) / m) if m > 0 else float('nan')


import glob, re
for cfg_path in glob.glob(config_dir + '/*.json'):
    file_name = os.path.splitext(os.path.basename(cfg_path))[0]

    with open(cfg_path) as f:
        data = json.load(f)

    K = data['structure']['K']
    S = data['structure']['S']
    I = data['structure']['I']
    g = data['structure']['g']
    F_epsilon = np.tile(data['Gamma']['F_epsilon'], (K, 1))
    F         = np.tile(data['Gamma']['F'],         (K, 1))

    max_iter = int(data['sampling']['max_iter'])
    min_iter = int(data['sampling']['min_iter'])
    burn_in  = int(data['sampling']['burn_in'])
    batch    = int(data['sampling']['batch'])
    var_calc = int(data['sampling']['min_iter'] * 0.9)

    suffix = '' if variant == 'normal' else f'_{variant}'
    sim_path = os.path.join(sim_dir, number, f'sample_{file_name}{suffix}')

    if not os.path.exists(sim_path):
        print(f'Missing {sim_path}, skipping')
        continue

    sample_1 = pickle.load(open(sim_path, 'rb'))
    n_lambda = sample_1.n

    # ── Tumoroscope ──────────────────────────────────────────────────────────
    tum_objs = []
    for cc in range(constants.CHAINS):
        tum_objs.append(tum.tumoroscope(
            name=f'{result_dir}/{file_name}_chain_{cc}',
            K=K, S=S, r=None, p=None, I=I,
            avarage_clone_in_spot=sample_1.avarage_clone_in_spot,
            F=F, C=sample_1.C, A=sample_1.A, D=sample_1.D,
            F_epsilon=F_epsilon, optimal_rate=optimal_rate,
            n_lambda=n_lambda, gamma=0.95, pi_2D=pi_2D,
            result_txt=result_txt + file_name + '.txt', rp_est_method='my'))

    tum_all = []
    for t in tum_objs:
        t.gibbs_sampling(
            seed=random.randint(1, 1000), min_iter=int(min_iter / 1.5),
            max_iter=int(max_iter / 2), burn_in=burn_in, batch=batch,
            simulated_data=sample_1, n_sampling=True, F_fraction=False,
            theta_variable=False, pi_2D=pi_2D, th=th,
            every_n_sample=every_n_sample, changes_batch=changes_batch,
            var_calculation=var_calc)
        tum_all.append(t)

    # ── ClonalGE ─────────────────────────────────────────────────────────────
    cl_objs = []
    for cc in range(constants.CHAINS):
        t1 = tum_all[cc]
        B_init = calc_B(sample_1.Y, t1.inferred_H, t1.inferred_n)
        inits  = (t1.inferred_n, t1.inferred_H, t1.inferred_G,
                  t1.inferred_pi, t1.inferred_phi, t1.inferred_Z, B_init)
        cl_objs.append(clonalGE(
            name=f'{result_dir}/{file_name}_chain_{cc}',
            K=K, S=S, g=g, r=None, q=None, I=I,
            avarage_clone_in_spot=sample_1.avarage_clone_in_spot,
            F=F, C=sample_1.C, A=sample_1.A, D=sample_1.D,
            F_epsilon=F_epsilon, optimal_rate=optimal_rate,
            n_lambda=n_lambda, pi_2D=pi_2D,
            result_txt=result_txt + file_name + '.txt',
            Y=sample_1.Y, p_y=sample_1.p_y,
            b_alpha=sample_1.b_alpha, b_beta=sample_1.b_beta,
            inits=inits))

    cl_all = []
    for c in cl_objs:
        c.gibbs_sampling(
            seed=random.randint(1, 1000), min_iter=min_iter,
            max_iter=max_iter, batch=batch, simulated_data=sample_1,
            n_sampling=True, F_fraction=False, pi_2D=pi_2D, th=th,
            every_n_sample=every_n_sample, changes_batch=changes_batch)
        cl_all.append(c)

    logliks   = [c.last_loglik for c in cl_all]
    chain_best = cl_all[int(np.argmax(logliks))]

    h_rmae = rmae(chain_best.inferred_H, sample_1.H)
    b_rmae = rmae(chain_best.inferred_B, sample_1.B)
    print(f'[{variant}] run={number} config={file_name}  '
          f'H_rMAE={h_rmae:.4f}  B_rMAE={b_rmae:.4f}')

    with open(result_txt + file_name + '.txt', 'a') as f:
        f.write(f'\nVariant: {variant}\n')
        f.write(f'H_rMAE (violated): {h_rmae}\n')
        f.write(f'B_rMAE (violated): {b_rmae}\n')
        f.write(f'Mean of true H: {np.mean(sample_1.H)}\n')
        f.write(f'Mean of true B: {np.mean(sample_1.B)}\n')

    chain_best.H = 0
    pickle.dump(chain_best,
                open(f'{result_dir}/best_oo{constants.CHAINS}_{file_name}', 'wb'))
