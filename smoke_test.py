"""Fast self-contained smoke test of the ClonalGE pipeline.

Exercises the real code path: simulate tiny data -> Tumoroscope init -> ClonalGE
Gibbs sampling -> inferred outputs. Tiny dimensions + few iterations so it runs
in seconds. Used to verify the package still works after files are moved.

Run:  python smoke_test.py
Exit code 0 = pass.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import random

import simulation as sim
import tumoroscope as tum
from clonalGE import clonalGE


def main():
    np.random.seed(0)
    random.seed(0)

    K, S, g, I = 4, 40, 10, 40
    C = np.random.binomial(1, 0.5, size=(I, K)).astype(float)
    C[:, 0] = 0.0                      # normal clone, no mutations
    F = np.tile([30, 1], (K, 1))
    F_epsilon = np.tile([2, 1], (K, 1))
    n_lambda = np.tile(45, S)
    b_alpha = np.random.gamma(1.0, 0.5, g)

    s = sim.simulation(
        K=K, S=S, g=g, r=0.005, q=1, I=I, F=F, D=None, A=None, C=C,
        avarage_clone_in_spot=2.5, random_seed=1, F_epsilon=F_epsilon, n=None,
        p_c_binom=None, theta=1, Z=None, n_lambda=n_lambda, F_fraction=False,
        pi_2D=True, Y=None, p_y=np.full(g, 0.05), b_alpha=b_alpha, b_beta=3.5,
        b_alpha_shape=1, b_alpha_scale=0.5)
    assert s.D.shape == (I, S) and s.Y.shape == (S, g)

    t = tum.tumoroscope(
        name='smoke_tum', K=K, S=S, r=None, p=None, I=I,
        avarage_clone_in_spot=s.avarage_clone_in_spot, F=F, C=s.C, A=s.A, D=s.D,
        F_epsilon=F_epsilon, optimal_rate=0.4, n_lambda=n_lambda, gamma=0.95,
        pi_2D=True, result_txt='smoke_tum.txt', rp_est_method='my')
    t.gibbs_sampling(seed=1, min_iter=30, max_iter=60, burn_in=5, batch=20,
                     simulated_data=s, n_sampling=True, F_fraction=False,
                     theta_variable=False, pi_2D=True, th=0.8, every_n_sample=5,
                     changes_batch=20, var_calculation=10)
    assert t.inferred_H.shape == (S, K)

    cg = clonalGE(
        name='smoke_cg', K=K, S=S, g=g, r=None, q=None, I=I,
        avarage_clone_in_spot=s.avarage_clone_in_spot, F=F, C=s.C, A=s.A, D=s.D,
        F_epsilon=F_epsilon, optimal_rate=0.4, n_lambda=n_lambda, pi_2D=True,
        result_txt='smoke_cg.txt', Y=s.Y, p_y=s.p_y, b_alpha=s.b_alpha,
        b_beta=s.b_beta,
        inits=(t.inferred_n, t.inferred_H, t.inferred_G, t.inferred_pi,
               t.inferred_phi, t.inferred_Z, np.ones((K, g))))
    cg.gibbs_sampling(seed=1, min_iter=30, max_iter=60, batch=20,
                      simulated_data=s, n_sampling=True, F_fraction=False,
                      pi_2D=True, th=0.8, every_n_sample=5, changes_batch=20)

    assert cg.inferred_B.shape == (K, g)
    assert cg.inferred_H.shape == (S, K)
    assert np.all(np.isfinite(cg.inferred_B))
    print("SMOKE OK: B", cg.inferred_B.shape, "H", cg.inferred_H.shape,
          "loglik", round(float(cg.last_loglik), 2))


if __name__ == '__main__':
    main()
