#!/usr/bin/env python3

import pickle
import numpy as np
import json
import random
import os
from clonalGE import clonalGE
import tumoroscope as tum
from scipy.optimize import lsq_linear

def calc_B(Y, H, N):
    """Calculate B matrix using least squares (clone-specific gene expression)"""
    K = H.shape[1]
    G = Y.shape[1]
    N = N * np.eye(len(N))
    X = np.matmul(N, H)
    B = np.zeros((K, G))
    for g in range(G):
        if sum(Y[:, g]) > 0:
            B[:, g] = lsq_linear(X, Y[:, g], bounds=(0, np.inf)).x
    return B

def run_toy_clonalge():
    # Load toy configuration
    with open('configs/toy.json') as f:
        data = json.load(f)
    
    # Load simulated data
    sample = pickle.load(open('test_sim/1/sample_toy', 'rb'))
    
    print("Loaded toy simulation data:")
    print(f"K (clones): {sample.K}, S (spots): {sample.S}, I (mutations): {sample.I}, g (genes): {sample.g}")
    
    # Extract parameters from config
    K = data['structure']['K']
    S = data['structure']['S']
    I = data['structure']['I']
    g = data['structure']['g']
    
    F_epsilon = np.tile(data['Gamma']['F_epsilon'], (K, 1))
    F = np.tile(data['Gamma']['F'], (K, 1))
    
    max_iter = data['sampling']['max_iter']
    min_iter = data['sampling']['min_iter']
    batch = data['sampling']['batch']
    
    optimal_rate = 0.4
    pi_2D = True
    th = 0.8
    every_n_sample = 5
    changes_batch = 100
    
    # Create output directory
    os.makedirs('toy_results', exist_ok=True)
    
    # Use true values from simulation as lambda (usually unknown)
    n_lambda = sample.n
    
    print("\n=== Step 1: Running Tumoroscope (spatial clone inference) ===")
    
    # Initialize Tumoroscope
    tum_obj = tum.tumoroscope(
        name='toy_results/tumoroscope_toy',
        K=sample.K, S=sample.S, r=None, p=None,
        I=sample.I, avarage_clone_in_spot=sample.avarage_clone_in_spot, 
        F=F, C=sample.C, A=sample.A, D=sample.D, F_epsilon=F_epsilon,
        optimal_rate=optimal_rate, n_lambda=n_lambda, gamma=0.95, pi_2D=pi_2D,
        result_txt='toy_results/tumoroscope_results.txt', rp_est_method='my'
    )
    
    # Run Tumoroscope
    print("Running Tumoroscope inference...")
    tum_result = tum_obj.gibbs_sampling(
        seed=42, min_iter=int(min_iter/3), max_iter=int(max_iter/3), 
        burn_in=int(data['sampling']['burn_in']/3), batch=int(batch/2),
        simulated_data=sample, n_sampling=True, F_fraction=False,
        theta_variable=False, pi_2D=pi_2D, th=th, 
        every_n_sample=every_n_sample, changes_batch=changes_batch,
        var_calculation=int(min_iter * 0.1)
    )
    
    print(f"Tumoroscope completed. Final log-likelihood: {tum_result.last_loglik:.2f}")
    
    print("\n=== Step 2: Running ClonalGE (adding gene expression) ===")
    
    # Calculate initial B matrix from Tumoroscope results
    B_init = calc_B(sample.Y, tum_result.inferred_H, tum_result.inferred_n)
    
    # Initialize ClonalGE with Tumoroscope results
    inits = (
        tum_result.inferred_n, tum_result.inferred_H, tum_result.inferred_G,
        tum_result.inferred_pi, tum_result.inferred_phi, tum_result.inferred_Z, B_init
    )
    
    clonal_obj = clonalGE(
        name='toy_results/clonalge_toy',
        K=sample.K, S=sample.S, g=sample.g, r=None, q=None,
        I=sample.I, avarage_clone_in_spot=sample.avarage_clone_in_spot,
        F=F, C=sample.C, A=sample.A, D=sample.D, F_epsilon=F_epsilon,
        optimal_rate=optimal_rate, n_lambda=n_lambda, pi_2D=pi_2D,
        result_txt='toy_results/clonalge_results.txt', Y=sample.Y, 
        p_y=sample.p_y, b_alpha=sample.b_alpha, b_beta=sample.b_beta, 
        inits=inits
    )
    
    # Run ClonalGE
    print("Running ClonalGE inference...")
    clonal_result = clonal_obj.gibbs_sampling(
        seed=42, min_iter=min_iter, max_iter=max_iter,
        batch=batch, simulated_data=sample, n_sampling=True, 
        F_fraction=False, pi_2D=pi_2D, th=th, 
        every_n_sample=every_n_sample, changes_batch=changes_batch
    )
    
    print(f"ClonalGE completed. Final log-likelihood: {clonal_result.last_loglik:.2f}")
    
    print("\n=== Results Comparison ===")
    
    # Compare true vs inferred values
    print("H matrix (clone proportions per spot) - True vs Inferred:")
    print(f"  H MAE: {np.mean(np.abs(sample.H - clonal_result.inferred_H)):.4f}")
    
    print("B matrix (clone-specific gene expression) - True vs Inferred:")
    print(f"  B MAE: {np.mean(np.abs(sample.B - clonal_result.inferred_B)):.4f}")
    
    print("phi matrix (mutation frequencies) - True vs Inferred:")
    print(f"  phi MAE: {np.mean(np.abs(sample.phi - clonal_result.inferred_phi)):.4f}")
    
    print("n vector (cell counts) - True vs Inferred:")
    print(f"  n MAE: {np.mean(np.abs(sample.n - clonal_result.inferred_n)):.4f}")
    
    # Save results
    pickle.dump(clonal_result, open('toy_results/clonalge_result.pkl', 'wb'))
    
    print("\nResults saved to toy_results/")
    print("- clonalge_results.txt: Detailed text results")
    print("- clonalge_result.pkl: Pickled result object")
    
    return clonal_result, sample

if __name__ == "__main__":
    result, sample = run_toy_clonalge()



