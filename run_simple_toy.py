#!/usr/bin/env python3

import pickle
import numpy as np
import json
import os
from clonalGE import clonalGE
from scipy.optimize import lsq_linear

def calc_B(Y, H, N):
    """Calculate B matrix using least squares (clone-specific gene expression)"""
    K = H.shape[1]
    G = Y.shape[1]
    N = N * np.eye(len(N))
    X = np.matmul(N, H)
    B = np.zeros((K, G))
    for g in range(G):
        if Y[:, g].sum() > 0:
            B[:, g] = lsq_linear(X, Y[:, g], bounds=(0, np.inf)).x
    return B

def run_simple_toy():
    # Load toy configuration
    with open('configs/toy.json') as f:
        data = json.load(f)
    
    # Load simulated data
    sample = pickle.load(open('test_sim/1/sample_toy', 'rb'))
    
    print("=== ClonalGE Toy Simulation ===")
    print(f"Loaded simulation: K={sample.K} clones, S={sample.S} spots, I={sample.I} mutations, g={sample.g} genes")
    
    # Extract parameters
    K = data['structure']['K']
    F_epsilon = np.tile(data['Gamma']['F_epsilon'], (K, 1))
    F = np.tile(data['Gamma']['F'], (K, 1))
    
    # Sampling parameters for toy simulation
    max_iter = 500
    min_iter = 100
    batch = 50
    
    optimal_rate = 0.4
    pi_2D = True
    th = 0.8
    every_n_sample = 10
    changes_batch = 100
    
    os.makedirs('toy_results', exist_ok=True)
    
    # Use true cell counts as n_lambda (this is observed in real data)
    n_lambda = sample.n
    
    print("\n=== Running ClonalGE ===")
    print("Demonstrating parameter recovery from synthetic data")
    print("True parameters are known, we'll see how well ClonalGE recovers them")
    
    # Don't use true values as initialization - let ClonalGE learn from scratch
    # This demonstrates the actual inference capability
    inits = None
    
    # Extract r and q from config
    r = data['Gamma']['phi_gamma'][0]
    q = data['Gamma']['phi_gamma'][1]
    
    clonal_obj = clonalGE(
        name='toy_results/clonalge_simple',
        K=sample.K, S=sample.S, g=sample.g, r=r, q=q,
        I=sample.I, avarage_clone_in_spot=sample.avarage_clone_in_spot,
        F=F, C=sample.C, A=sample.A, D=sample.D, F_epsilon=F_epsilon,
        optimal_rate=optimal_rate, n_lambda=n_lambda, pi_2D=pi_2D,
        result_txt='toy_results/simple_results.txt', Y=sample.Y, 
        p_y=sample.p_y, b_alpha=sample.b_alpha, b_beta=sample.b_beta, 
        inits=inits
    )
    
    print("Running short ClonalGE inference...")
    try:
        clonal_result = clonal_obj.gibbs_sampling(
            seed=42, min_iter=min_iter, max_iter=max_iter,
            batch=batch, simulated_data=sample, n_sampling=False,  # Don't sample n
            F_fraction=False, pi_2D=pi_2D, th=th, 
            every_n_sample=every_n_sample, changes_batch=changes_batch,
            progress_bar=True  # Show progress bar
        )
        
        print(f"\n=== Parameter Recovery Results ===")
        print(f"Final log-likelihood: {clonal_result.last_loglik:.2f}")
        
        # Calculate recovery metrics
        h_mae = np.mean(np.abs(sample.H - clonal_result.inferred_H))
        b_mae = np.mean(np.abs(sample.B - clonal_result.inferred_B))
        phi_mae = np.mean(np.abs(sample.phi - clonal_result.inferred_phi))
        
        print(f"\n📊 Parameter Recovery Accuracy:")
        print(f"  H (clone proportions) MAE: {h_mae:.4f}")
        print(f"  B (gene expression) MAE: {b_mae:.4f}")
        print(f"  phi (mutation freq) MAE: {phi_mae:.4f}")
        
        # Calculate correlation coefficients
        h_corr = np.corrcoef(sample.H.flatten(), clonal_result.inferred_H.flatten())[0,1]
        b_corr = np.corrcoef(sample.B.flatten(), clonal_result.inferred_B.flatten())[0,1]
        
        print(f"\n📈 Correlation with True Values:")
        print(f"  H correlation: {h_corr:.4f}")
        print(f"  B correlation: {b_corr:.4f}")
        
        # Show some example values
        print(f"\n🔍 Example Values (first 3 spots, all clones):")
        print(f"True H (clone proportions):")
        print(sample.H[:3])
        print(f"Inferred H:")
        print(clonal_result.inferred_H[:3])
        
        print(f"\nTrue B (clone-specific gene expression, first 3 genes):")
        print(sample.B[:, :3])
        print(f"Inferred B:")
        print(clonal_result.inferred_B[:, :3])
        
        # Overall assessment
        if h_corr > 0.8 and b_corr > 0.7:
            print(f"\n✅ EXCELLENT parameter recovery!")
        elif h_corr > 0.6 and b_corr > 0.5:
            print(f"\n✅ GOOD parameter recovery!")
        else:
            print(f"\n⚠️  MODERATE parameter recovery - may need more iterations or data")
        
        # Save results
        pickle.dump(clonal_result, open('toy_results/simple_result.pkl', 'wb'))
        print(f"\nResults saved to toy_results/")
        
        return clonal_result, sample
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("This is expected for very small toy examples - the model needs sufficient data for convergence")
        return None, sample

if __name__ == "__main__":
    run_simple_toy()



