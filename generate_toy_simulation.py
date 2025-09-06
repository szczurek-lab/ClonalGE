#!/usr/bin/env python3

import simulation as sim
import pickle
import numpy as np
import json
import random
import os

def generate_toy_simulation():
    # Load toy configuration
    with open('configs/toy.json') as f:
        data = json.load(f)
    
    # Extract parameters from config
    K = data['structure']['K']
    S = data['structure']['S'] 
    I = data['structure']['I']
    g = data['structure']['g']
    theta = data['structure']['theta']
    
    # Process C matrix
    C_temp = data['C_variation']['C']
    repeat_temp = data['C_variation']['repeat']
    C_t = []
    for ct in range(len(C_temp)):
        C_t.append(np.tile(C_temp[ct], (repeat_temp[ct], 1)))
    C = np.concatenate(C_t)
    
    # Other parameters
    n_sampling = data['n_variation']['n_sampling']
    n = None
    Z = data['Z_variation']['Z']
    avarage_clone_in_spot = data['Z_variation']['avarage_clone_in_spot']
    
    phi_gamma = np.array(data['Gamma']['phi_gamma'])
    F_epsilon = np.tile(data['Gamma']['F_epsilon'], (K, 1))
    F_fraction = data['Gamma']['F_fraction']
    F = np.tile(data['Gamma']['F'], (K, 1))
    
    n_lambda = np.tile(data['n_variation']['n_lambda'], (S))
    
    # Gene expression parameters
    p_mean = data['Y_variation']['p_mean']
    p_std = data['Y_variation']['p_std']
    p_y = np.random.normal(p_mean, p_std, g)
    p_y = np.clip(p_y, 0.01, 0.99)  # Keep in valid range
    
    b_alpha_shape = data['Y_variation']['b_alpha_shape']
    b_alpha_var = data['Y_variation']['b_alpha_var']
    b_alpha_scale = np.sqrt(b_alpha_var / b_alpha_shape)
    b_alpha = np.random.gamma(b_alpha_shape, b_alpha_scale, g)
    
    b_beta = data['Y_variation']['b_beta']
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Generating toy simulation with parameters:")
    print(f"K (clones): {K}, S (spots): {S}, I (mutations): {I}, g (genes): {g}")
    print(f"C matrix shape: {C.shape}")
    
    # Generate simulation
    sample = sim.simulation(
        K=K, S=S, g=g, r=phi_gamma[0], q=phi_gamma[1], I=I, F=F, 
        D=None, A=None, C=C, avarage_clone_in_spot=avarage_clone_in_spot, 
        random_seed=42, F_epsilon=F_epsilon, n=n, p_c_binom=None, 
        theta=theta, Z=Z, n_lambda=n_lambda, F_fraction=F_fraction, 
        pi_2D=True, Y=None, p_y=p_y, b_alpha=b_alpha, b_beta=b_beta,
        b_alpha_shape=b_alpha_shape, b_alpha_scale=b_alpha_scale
    )
    
    # Create output directory and save
    os.makedirs('test_sim/1', exist_ok=True)
    pickle.dump(sample, open('test_sim/1/sample_toy', 'wb'))
    
    print("Toy simulation generated and saved to test_sim/1/sample_toy")
    print(f"Sample H shape: {sample.H.shape}")
    print(f"Sample Y shape: {sample.Y.shape}")
    print(f"Sample D shape: {sample.D.shape}")
    print(f"Sample A shape: {sample.A.shape}")
    print(f"Sample B shape: {sample.B.shape}")
    print(f"Mean number of cells per spot: {np.mean(sample.n):.2f}")
    print(f"Mean gene expression per spot: {np.mean(sample.Y):.2f}")
    
    return sample

if __name__ == "__main__":
    generate_toy_simulation()



