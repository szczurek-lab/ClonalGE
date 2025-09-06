#!/usr/bin/env python3
"""
Test the negative binomial regression function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, minimize
from scipy.stats import nbinom

def calc_B_negative_binomial(Y, H, N):
    """Calculate B matrix using negative binomial regression (Tumoroscope+NB approach)"""
    K = H.shape[1]
    G = Y.shape[1]
    S = Y.shape[0]
    B = np.zeros((K, G))
    
    # Create design matrix: X = N * H (cell counts × clone proportions)
    X = np.zeros((S, K))
    for s in range(S):
        for k in range(K):
            X[s, k] = N[s] * H[s, k]
    
    for g in range(G):
        if Y[:, g].sum() > 0:
            y_g = Y[:, g]
            
            # Negative binomial regression: Y ~ NB(mean, dispersion)
            # where mean = X * B and we need to estimate B and dispersion
            
            def neg_binom_loglik(params):
                """Negative log-likelihood for negative binomial regression"""
                B_g = params[:K]  # Clone-specific expression for gene g
                r = params[K]     # Dispersion parameter (r > 0)
                
                # Ensure non-negative expression and positive dispersion
                if np.any(B_g < 0) or r <= 0:
                    return 1e10
                
                # Calculate mean expression for each spot
                mu = np.dot(X, B_g)
                
                # Avoid numerical issues
                mu = np.maximum(mu, 1e-6)
                
                # Negative binomial log-likelihood
                # P(Y=y) = Γ(y+r)/(Γ(r)Γ(y+1)) * (r/(r+μ))^r * (μ/(r+μ))^y
                try:
                    # Using scipy's negative binomial PMF
                    loglik = 0
                    for s in range(S):
                        if y_g[s] >= 0:  # Only for non-negative counts
                            loglik += nbinom.logpmf(y_g[s], r, r/(r + mu[s]))
                    return -loglik  # Return negative log-likelihood for minimization
                except:
                    return 1e10
            
            # Initial parameters: B from linear regression, r=1
            try:
                # Get initial B from linear regression
                B_init = lsq_linear(X, y_g, bounds=(0, np.inf)).x
                r_init = 1.0
                initial_params = np.concatenate([B_init, [r_init]])
                
                # Bounds: B >= 0, r > 0
                bounds = [(0, None)] * K + [(1e-6, None)]
                
                # Optimize
                result = minimize(neg_binom_loglik, initial_params, bounds=bounds, 
                                method='L-BFGS-B', options={'maxiter': 1000})
                
                if result.success:
                    B[:, g] = result.x[:K]
                    print(f"Gene {g}: NB regression successful, r={result.x[K]:.3f}")
                else:
                    # Fallback to linear regression if NB regression fails
                    B[:, g] = lsq_linear(X, y_g, bounds=(0, np.inf)).x
                    print(f"Gene {g}: NB regression failed, using LR fallback")
                    
            except Exception as e:
                # Fallback to linear regression if anything goes wrong
                B[:, g] = lsq_linear(X, y_g, bounds=(0, np.inf)).x
                print(f"Gene {g}: Exception in NB regression, using LR fallback: {e}")
    
    return B

def calc_B_linear(Y, H, N):
    """Calculate B matrix using linear regression (Tumoroscope+LR approach)"""
    K = H.shape[1]
    G = Y.shape[1]
    N = N * np.eye(len(N))
    X = np.matmul(N,H)
    B = np.zeros((K,G))
    for g in range(G):
        if Y[:,g].sum() > 0:
            B[:,g] = lsq_linear(X, Y[:,g], bounds=(0, np.inf)).x
    return B

def test_negative_binomial():
    """Test the negative binomial regression with synthetic data"""
    print("🧪 Testing Negative Binomial Regression")
    print("=" * 40)
    
    # Create synthetic data
    np.random.seed(42)
    K = 3  # 3 clones
    S = 50  # 50 spots
    G = 10  # 10 genes
    
    # True parameters
    H_true = np.random.dirichlet([1, 1, 1], S)  # Clone proportions per spot
    N_true = np.random.poisson(100, S)  # Cell counts per spot
    B_true = np.random.exponential(5, (K, G))  # True clone-specific expression
    
    # Generate expression data using negative binomial
    Y = np.zeros((S, G))
    for s in range(S):
        for g in range(G):
            # Expected expression = sum over clones of (cell_count * clone_proportion * clone_expression)
            mu = N_true[s] * np.sum(H_true[s, :] * B_true[:, g])
            # Generate negative binomial count
            r = 5  # dispersion parameter
            p = r / (r + mu)  # probability parameter
            Y[s, g] = np.random.negative_binomial(r, p)
    
    print(f"Generated synthetic data:")
    print(f"  K (clones): {K}")
    print(f"  S (spots): {S}")
    print(f"  G (genes): {G}")
    print(f"  Expression range: {Y.min():.1f} - {Y.max():.1f}")
    print()
    
    # Test linear regression
    print("📊 Testing Linear Regression...")
    B_lr = calc_B_linear(Y, H_true, N_true)
    print(f"  Linear regression completed")
    
    # Test negative binomial regression
    print("📊 Testing Negative Binomial Regression...")
    B_nb = calc_B_negative_binomial(Y, H_true, N_true)
    print(f"  Negative binomial regression completed")
    print()
    
    # Compare results
    print("📈 Results Comparison:")
    print(f"  True B range: {B_true.min():.3f} - {B_true.max():.3f}")
    print(f"  LR B range: {B_lr.min():.3f} - {B_lr.max():.3f}")
    print(f"  NB B range: {B_nb.min():.3f} - {B_nb.max():.3f}")
    
    # Calculate correlations with true values
    lr_corr = np.corrcoef(B_true.flatten(), B_lr.flatten())[0, 1]
    nb_corr = np.corrcoef(B_true.flatten(), B_nb.flatten())[0, 1]
    
    print(f"  LR correlation with true: {lr_corr:.3f}")
    print(f"  NB correlation with true: {nb_corr:.3f}")
    
    # Calculate prediction accuracy
    Y_pred_lr = np.zeros_like(Y)
    Y_pred_nb = np.zeros_like(Y)
    
    for s in range(S):
        for g in range(G):
            Y_pred_lr[s, g] = N_true[s] * np.sum(H_true[s, :] * B_lr[:, g])
            Y_pred_nb[s, g] = N_true[s] * np.sum(H_true[s, :] * B_nb[:, g])
    
    lr_pred_corr = np.corrcoef(Y.flatten(), Y_pred_lr.flatten())[0, 1]
    nb_pred_corr = np.corrcoef(Y.flatten(), Y_pred_nb.flatten())[0, 1]
    
    print(f"  LR prediction correlation: {lr_pred_corr:.3f}")
    print(f"  NB prediction correlation: {nb_pred_corr:.3f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: True vs Predicted B
    axes[0].scatter(B_true.flatten(), B_lr.flatten(), alpha=0.6, label=f'LR (r={lr_corr:.3f})')
    axes[0].scatter(B_true.flatten(), B_nb.flatten(), alpha=0.6, label=f'NB (r={nb_corr:.3f})')
    axes[0].plot([0, B_true.max()], [0, B_true.max()], 'r--', alpha=0.7)
    axes[0].set_xlabel('True B')
    axes[0].set_ylabel('Predicted B')
    axes[0].set_title('Clone Expression (B) Recovery')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: True vs Predicted Y
    axes[1].scatter(Y.flatten(), Y_pred_lr.flatten(), alpha=0.6, label=f'LR (r={lr_pred_corr:.3f})')
    axes[1].scatter(Y.flatten(), Y_pred_nb.flatten(), alpha=0.6, label=f'NB (r={nb_pred_corr:.3f})')
    axes[1].plot([0, Y.max()], [0, Y.max()], 'r--', alpha=0.7)
    axes[1].set_xlabel('True Expression')
    axes[1].set_ylabel('Predicted Expression')
    axes[1].set_title('Gene Expression (Y) Prediction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_negative_binomial.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Test completed! Check 'test_negative_binomial.png' for visualization")
    
    return lr_corr, nb_corr, lr_pred_corr, nb_pred_corr

if __name__ == "__main__":
    test_negative_binomial()
