#!/usr/bin/env python3

import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_toy_data():
    """Explore the generated toy simulation data to understand ClonalGE's inputs and outputs"""
    
    # Load toy configuration and simulation
    with open('configs/toy.json') as f:
        config = json.load(f)
    
    sample = pickle.load(open('test_sim/1/sample_toy', 'rb'))
    
    print("=== ClonalGE Toy Data Exploration ===")
    print("\n1. SIMULATION PARAMETERS:")
    print(f"   K = {sample.K} clones")
    print(f"   S = {sample.S} spots (spatial locations)")
    print(f"   I = {sample.I} mutations")
    print(f"   g = {sample.g} genes")
    print(f"   Average clones per spot = {sample.avarage_clone_in_spot}")
    
    print("\n2. KEY DATA MATRICES:")
    
    # C matrix: clone-mutation relationships
    print(f"\n   C matrix (mutations × clones): {sample.C.shape}")
    print("   Shows which mutations are present in which clones")
    print("   Values: 0=absent, 1=present, 0.5=partial")
    print(f"   Example (first 5 mutations, all clones):")
    print(sample.C[:5])
    
    # H matrix: clone proportions per spot
    print(f"\n   H matrix (spots × clones): {sample.H.shape}")
    print("   Shows proportion of each clone in each spatial spot")
    print("   Values sum to 1 across clones for each spot")
    print(f"   Example (first 5 spots, all clones):")
    print(sample.H[:5])
    print(f"   Sum across clones (should be ~1): {np.sum(sample.H[:5], axis=1)}")
    
    # Z matrix: clone presence indicators
    print(f"\n   Z matrix (spots × clones): {sample.Z.shape}")
    print("   Binary indicators of clone presence in each spot")
    print(f"   Example (first 5 spots, all clones):")
    print(sample.Z[:5])
    
    # Y matrix: gene expression per spot
    print(f"\n   Y matrix (spots × genes): {sample.Y.shape}")
    print("   Gene expression counts per spatial spot")
    print(f"   Example (first 5 spots, first 5 genes):")
    print(sample.Y[:5, :5])
    print(f"   Mean expression per gene: {np.mean(sample.Y, axis=0)[:5]}")
    
    # B matrix: clone-specific gene expression
    print(f"\n   B matrix (clones × genes): {sample.B.shape}")
    print("   Average gene expression per cell for each clone")
    print("   This is what ClonalGE aims to infer!")
    print(f"   Example (all clones, first 5 genes):")
    print(sample.B[:, :5])
    
    # D matrix: sequencing depth per mutation per spot
    print(f"\n   D matrix (mutations × spots): {sample.D.shape}")
    print("   Sequencing read depth for each mutation at each spot")
    print(f"   Example (first 5 mutations, first 5 spots):")
    print(sample.D[:5, :5])
    print(f"   Mean depth per spot: {np.mean(sample.D, axis=0)[:5]}")
    
    # A matrix: alternative allele counts
    print(f"\n   A matrix (mutations × spots): {sample.A.shape}")
    print("   Alternative (mutated) allele counts")
    print(f"   Example (first 5 mutations, first 5 spots):")
    print(sample.A[:5, :5])
    
    # n vector: cell counts per spot
    print(f"\n   n vector (spots): {sample.n.shape}")
    print("   Number of cells in each spatial spot")
    print(f"   Example (first 10 spots): {sample.n[:10]}")
    print(f"   Mean cells per spot: {np.mean(sample.n):.1f}")
    
    print("\n3. CLONALGE MODEL PURPOSE:")
    print("   - Input: Spatial transcriptomics (Y), bulk DNA-seq (D, A), cell counts (n)")
    print("   - Goal: Infer clone locations (H, Z) AND clone-specific expression (B)")
    print("   - Novel contribution: Links spatial genomics with spatial transcriptomics")
    
    print("\n4. BIOLOGICAL INTERPRETATION:")
    print("   - Each spot represents a small tissue region")
    print("   - Each clone has different mutations (C matrix)")
    print("   - Clones are spatially distributed (H matrix)")
    print("   - Each clone has distinct gene expression profile (B matrix)")
    print("   - Observed expression (Y) is mixture of clone expressions weighted by proportions")
    
    # Create output directory and save visualizations
    os.makedirs('toy_results', exist_ok=True)
    
    # Visualize key matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # C matrix heatmap
    sns.heatmap(sample.C, ax=axes[0,0], cmap='RdBu_r', center=0.5, 
                xticklabels=[f'Clone {i+1}' for i in range(sample.K)],
                yticklabels=[f'Mut {i+1}' for i in range(min(sample.I, 10))])
    axes[0,0].set_title('C: Clone-Mutation Matrix')
    
    # H matrix heatmap  
    sns.heatmap(sample.H[:20], ax=axes[0,1], cmap='viridis',
                xticklabels=[f'Clone {i+1}' for i in range(sample.K)],
                yticklabels=[f'Spot {i+1}' for i in range(20)])
    axes[0,1].set_title('H: Clone Proportions (first 20 spots)')
    
    # Z matrix heatmap
    sns.heatmap(sample.Z[:20], ax=axes[0,2], cmap='RdBu_r',
                xticklabels=[f'Clone {i+1}' for i in range(sample.K)],
                yticklabels=[f'Spot {i+1}' for i in range(20)])
    axes[0,2].set_title('Z: Clone Presence (first 20 spots)')
    
    # Y matrix heatmap
    sns.heatmap(sample.Y[:20], ax=axes[1,0], cmap='plasma',
                xticklabels=[f'Gene {i+1}' for i in range(sample.g)],
                yticklabels=[f'Spot {i+1}' for i in range(20)])
    axes[1,0].set_title('Y: Gene Expression (first 20 spots)')
    
    # B matrix heatmap
    sns.heatmap(sample.B, ax=axes[1,1], cmap='plasma',
                xticklabels=[f'Gene {i+1}' for i in range(sample.g)],
                yticklabels=[f'Clone {i+1}' for i in range(sample.K)])
    axes[1,1].set_title('B: Clone-Specific Expression')
    
    # Cell count histogram
    axes[1,2].hist(sample.n, bins=15, alpha=0.7, color='skyblue')
    axes[1,2].set_xlabel('Cells per spot')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('n: Cell Count Distribution')
    
    plt.tight_layout()
    plt.savefig('toy_results/toy_data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n5. VISUALIZATION SAVED:")
    print("   toy_results/toy_data_overview.png - Overview of all key matrices")
    
    # Show relationship between true H and Y
    print("\n6. TESTING CLONE-EXPRESSION RELATIONSHIP:")
    print("   Y = n * H * B  (observed expression = cells × proportions × clone expression)")
    
    # Calculate expected Y from true values
    Y_expected = np.zeros_like(sample.Y)
    for s in range(sample.S):
        for g in range(sample.g):
            Y_expected[s, g] = sample.n[s] * np.sum(sample.H[s, :] * sample.B[:, g])
    
    correlation = np.corrcoef(sample.Y.flatten(), Y_expected.flatten())[0, 1]
    print(f"   Correlation between true Y and calculated Y: {correlation:.3f}")
    print("   (High correlation confirms the model's mathematical foundation)")
    
    return sample

if __name__ == "__main__":
    sample = explore_toy_data()



