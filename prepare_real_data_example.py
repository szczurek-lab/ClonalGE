#!/usr/bin/env python3
"""
Example script showing how to prepare real data for ClonalGE analysis.
This demonstrates the data preprocessing steps for spatial transcriptomics + genomics integration.
"""

import pandas as pd
import numpy as np
import json
import os

def create_example_data_structure():
    """
    Creates an example data structure showing the format needed for ClonalGE.
    This is a template - replace with your actual data processing.
    """
    
    print("🔧 Creating Example Data Structure for ClonalGE")
    
    # Create main data directory
    data_dir = "example_cancer_data"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/STdata", exist_ok=True)
    os.makedirs(f"{data_dir}/vardict_calling", exist_ok=True)
    
    print(f"📁 Created directory: {data_dir}")
    
    # 1. Create example spatial transcriptomics data
    create_example_expression_data(data_dir)
    
    # 2. Create example variant calling data
    create_example_variant_data(data_dir)
    
    # 3. Create example clone tree
    create_example_clone_tree(data_dir)
    
    # 4. Create example cell count data
    create_example_cell_counts(data_dir)
    
    # 5. Create example bulk DNA-seq data
    create_example_bulk_data(data_dir)
    
    # 6. Create example gene list
    create_example_gene_list(data_dir)
    
    # 7. Create configuration file
    create_example_config(data_dir)
    
    print("\n✅ Example data structure created!")
    print(f"📂 Check the '{data_dir}' directory for all files")
    print("\n🚀 To run ClonalGE on this example data:")
    print(f"python main_real_1000.py results_example {data_dir}/config.json true true true true false 42")

def create_example_expression_data(data_dir):
    """Create example spatial transcriptomics expression matrices"""
    
    genes = ["FUCA1", "RNF126", "CYB5A", "CAMKK2", "FOLH1", "GUSB", "BRD4", "ARID5A"]
    
    for section in ["T1.1", "T1.2"]:
        # Create example spots
        spots = [f"{section}_{i}x{j}" for i in range(10, 20) for j in range(15, 25)][:50]
        
        # Generate realistic expression data (using negative binomial-like distribution)
        np.random.seed(42)
        expression_data = np.random.negative_binomial(n=5, p=0.3, size=(len(genes), len(spots)))
        
        # Create DataFrame
        expr_df = pd.DataFrame(expression_data, index=genes, columns=spots)
        
        # Save as TSV
        expr_df.to_csv(f"{data_dir}/STdata/{section}.tsv", sep='\t')
        print(f"📊 Created expression data: STdata/{section}.tsv ({len(genes)} genes, {len(spots)} spots)")

def create_example_variant_data(data_dir):
    """Create example variant calling data"""
    
    chromosomes = ["1", "2", "3", "17", "X"]
    
    for section in ["T1.1", "T1.2"]:
        # Create example variant calls
        variants = []
        
        for i, chrom in enumerate(chromosomes):
            for pos in range(1000000 + i*1000, 1000000 + i*1000 + 20, 100):
                for spot_idx in range(25):  # Some spots have variants
                    if np.random.random() > 0.7:  # 30% chance of variant
                        variants.append({
                            'refContig': chrom,
                            'refPos': pos,
                            'refAllele': 'A',
                            'base': 'T',
                            'source': section,
                            'spot': f"{section}_{10+spot_idx//5}x{15+spot_idx%5}",
                            'cnt': np.random.randint(1, 10)
                        })
        
        # Create DataFrame and save
        variant_df = pd.DataFrame(variants)
        variant_df.to_csv(f"{data_dir}/vardict_calling/{section}.ac", sep='\t', index=False)
        print(f"🧬 Created variant data: vardict_calling/{section}.ac ({len(variants)} variants)")

def create_example_clone_tree(data_dir):
    """Create example clone-mutation matrix"""
    
    # Define mutations (chromosome position format)
    mutations = [
        "1 1000000", "1 1000100", "1 1000200",
        "2 1001000", "2 1001100", "2 1001200", "2 1001300",
        "3 1002000", "3 1002100",
        "17 1003000", "17 1003100", "17 1003200", "17 1003300",
        "X 1004000", "X 1004100"
    ]
    
    # Define clone profiles (4 clones)
    # Clone 1: Early mutations
    # Clone 2: Branched from Clone 1 + new mutations  
    # Clone 3: Different branch
    # Clone 4: Highly mutated
    
    clone_data = []
    for mutation in mutations:
        if "1 100" in mutation:  # Early mutations in all clones
            clone_data.append([mutation, 0, 0.65, 0.65, 0.65])
        elif "2 100" in mutation:  # Clone 2 and 4 specific
            clone_data.append([mutation, 0, 0.64, 0, 0.64])
        elif "3 100" in mutation:  # Clone 3 specific
            clone_data.append([mutation, 0, 0, 0.66, 0])
        elif "17 100" in mutation:  # Clone 4 specific
            clone_data.append([mutation, 0, 0, 0, 0.65])
        elif "X 100" in mutation:  # Clone 2 and 3
            clone_data.append([mutation, 0, 0.63, 0.63, 0])
    
    # Create DataFrame
    clone_df = pd.DataFrame(clone_data, columns=['mutation_id', 'clone1', 'clone2', 'clone3', 'clone4'])
    
    # Save with tab separation and proper header
    with open(f"{data_dir}/C_tree.txt", 'w') as f:
        f.write("clone1\tclone2\tclone3\tclone4\n")
        for _, row in clone_df.iterrows():
            f.write(f"{row['mutation_id']}\t{row['clone1']}\t{row['clone2']}\t{row['clone3']}\t{row['clone4']}\n")
    
    print(f"🌳 Created clone tree: C_tree.txt ({len(mutations)} mutations, 4 clones)")

def create_example_cell_counts(data_dir):
    """Create example cell count data"""
    
    cell_data = []
    
    for section in ["T1.1", "T1.2"]:
        for i in range(10, 20):
            for j in range(15, 25):
                if np.random.random() > 0.3:  # 70% of spots are cancer
                    cell_data.append({
                        'section': section,
                        'coordinates': f"{i}x{j}",
                        'nuclei': np.random.randint(5, 50),  # 5-50 cells per spot
                        'type': 'Cancer',
                        'x': i,
                        'y': j,
                        'barcode': f"{section}_{i}x{j}"
                    })
    
    # Create DataFrame and save
    cell_df = pd.DataFrame(cell_data)
    cell_df.to_csv(f"{data_dir}/cell_counts.txt", index=False)
    print(f"🏥 Created cell counts: cell_counts.txt ({len(cell_data)} spots)")

def create_example_bulk_data(data_dir):
    """Create example bulk DNA sequencing reference data"""
    
    mutations = [
        "1 1000000", "1 1000100", "1 1000200",
        "2 1001000", "2 1001100", "2 1001200", "2 1001300", 
        "3 1002000", "3 1002100",
        "17 1003000", "17 1003100", "17 1003200", "17 1003300",
        "X 1004000", "X 1004100"
    ]
    
    bulk_data = []
    for i, mutation in enumerate(mutations):
        # Simulate bulk sequencing data (3 samples)
        alt_counts = [np.random.randint(5, 25) for _ in range(3)]
        total_counts = [ac + np.random.randint(20, 80) for ac in alt_counts]
        
        bulk_data.append({
            'id': f's{i}',
            'gene': mutation,
            'a': ','.join(map(str, alt_counts)),
            'd': ','.join(map(str, total_counts)),
            'mu_r': 0.999,
            'mu_v': 0.499
        })
    
    # Create DataFrame and save
    bulk_df = pd.DataFrame(bulk_data)
    bulk_df.to_csv(f"{data_dir}/ssm_data.txt", sep='\t', index=False)
    print(f"🔬 Created bulk data: ssm_data.txt ({len(mutations)} mutations)")

def create_example_gene_list(data_dir):
    """Create example gene list"""
    
    genes = [
        "FUCA1 ENSG00000179163",
        "RNF126 ENSG00000070423", 
        "CYB5A ENSG00000166347",
        "CAMKK2 ENSG00000110931",
        "FOLH1 ENSG00000086205",
        "GUSB ENSG00000169919",
        "BRD4 ENSG00000141867",
        "ARID5A ENSG00000196843"
    ]
    
    with open(f"{data_dir}/selected_genes.txt", 'w') as f:
        for gene in genes:
            f.write(f"{gene}\n")
    
    print(f"📋 Created gene list: selected_genes.txt ({len(genes)} genes)")

def create_example_config(data_dir):
    """Create example configuration file"""
    
    config = {
        "criteria": {
            "offset": 1,
            "st_read_limit": 0,
            "var_calculation": True
        },
        "results": {
            "text_result": "inferred_vars_",
            "data_configs": f"{data_dir}/"
        },
        "structure": {
            "section": "all",
            "optimal_rate": 0.40,
            "pi_2D": True,
            "all_spots": False
        },
        "observed": {
            "T1.1": f"{data_dir}/vardict_calling/T1.1.ac",
            "T1.2": f"{data_dir}/vardict_calling/T1.2.ac"
        },
        "C_variation": {
            "WES_file": f"{data_dir}/ssm_data.txt",
            "selected_canopy_tree": f"{data_dir}/C_tree.txt",
            "method": "canopy"
        },
        "n_variation": {
            "n_sampling": True,
            "n_file": f"{data_dir}/cell_counts.txt"
        },
        "Z_variation": {
            "avarage_clone_in_spot": 1.5,
            "threshold": 0.8
        },
        "sampling": {
            "max_iter": 5000,  # Reduced for example
            "min_iter": 2000,
            "batch": 500,
            "burn_in": 1000,
            "every_n_sample": 5,
            "changes_batch": 200
        },
        "Gamma": {
            "phi_gamma": [0.1, 0.5],
            "F_epsilon": [2, 1],
            "F_fraction": False,
            "F_file": f"{data_dir}/F_tree.txt"
        },
        "theta": {
            "gamma": 1,
            "theta_variable": False
        },
        "expression": {
            "T1.1": f"{data_dir}/STdata/T1.1.tsv",
            "T1.2": f"{data_dir}/STdata/T1.2.tsv"
        }
    }
    
    with open(f"{data_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️ Created config: config.json")

if __name__ == "__main__":
    create_example_data_structure()
    print("\n📖 See REAL_DATA_GUIDE.md for detailed explanations of each file format")
    print("🔄 Modify the generated files with your actual data following the same structure")



