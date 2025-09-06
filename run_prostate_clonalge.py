#!/usr/bin/env python3
"""
Run ClonalGE on the prostate cancer dataset
"""

import subprocess
import sys
import os

def run_clonalge_prostate():
    """Run ClonalGE on the prostate cancer dataset"""
    
    print("🚀 Starting ClonalGE on Prostate Cancer Dataset")
    print("=" * 60)
    
    # Check if all required files exist
    required_files = [
        "prostate_data_configs/config_selected_spots_any_mutations_prostate.json",
        "prostate_data_configs/STdata/P1.2.tsv",
        "prostate_data_configs/STdata/P2.4.tsv", 
        "prostate_data_configs/STdata/P3.3.tsv",
        "prostate_data_configs/prostate_cell_count_annotation_any_fixed_margin.txt",
        "prostate_data_configs/vardict2_st_calling/vardict2_1.2.ac",
        "prostate_data_configs/vardict2_st_calling/vardict2_2.4.ac",
        "prostate_data_configs/vardict2_st_calling/vardict2_3.3.ac",
        "prostate_data_configs/ssm_data.txt",
        "prostate_data_configs/C_tree_1.txt",
        "prostate_data_configs/F_tree_1.txt"
    ]
    
    print("🔍 Checking required files...")
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Please ensure all required files are present before running ClonalGE.")
        return False
    
    print(f"\n✅ All required files found!")
    
    # Command parameters
    result_dir = "prostate_results"
    config_file = "prostate_data_configs/config_selected_spots_any_mutations_prostate.json"
    
    # Boolean parameters (True/False)
    run_sampling = "True"      # Run the Gibbs sampling
    save_observed = "True"     # Save observed data
    vis_observed = "True"      # Visualize observed data
    vis_results = "True"       # Visualize results
    saved_inputs = "False"     # Use saved inputs (False for fresh run)
    seed = "42"                # Random seed
    
    # Build the command
    cmd = [
        "python", "main_real_1000.py",
        result_dir,
        config_file,
        run_sampling,
        save_observed,
        vis_observed,
        vis_results,
        saved_inputs,
        seed
    ]
    
    print(f"\n🎯 Running ClonalGE with parameters:")
    print(f"  Result directory: {result_dir}")
    print(f"  Config file: {config_file}")
    print(f"  Run sampling: {run_sampling}")
    print(f"  Save observed: {save_observed}")
    print(f"  Visualize observed: {vis_observed}")
    print(f"  Visualize results: {vis_results}")
    print(f"  Saved inputs: {saved_inputs}")
    print(f"  Seed: {seed}")
    
    print(f"\n📋 Command:")
    print(" ".join(cmd))
    
    print(f"\n⏳ Starting ClonalGE analysis...")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n🎉 ClonalGE completed successfully!")
            print(f"📁 Results saved in: {result_dir}")
            print(f"📊 Check the log file: {result_dir}/log.txt")
        else:
            print(f"\n❌ ClonalGE failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running ClonalGE: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_clonalge_prostate()
    if success:
        print(f"\n🚀 Prostate cancer analysis complete!")
        print(f"📈 Check the results in the 'prostate_results' directory")
    else:
        print(f"\n❌ Analysis failed. Please check the error messages above.")
        sys.exit(1)




