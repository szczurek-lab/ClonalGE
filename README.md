# ClonalGE: Spatial Clonal Genomics with Gene Expression

ClonalGE is an extension of the [Tumoroscope](https://www.nature.com/articles/s41467-024-53374-3) model that integrates **spatial transcriptomics** with **spatial genomics** for comprehensive cancer clone characterization. It's the first method to map both WHERE cancer clones are located in tissue AND WHAT each clone expresses.

## 🎯 What ClonalGE Does

- **Tumoroscope**: Maps WHERE cancer clones are located in tissue
- **ClonalGE**: ALSO maps WHAT each clone expresses (gene expression profiles)

### Key Innovation
ClonalGE adds one more latent variable (B<sub>kg</sub> - average expression of gene g per one cell of clone k), one observed variable (Y<sub>sg</sub> - gene expression of gene g in spot s), and several new hyperparameters to the original Tumoroscope framework.

## 📁 Project Structure

### Core Classes
- `clonalGE.py` - Main ClonalGE model implementation
- `tumoroscope.py` - Base Tumoroscope model
- `simulation.py` - Data simulation utilities
- `visualization.py` - Results visualization

### Main Scripts
- `main_real_1000.py` - Run ClonalGE on real cancer data
- `main_diff_config.py` - Run ClonalGE on simulated data
- `generate_simulations.py` - Generate synthetic datasets

### Configuration Directories
- `configs/` - Configuration files for different simulation setups
  - `toy.json` - Small toy simulation (3 clones, 50 spots, 10 genes)
  - `normal.json` - Standard simulation setup
  - `high_coverage.json` - High coverage simulation
  - `low_variance.json` - Low variance simulation
- `prostate_data_configs/` - Real prostate cancer dataset configuration
- `example_cancer_data/` - Example data structure template

## 🚀 Quick Start

### Running on Real Data (Prostate Cancer Example)

```bash
# Option 1: Use the automated script (recommended)
python run_prostate_clonalge.py

# Option 2: Run directly with progress bar
python main_real_1000.py prostate_results prostate_data_configs/config_selected_spots_any_mutations_prostate.json true true true true false true 42
```

**Command Parameters:**
1. `prostate_results` - Output directory
2. `prostate_data_configs/config_selected_spots_any_mutations_prostate.json` - Configuration file
3. `true` - Run sampling (Gibbs sampling)
4. `true` - Save observed data
5. `true` - Visualize observed data
6. `true` - Visualize results
7. `false` - Use saved inputs (false = fresh run)
8. `true` - Show progress bar (NEW!)
9. `42` - Random seed

### Running on Synthetic Data

```bash
# Generate synthetic data
python generate_simulations.py

# Run ClonalGE on synthetic data
python main_diff_config.py synthetic_results configs/normal.json true true true true false 42

# Quick toy simulation (for testing)
python generate_toy_simulation.py  # Generate toy data
python run_simple_toy.py           # Run ClonalGE on toy data

# Complete toy demonstration
python run_toy_demonstration.py    # Full workflow demonstration
```

**Note**: Toy simulations demonstrate parameter recovery from synthetic data. They show how ClonalGE learns clone proportions (H) and clone-specific gene expression (B) from spatial transcriptomics data. For production use, use larger datasets or real cancer data.

## 🧪 Toy Experiments: Complete Setup Guide

### Understanding Toy Simulations

Toy simulations are perfect for:
- **Learning how ClonalGE works** - See parameter recovery in action
- **Testing your setup** - Verify everything works before using real data
- **Understanding the model** - See how spatial transcriptomics relates to clone inference
- **Parameter tuning** - Test different configurations safely

### Step-by-Step Toy Setup

#### 1. **Generate Synthetic Data**
```bash
python generate_toy_simulation.py
```
This creates a small synthetic dataset with known parameters:
- **3 clones** across **50 spots** with **10 genes**
- **30 mutations** with known clone assignments
- **True parameters** stored for comparison

#### 2. **Run Parameter Recovery**
```bash
python run_simple_toy.py
```
This runs ClonalGE to recover the true parameters and shows:
- **Progress bar** with convergence monitoring
- **Parameter recovery accuracy** (correlation with true values)
- **Example comparisons** of true vs inferred values

#### 3. **Complete Demonstration**
```bash
python run_toy_demonstration.py
```
This runs the full workflow with explanations of each step.

### 🎛️ Configuration: Understanding Parameters

The toy simulation uses `configs/toy.json`. Here's what each parameter does:

#### **Structure Parameters**
```json
{
  "structure": {
    "K": 3,        // Number of clones
    "S": 50,       // Number of spatial spots  
    "I": 30,       // Number of mutations
    "g": 10,       // Number of genes
    "theta": 1     // Mutation rate parameter
  }
}
```

#### **Clone-Mutation Matrix (C)**
```json
{
  "C_variation": {
    "C": [[0, 1, 0.5], [0, 1, 0], [0, 0, 1]],  // Clone-mutation assignments
    "repeat": [10, 15, 5]                       // How many mutations per pattern
  }
}
```
- **C matrix**: Defines which mutations belong to which clones
- **repeat**: How many mutations follow each pattern

#### **Sampling Parameters**
```json
{
  "sampling": {
    "max_iter": 200,    // Maximum iterations
    "min_iter": 50,     // Minimum iterations  
    "burn_in": 30,      // Burn-in period
    "batch": 25         // Batch size for convergence testing
  }
}
```

#### **Gene Expression Parameters**
```json
{
  "Y_variation": {
    "b_alpha_shape": 1,     // Shape parameter for gene expression
    "b_alpha_var": 0.25,    // Variance parameter
    "b_beta": 3.5,          // Beta parameter for expression
    "p_mean": 0.045,        // Mean expression probability
    "p_std": 0.02           // Expression probability std dev
  }
}
```

### 🔧 Customizing Toy Experiments

#### **Create Your Own Toy Configuration**

1. **Copy the toy config**:
```bash
cp configs/toy.json configs/my_toy.json
```

2. **Modify parameters**:
```json
{
  "structure": {
    "K": 4,        // More clones
    "S": 100,      // More spots
    "I": 50,       // More mutations
    "g": 20        // More genes
  }
}
```

3. **Update the generation script** to use your config:
```python
# In generate_toy_simulation.py, change:
with open('configs/my_toy.json') as f:
    data = json.load(f)
```

#### **Adjusting Sampling Parameters**

For better parameter recovery:
```json
{
  "sampling": {
    "max_iter": 1000,   // More iterations
    "min_iter": 200,    // Longer burn-in
    "batch": 50         // More frequent convergence testing
  }
}
```

#### **Understanding Results**

The toy simulation shows:
- **MAE (Mean Absolute Error)**: Lower is better
- **Correlation**: Higher is better (closer to 1.0)
- **Convergence percentages**: Should increase over time

**Good recovery**: H correlation > 0.8, B correlation > 0.7
**Moderate recovery**: H correlation > 0.6, B correlation > 0.5

### 🚀 From Toy to Real Data

Once you understand the toy simulation:
1. **Use larger synthetic data**: `python generate_simulations.py`
2. **Try real prostate data**: Follow the prostate cancer example
3. **Prepare your own data**: Use the data structure template above

### 📋 Quick Reference: Available Configurations

| Configuration | Use Case | Clones | Spots | Genes | Mutations |
|---------------|----------|--------|-------|-------|-----------|
| `toy.json` | Learning/testing | 3 | 50 | 10 | 30 |
| `normal.json` | Standard simulation | 5 | 300 | 50 | 200 |
| `high_coverage.json` | High coverage data | 5 | 300 | 50 | 200 |
| `low_variance.json` | Low variance scenario | 5 | 300 | 50 | 200 |

**Note**: The main difference between `normal.json`, `high_coverage.json`, and `low_variance.json` is in the sampling and variance parameters, not the structure.

**Prostate Data**: Real cancer dataset with 4 clones, 294 spots, 12,873 genes, 333 mutations

## 📊 Data Requirements

To run ClonalGE on your own cancer data, you need:

### 1. 🧬 Spatial Transcriptomics Data
- **Format**: TSV files with genes (rows) × spots (columns)
- **Location**: `STdata/section_name.tsv`
- **Source**: 10x Visium, Slide-seq, or similar spatial transcriptomics

### 2. 🔬 Variant Calling Data
- **Format**: AC files with mutation counts per spot
- **Location**: `vardict_calling/section_name.ac`
- **Source**: Variant calling on spatial transcriptomics reads

### 3. 🌳 Clone Tree
- **Format**: Tab-separated matrix of mutations × clones
- **File**: `C_tree.txt`
- **Source**: Phylogenetic reconstruction from bulk sequencing

### 4. 🏥 Cell Count Annotations
- **Format**: CSV with spot coordinates and nuclei counts
- **File**: `cell_counts.txt`
- **Source**: H&E image analysis

### 5. 📋 Gene List
- **Format**: Text file with one gene per line
- **File**: `selected_genes.txt`
- **Content**: 1000+ highly variable genes

### 6. ⚙️ Configuration File
- **Format**: JSON file pointing to all data files
- **File**: `config.json`
- **Purpose**: Defines analysis parameters and file paths

## 📁 Complete Data Structure Template

```
my_cancer_data/
├── config.json                    # Configuration file
├── C_tree.txt                     # Clone-mutation matrix
├── cell_counts.txt                # Cell counts per spot
├── selected_genes.txt             # Gene list for analysis
├── STdata/                        # Spatial transcriptomics
│   ├── section1.tsv
│   ├── section2.tsv
│   └── section3.tsv
└── vardict_calling/               # Variant calls
    ├── section1.ac
    ├── section2.ac
    └── section3.ac
```

## 🎛️ New Features

### Progress Bar Support
ClonalGE now includes optional progress bars for better monitoring:

```bash
# With progress bar
python main_real_1000.py results config.json true true true true false true 42

# Without progress bar (default)
python main_real_1000.py results config.json true true true true false false 42
```

### Clean Console Output
All warnings have been resolved for a clean, professional output experience.

## 📈 Expected Outputs

After running ClonalGE, you'll get:

### 📊 Results Files
- `section_h.txt` - Clone proportions per spot (H matrix)
- `section_z.txt` - Clone presence indicators (Z matrix)
- `section_n.txt` - Inferred cell counts
- `section_b.npy` - Clone-specific gene expression (B matrix)
- `log.txt` - Detailed execution log

### 📈 Visualizations
- Clone distribution heatmaps
- Spatial clone maps with pie charts
- Gene expression profiles per clone
- Model convergence diagnostics
- Clone co-localization patterns

### 🔬 Key Insights
- **Spatial clone mapping**: Where each clone is located in tissue
- **Clone-specific expression**: What genes each clone expresses
- **Tumor heterogeneity**: Functional differences between clones
- **Spatial patterns**: Co-localization and exclusion patterns

## 🧪 Example Datasets

### Prostate Cancer Dataset
- **Location**: `prostate_data_configs/`
- **Sections**: P1.2, P2.4, P3.3
- **Genes**: 12,873 genes across 294 spots
- **Clones**: 4 clones with proportions and cell counts
- **Mutations**: 333 mutations in clone tree

### Toy Simulation
- **Location**: `configs/toy.json`
- **Purpose**: Quick testing and validation
- **Size**: Small dataset for rapid iteration

## 🔧 Technical Details

### Model Components
- **H**: Clone proportions per spot (S × K matrix)
- **Z**: Clone presence indicators (S × K matrix)
- **B**: Clone-specific gene expression (K × G matrix)
- **n**: Cell counts per spot (S vector)
- **pi**: Clone prevalence probabilities
- **phi**: Mutation frequency parameters

### Sampling Parameters
- **Default iterations**: 30,000 max, 10,000 min
- **Batch size**: 1,000 iterations
- **Burn-in**: 10,000 iterations
- **Convergence**: Automatic early stopping at 99.5% convergence

## 📚 References

[1] Shafighi, Shadi, et al. "Integrative spatial and genomic analysis of tumor heterogeneity with Tumoroscope." Nature Communications 15.1 (2024): 9343.  

[2] Berglund, Emelie, et al. "Spatial maps of prostate cancer transcriptomes reveal an unexplored landscape of heterogeneity." Nature communications 9.1 (2018): 1-13.

## 🆘 Troubleshooting

### Common Issues
1. **Gene name mismatch**: Ensure gene names in expression data match your gene list
2. **Missing data files**: Check that all required files are in the correct locations
3. **Memory issues**: Reduce the number of genes or spots for large datasets
4. **Convergence problems**: Increase iteration counts or adjust sampling parameters

### Getting Help
- Check the log file for detailed error messages
- Verify your data format matches the expected structure
- Use the toy simulation to test your setup

---

**Ready to analyze your spatial cancer data? Start with the prostate cancer example or prepare your own dataset following the structure above!**