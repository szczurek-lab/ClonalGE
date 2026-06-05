# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClonalGE extends the Tumoroscope model for analyzing tumor heterogeneity by integrating spatial transcriptomics with genomic/somatic SNV data. It adds latent variable B_kg (average gene expression per clone) and observed variable Y_sg (gene expression per spot) to infer clonal composition and gene expression patterns. Applied to prostate cancer spatial transcriptomics data (Berglund et al., 2018).

## Running the Code

**Smoke test (verify the installation):**
```bash
python smoke_test.py          # should print "SMOKE OK" and exit 0
```

**Generate simulated data:**
```bash
python generate_simulations.py {run_number} {output_dir} {config_dir} {noise_level}
# Batch: bash scripts/generate_simulations.sh (generates 20 replicates)
```

**Run ClonalGE on simulated data:**
```bash
python main_diff_config.py {run_num} {output_dir} {config_dir} {noise_level}
# Batch: bash scripts/run_main_diff_config.sh
```

**Run ClonalGE on real prostate data:**
```bash
python main_real_1000.py {result_dir} {config_file} {run_sampling} {save_obs} {vis_obs} {vis_results} {saved_inputs} {seed}
# Batch: bash scripts/run_main_real_1000.sh (100 independent runs)
```

## Dependencies

numpy, scipy, pandas, matplotlib, seaborn, pickle, multiprocessing. No requirements.txt or package manager — standard scientific Python stack.

## Architecture

### Repository Layout

```
clonalge/                   ← Python package (core library)
    __init__.py
    model.py                ← ClonalGE Gibbs-sampler (was clonalGE.py)
    tumoroscope.py          ← Tumoroscope baseline model
    simulation.py           ← Synthetic-data generator (ClonalGE)
    simulation_tumoroscope.py ← Synthetic-data generator (Tumoroscope)
    visualization.py        ← Plotting utilities
    pre_processing.py       ← ST + WES data integration
    constants.py            ← Global MCMC settings (CHAINS, CORES)
    run_selection.py        ← Cross-run consensus and chain selection
    select_chains.py        ← Per-run chain selection helper

main_*.py                   ← Entry-point scripts (run from repo root)
plot_*.py                   ← Figure-generation scripts
run_*.py                    ← Analysis and utility scripts
generate_*.py               ← Simulation-data generation scripts
compute_*.py / evaluate_*.py / export_*.py / extract_*.py
smoke_test.py               ← Fast end-to-end integration test

scripts/                    ← Shell scripts (each cd to repo root automatically)
    run_main_diff_config.sh
    run_main_real_1000.sh
    run_figure2_simulations.sh
    run_figure5.sh
    run_clonalge_5chains.sh
    run_sensitivity_cluster.sh
    run_violated_cluster.sh
    …

configs/                    ← JSON configs for simulated data
configs_tumoroscope/        ← JSON configs for Tumoroscope baseline
configs_smoke/              ← Tiny JSON config for smoke test
prostate_data_configs/      ← Real prostate ST + WES data and configs
plots_paper/                ← Final paper figures (tracked)
```

### Core Classes

- **`clonalge/model.py`** — Main inference model (`class clonalGE`). Extends Tumoroscope with gene expression. Implements Gibbs sampling with Metropolis-Hastings steps, adaptive proposal scales, and Geweke convergence diagnostics. Entry point is `gibbs_sampling()`.
- **`clonalge/tumoroscope.py`** — Parent Tumoroscope model without gene expression component. Used as baseline and for initializing ClonalGE (Tumoroscope runs first, then its output seeds ClonalGE).
- **`clonalge/simulation.py`** / **`clonalge/simulation_tumoroscope.py`** — Generate synthetic data with known ground truth for validation.
- **`clonalge/visualization.py`** — Plotting utilities: heatmaps, spatial pie charts, convergence traces, error boxplots.
- **`clonalge/pre_processing.py`** — Data integration: builds clone-mutation matrix C from phyloWGS output, integrates spatial transcriptomics with WES data, handles spot filtering.

### Inference Pipeline

1. Load config (JSON) and data (simulated or real ST + WES)
2. Preprocess: construct C matrix, filter spots, extract gene expression
3. Run Tumoroscope (baseline) → infer H, G estimates
4. Initialize ClonalGE from Tumoroscope output
5. Run ClonalGE Gibbs sampling with gene expression
6. Compute posterior means (after burn-in), export results to text files
7. Evaluate: error metrics (H_SEE, phi_SEE, pi_SEE, B_SEE, etc.) against ground truth for simulations

### Key Model Variables

| Variable | Meaning | Dimension |
|----------|---------|-----------|
| K, S, g, I | Clones, spots, genes, mutations | scalars |
| H | Clone composition per spot | (S, K) |
| B | Gene expression baseline per clone | (K, g) |
| Y | Observed gene expression | (S, g) |
| Z | Clone presence indicator (binary) | (S, K) |
| phi | Overdispersion parameter | (I, K) |
| C | Clone-SNV assignment matrix | (I, K) |
| A, D | Allele counts, read depth | (I, S) |

### Configuration

JSON configs in `configs/` (simulated: normal, low_variance, high_coverage) and `prostate_data_configs/` (real data). Configs control: model dimensions (K, S, I, g), MCMC settings (max_iter, burn_in, batch), distribution hyperparameters, and data file paths.

### Global Constants (`clonalge/constants.py`)

`CHAINS` and `CORES` control parallel chain execution via `multiprocessing.Pool`. Default: 2 chains, 2 cores.

### MCMC Tuning

- Adaptive proposal scales targeting ~0.4 acceptance rate
- Convergence via Geweke test; early stopping at >99.5% convergence
- Burn-in discarded before posterior averaging
- Thinning factor (`every_n_sample`, default 5)
