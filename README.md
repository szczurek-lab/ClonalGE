# ClonalGE
This in an extention of Tumoroscope model [1]. It adds one more latent variable (B<sub>kg</sub> - average expression of gene g per one cell of clone k), one observed variable (Y<sub>sg</sub> - gene expression of gene g in spot s) and several new hyperparameters.

There are 3 main classes:
* `clonalGE.py`
* `simulation.py`
* `visualization.py`

and additional scripts, including:
* `main_diff_config.py` and `run_main_diff_config.sh` - for running ClonalGE on simulated data
* `generate_simulations.py` and `generate_simulations.sh` - for generating the simulated data
* `main_real_1000.py` and `run_main_real_1000.sh` - for running ClonalGE on the real data

The directory `configs` contains json files describing 3 different setups used to generate the simulations.
The directory `prostate_data_configs` contains files necessary for running models on the real data (prostate cancer dataset [2])

## Figure scripts

Each publication figure has a dedicated script in the root directory:

| Script | Figure | Description |
|---|---|---|
| `plot_figure3.py` | **Fig 3** | Top 30 differentially expressed genes: (a) min-max scaled B heatmap clustered by gene/clone; (b) posterior P(B_k1 > B_k2 \| Data) per clone pair |
| `plot_figure4.py` | **Fig 4** | Inferred clonal composition per spot — 3 sections (SP1/SP2/SP3) × 4 clones, coloured by proportion |
| `plot_figure5.py` | **Fig 5** + **Fig S4** | ClonalGE vs Tumoroscope+LR comparison (main); per-run supplementary panel |
| `plot_figureS1_S2.py` | **Fig S1**, **Fig S2** | Within-run chain agreement heatmaps (best run; all 10 runs) |
| `plot_figureS3.py` | **Fig S3** | Across-run agreement after Hungarian clone alignment |

All scripts that use the real prostate data reconstruct the consensus posterior by running the two-stage chain/run selection pipeline (within-run Pearson filtering + cross-run Hungarian alignment) before plotting. Output figures are written to `plots_paper/`.

## Generating figures for a new dataset

Use `generate_all_figures.py` to produce Figures 3, 4, and 5 under all three run-selection approaches in one command.

### What you need to provide

| Input | Description |
|---|---|
| Run result directories | Paths of the form `<prefix>_1/`, `<prefix>_2/`, … each containing `inferred_vars_<section>_chain_0` … `_chain_9`, `spots_order.txt`, `genes_order.txt` |
| Spatial coordinate CSV | Columns: `section`, `barcode`, `x`, `y`, `nuclei`, `type` — same format as `prostate_data_configs/prostate_cell_count_annotation_any_fixed_margin.txt` |
| *(optional)* Tumoroscope H/N `.npy` | Required only for the Tumoroscope+LR comparison panel in Figure 5 |

### Command

```bash
python generate_all_figures.py \
    --result_prefix /path/to/Results_newdata \
    --num_runs      10 \
    --start_seed    1 \
    --section       all \
    --n_s_data      /path/to/spatial_coords.csv \
    --base_outdir   plots_paper_newdata
```

Add `--tum_h /path/H.npy --tum_n /path/N.npy` if Tumoroscope outputs are available.

### Output structure

```
plots_paper_newdata/
  approach_1_highest_loglik/    figure3.png  figure4.png  figure5.png  figure5_supp.png
  approach_2_consensus/         figure3.png  figure4.png  figure5.png  figure5_supp.png
  approach_3_consensus_hl/      figure3.png  figure4.png  figure5.png  figure5_supp.png
```

### Run-selection approaches

| # | Name | Method |
|---|---|---|
| 1 | Highest log-likelihood | Use the single run with the highest within-run log-likelihood |
| 2 | Consensus | Hungarian clone alignment; pick the reference run that maximises the number of agreeing runs (r > threshold) |
| 3 | Consensus anchored to HL | Hungarian alignment anchored to the highest-loglik run; include all runs that agree with it above the threshold |

To run only specific approaches or figures:
```bash
python generate_all_figures.py ... --approaches 2 3 --figures 3 4
```

[1] Shafighi, Shadi, et al. "Integrative spatial and genomic analysis of tumor heterogeneity with Tumoroscope." Nature Communications 15.1 (2024): 9343.  
[2] Berglund, Emelie, et al. "Spatial maps of prostate cancer transcriptomes reveal an unexplored landscape of heterogeneity." *Nature communications* 9.1 (2018): 1-13.
