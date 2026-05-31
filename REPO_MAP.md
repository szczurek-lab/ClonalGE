# Repository Map

This document separates the repository into clear categories: manuscript figure
code, figure assets, benchmarking/comparison work, revision analyses, exploratory
material, and files that must not be pushed.

Importable Python modules are intentionally kept in the repository root: the
scripts use a flat import namespace (e.g. `import run_selection`, `import
clonalGE`), so relocating them into sub-folders would break imports. Separation
is therefore documented here rather than enforced by moving code.

## Manuscript figure map

The manuscript (`oup-authoring-template.tex`) is compiled from the Overleaf
folder `ClonalGE_revision/`, so its `figures/` paths resolve inside that folder.

| Figure | tex label | Image file (Overleaf-relative) | Generating script |
|---|---|---|---|
| Fig 1 | `fig:overview` | `figures/overview.png` | manual diagram |
| Fig 2 | `fig:perfsim` | `Figures-main/Fig2_simulation.pdf` | `generate_fig2_latex.py`, `plot_simulated_results.py` |
| Fig 3 | `fig:B_selected` | `Figures-main/figure3.png` | `plot_figure3.py` |
| Fig 4 | `prostate` | `figures/real_H.eps` | `plot_figure4.py` |
| Fig 5 | `comparison` | `Figures-main/figure5_20runs_nb_fixed.png` | `plot_figure5.py`, `plot_nb_comparison.py` |
| Fig 6 | `fig:stdeconvolve_comparison` | `figures/stdeconvolve_comparison_figure.pdf` | `plot_stdeconvolve_comparison.py` |
| Fig S1 | `fig:chain_agreement` | `figures/chain_agreement.png` | `plot_figureS1_S2.py` |
| Fig S2a | `fig:chain_agreement_supp` | `Figures-sup/chain_agreement_supp_runs1-10.pdf` | `plot_figureS1_S2.py` |
| Fig S2b | `fig:chain_agreement_supp` | `Figures-sup/chain_agreement_supp_runs11-20.pdf` | `plot_figureS1_S2.py` |
| Fig S3 | `fig:run_agreement` | `Figures-sup/run_agreement.png` | `plot_figureS3.py` |
| Fig S4 | `fig:figure5_supp` | `figures/figure5_supp.png` | `plot_figure5.py` |
| Fig S5 | `fig:runtime` | `figures/figure_runtime.png` | `plot_runtime_analysis.py` |
| Fig S6 | `fig:sensitivity` | `Figures-sup/sensitivity_analysis.png` | `plot_sensitivity.py` |
| Fig S7 | `fig:gsea` | `Figures-sup/gsea_results.png` | `run_gsea.py` |
| Fig S8 | `fig:violated` | `figures/violated_simulations.png` | `plot_violated_results.py` |

Generated outputs are also mirrored in `plots_paper/`.

## Categories

### Core model code (root)
`clonalGE.py`, `tumoroscope.py`, `simulation.py`, `simulation_tumoroscope.py`,
`visualization.py`, `pre_processing.py`, `constants.py`

### Main experiment drivers (root)
`main_diff_config.py`, `main_real_1000.py`, `generate_simulations.py`,
`run_main_diff_config.sh`, `run_main_real_1000.sh`, `generate_simulations.sh`,
`run_figure2_simulations.sh`, `run_figure5.sh`, `run_clonalge_5chains.sh`

### Manuscript figure-generation code (root)
`plot_figure3.py`, `plot_figure4.py`, `plot_figure5.py`, `plot_figureS1_S2.py`,
`plot_figureS3.py`, `plot_simulated_results.py`, `generate_fig2_latex.py`,
`generate_all_figures.py`, `run_selection.py`

### Figure assets
`plots_paper/` (generated outputs). Publication-ready copies live in the
Overleaf folder (`ClonalGE_revision/Figures-main`, `Figures-sup`, `figures`),
which is not pushed.

### Benchmarking / comparison with other models
- STdeconvolve baseline: `export_sim_for_stdeconvolve.py`, `run_stdeconvolve.R`,
  `evaluate_stdeconvolve.py`, `plot_stdeconvolve_comparison.py`,
  `stdeconvolve_comparison_latex.tex`, `stdeconvolve_comparison_figure.{png,pdf}`
- Tumoroscope + NB: `compute_tumoroscope_nb.py`, `plot_nb_comparison.py`
- Tumoroscope + LR: `compute_tumoroscope_lr.py`
- Tumoroscope runs: `run_tumoroscope_simulated.py`, `run_tumoroscope_simulated.sh`,
  `run_tumoroscope_simulated_slurm.sh`, `configs_tumoroscope/`,
  `tumoroscope_b_estimation.md`

The archival data/results layout for these is produced by
`build_zenodo_deposit.sh` (`01_input_data / 02_clonalge_results /
03_baselines / 04_downstream_analyses / 05_figures`).

### Revision analyses (supplementary figures)
- Sensitivity (Fig S6): `run_sensitivity.py`, `run_sensitivity_one.py`,
  `plot_sensitivity.py`, `run_sensitivity_slurm.sh`, `run_sensitivity_cluster.sh`
- Assumption-violating sims (Fig S8): `generate_simulations_violated.py`,
  `main_violated.py`, `plot_violated_results.py`, `run_violated_simulations.sh`,
  `run_violated_slurm.sh`, `run_violated_cluster.sh`
- Runtime / scalability (Fig S5): `run_scalability.py`, `plot_runtime_analysis.py`,
  `plot_runtime.py`, `runtime_section.tex`
- GSEA (Fig S7): `run_gsea.py`

### Exploratory (not in final paper)
`plots_paper.ipynb`, `plot_real_pie.py`, `plot_real_H.py`,
`plot_prostate_run_selection.py`, `extract_run_logliks.py`, `extract_best_see.py`,
`select_chains.py`, and `notes_slides/` (slide decks and development notes).

### Must NOT be pushed (gitignored)
`ClonalGE_revision/` (Overleaf), `ClonalGE revision_final.docx`,
`ClonalGE_Revision_Response.docx`, `ClonalGE_Task_Division.docx`,
`ClonalGE_revision.zip`, `*.pptx`.

### Flagged for review (left in place, purpose unclear)
- `prostate_data_configs1/` — possible duplicate of `prostate_data_configs/`
- `oup-authoring-template-1.tex` — possible older manuscript copy
- `Rplots.pdf` — default R output (likely from STdeconvolve)
- `run_logliks.csv`, `run_H_averaged.npy`, `run_H_averaged_ids.txt` — intermediates
