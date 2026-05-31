# ClonalGE Prostate Data Analysis Journey

Analysis of spatial transcriptomics + somatic SNV data from the Berglund et al. (2018) prostate cancer dataset.
Three sections: **P1.2**, **P2.4**, **P3.3**. Four clones (K=4). 294 spots. 1000 most-variable genes.

---

## Step 1 — Figure 5 with 10 independent runs

**Runs used:** `Results_figure5_1` … `Results_figure5_10` (seeds 1–10, from `/Volumes/LenovoPS8/ClonalGE/`)
**Chains per run:** 10 (total 100 chains)
**MCMC settings:** Tumoroscope max\_iter=15 000, ClonalGE max\_iter=30 000, burn\_in=2 000, batch=1 000

### Chain / run selection (approach 2 — consensus)
- Within-run: all 10 runs kept 10/10 chains except runs 4, 10 (6/10 each)
- Cross-run consensus: ref run = 8; **6/10 runs selected** (runs 3, 4, 5, 6, 7, 8)
- B\_samples pooled: shape (1680, 4, 1000)
- Among selected — H: mean r=0.989, min=0.964 · B: mean r=0.957, min=0.876
- Clone permutations: runs 3–8 align as [0,1,2,3]; runs 1,9 need [3,1,2,0]; run 2 [3,2,1,0]

### Command
```bash
python plot_figure5.py \
  --result_prefix /Volumes/LenovoPS8/ClonalGE/Results_figure5 \
  --num_runs 10 --start_seed 1 --section all \
  --tum_h Results_figure5_1/all_tum_h.npy \
  --tum_n Results_figure5_1/all_tum_n.npy \
  --outdir plots_paper/prostate_10runs
```

### Results
| Figure | Path |
|--------|------|
| Within-run chain agreement | `plots_paper/prostate_10runs/chain_agreement.png` |
| Within-run chain agreement supplementary | `plots_paper/prostate_10runs/chain_agreement_supp.png` |
| **Approach 1** figure5 + supp + run_agreement | `plots_paper/prostate_10runs/approach_1/` |
| **Approach 2** figure5 + supp + run_agreement | `plots_paper/prostate_10runs/approach_2/` |
| **Approach 3** figure5 + supp + run_agreement | `plots_paper/prostate_10runs/approach_3/` |

### Figure 5 panels — by approach
| Approach | Selection | Runs selected | ClonalGE r |
|----------|-----------|---------------|-----------|
| 1 — highest loglik | single best-loglik run | 1/10 (run 2) | 0.921 |
| 2 — consensus | max agreement group | 6/10 (runs 3–8) | 0.898 |
| 3 — consensus anchored to HL | agree with best loglik | 6/10 (runs 3–8) | 0.921 |

Tumoroscope+LR: r=0.828 · Tumoroscope+NB: r=0.781 (identical across approaches — same baseline)

---

## Step 2 — Figure 5 with all 20 independent runs

**Runs used:** `Results_figure5_1` … `Results_figure5_20` (seeds 1–20, from `/Volumes/LenovoPS8/ClonalGE/`)
**Chains per run:** 10 (total 200 chains)
**MCMC settings:** same as Step 1

### Chain / run selection (approach 2 — consensus)
- Within-run: all 20 runs kept 10/10 chains except runs 4, 10, 19 (6/10 each)
- Cross-run consensus: ref run = 8; **8/20 runs selected** (runs 3, 4, 5, 6, 7, 8, 14, 16)
- B\_samples pooled: shape (2280, 4, 1000)
- Among selected — H: mean r=0.988, min=0.964 · B: mean r=0.954, min=0.876

### Command
```bash
python plot_figure5.py \
  --result_prefix /Volumes/LenovoPS8/ClonalGE/Results_figure5 \
  --num_runs 20 --start_seed 1 --section all \
  --tum_h Results_figure5_1/all_tum_h.npy \
  --tum_n Results_figure5_1/all_tum_n.npy \
  --outdir plots_paper/prostate_20runs
```

### Results
| Figure | Path |
|--------|------|
| Within-run chain agreement | `plots_paper/prostate_20runs/chain_agreement.png` |
| Within-run chain agreement supplementary | `plots_paper/prostate_20runs/chain_agreement_supp.png` |
| **Approach 1** figure5 + supp + run_agreement | `plots_paper/prostate_20runs/approach_1/` |
| **Approach 2** figure5 + supp + run_agreement | `plots_paper/prostate_20runs/approach_2/` |
| **Approach 3** figure5 + supp + run_agreement | `plots_paper/prostate_20runs/approach_3/` |

### Figure 5 panels — by approach
| Approach | Selection | Runs selected | ClonalGE r |
|----------|-----------|---------------|-----------|
| 1 — highest loglik | single best-loglik run | 1/20 (run 2) | 0.921 |
| 2 — consensus | max agreement group | 8/20 (runs 3–8, 14, 16) | 0.857 |
| 3 — consensus anchored to HL | agree with best loglik | 1/20 (run 2) | 0.921 |

Tumoroscope+LR: r=0.828 · Tumoroscope+NB: r=0.781 (identical across approaches — same baseline)

**Key finding:** ClonalGE (joint probabilistic model) outperforms both post-hoc regression baselines.
NB regression performs slightly worse than LR — the X→Y relationship is approximately linear,
not log-linear, so the NB log link is less natural here. ClonalGE benefits from jointly
inferring B within the full generative model.

Approaches 1 and 3 both converge to the single best run (run 2, loglik=−41885) and give r=0.921.
Approach 2 (consensus over 8 runs) yields r=0.857 — averaging over more runs smooths the estimate
but may dilute the best-performing solution.

---

## Per-run log-likelihoods (all 20 runs)

Best-loglik chain per run, from within-run selection (approach 2 / consensus shown).

| Run | Best loglik | Chains kept | Consensus (approach 2) |
|-----|-------------|-------------|------------------------|
|  1  | −42 144.05  | 10/10       | excluded               |
|  2  | **−41 885.18** | 10/10    | excluded ⚠️            |
|  3  | −42 663.97  | 10/10       | **selected**           |
|  4  | −42 887.68  | 6/10        | **selected**           |
|  5  | −42 827.50  | 10/10       | **selected**           |
|  6  | −42 793.94  | 10/10       | **selected**           |
|  7  | −42 664.11  | 10/10       | **selected**           |
|  8  | −42 659.08  | 10/10       | **selected (ref)**     |
|  9  | −42 035.15  | 10/10       | excluded               |
| 10  | −42 949.10  | 6/10        | excluded               |
| 11  | −42 021.81  | 10/10       | excluded               |
| 12  | −42 038.15  | 10/10       | excluded               |
| 13  | −42 337.61  | 10/10       | excluded               |
| 14  | −42 665.04  | 10/10       | **selected**           |
| 15  | −42 089.56  | 10/10       | excluded               |
| 16  | −43 079.63  | 10/10       | **selected**           |
| 17  | −42 062.65  | 10/10       | excluded               |
| 18  | −43 128.91  | 10/10       | excluded               |
| 19  | −42 846.75  | 6/10        | excluded               |
| 20  | −42 006.01  | 10/10       | excluded               |

**Notable:** Run 2 achieves the single best log-likelihood (−41 885) and is the reference for
approaches 1 and 3, yet is **excluded** from the approach-2 consensus group. After Hungarian
clone alignment, run 2 still fails the r > 0.9 threshold against the consensus cluster
(runs 3–8, 14, 16) — it converged to a different clone-labelling orientation. This is classic
label-switching in Bayesian mixture models: run 2 is not a worse solution, just a differently
labelled equivalent one. Approach 1/3 (highest log-likelihood) naturally selects it.

This explains why approaches 1 and 3 give r = 0.921 while approach 2 gives r = 0.857 for 20 runs —
the two groups represent equally valid but differently labelled posteriors.

---

## Data sources

| File | Description |
|------|-------------|
| `prostate_data_configs/vardict2_st_calling/vardict2_*.ac` | Allele counts per section (VarDict2 output) |
| `prostate_data_configs/STdata/P*.tsv` | Spatial transcriptomics count matrices |
| `prostate_data_configs/C_tree_1.txt` | Clone–SNV assignment matrix (phyloWGS/Canopy) |
| `prostate_data_configs/F_tree_1.txt` | Clone frequency priors |
| `prostate_data_configs/ssm_data.txt` | Somatic SNV data |
| `prostate_data_configs/prostate_cell_count_annotation_any_fixed_margin.txt` | Nuclei count per spot (n_lambda) |
| `most_var_genes1000.txt` | 1000 most-variable genes selected for expression model |

## Tumoroscope baseline (tum_h / tum_n)

Tumoroscope H and N were extracted from the ClonalGE chain initialisation (`chain.inits[1]`, `chain.inits[0]`)
of run 1 chain 0, and saved to:

- `Results_figure5_1/all_tum_h.npy` — shape (294, 4)
- `Results_figure5_1/all_tum_n.npy` — shape (294,)

These are used as the shared baseline for both Tumoroscope + LR and Tumoroscope + NB panels.

---

## Unified log-likelihood colorbar

All 6 `figure5.png` files (10runs × 3 approaches, 20runs × 3 approaches) share a single
log-likelihood colorbar range so that panels are directly comparable across figures:

- **vmin = −47.5963** (global minimum across all 6 configurations and all 3 panels)
- **vmax = 0**

The global vmin was pre-computed across every (num_runs, approach) combination by evaluating
NB log-PMF for all three predictions (Tumoroscope+LR, Tumoroscope+NB, ClonalGE). The NB panel
dominated with the lowest per-spot likelihoods, setting the floor.

Passed to the script via `--vmin -47.5963`. Colorbars now show **−47.6 to 0** in all figures.

---

## Code changes made

| File | Change |
|------|--------|
| `plot_figure5.py` | Added `calc_B_nb()` — NB GLM (log link + intercept) per gene using statsmodels |
| `plot_figure5.py` | Fixed `B_tum` shape bug: `H_tum.shape[0]` → `H_tum.shape[1]` (S→K) |
| `plot_figure5.py` | Extended panel letter labeling from `'ab'` to `'abcdef'` |
| `plot_figure5.py` | Added `--vmin` CLI argument for unified colorbar range across figures |
| `plot_figure5.py` | `scatter_panel()` accepts `vmin`/`vmax` for consistent colorbar |
| `main_real_1000.py` | Added save of Tumoroscope H/N as `<section>_tum_h.npy` / `<section>_tum_n.npy` |
| `plot_figureS3.py` | Added `--approach` argument; routes through `run_selection.select_runs()` |
