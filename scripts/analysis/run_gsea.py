"""
GSEA for clone-specific gene expression using ClonalGE posterior effect sizes.

For each clone k, genes are ranked by the posterior mean effect size relative
to all other non-normal clones:

    score_k(g) = mean_{k' != k} E[B_{k,g} - B_{k',g} | Data]

This Bayesian ranking score captures both the direction and the magnitude of
differential expression, making it more informative for GSEA than a
directional probability. Positive scores → gene up in clone k; negative → down.

Pre-ranked GSEA is run with gseapy using MSigDB Hallmark gene sets (H),
KEGG pathways (C2:CP:KEGG), and GO Biological Process (C5:GO:BP).

Outputs (per clone):
    <outdir>/gsea_clone{k}_{gene_set}/
        gseapy preranked results (including NES, p-val, q-val)
    <outdir>/gsea_summary.csv      — top hits per clone across all gene sets
    plots_paper/gsea_results.png   — summary figure

Usage:
    python run_gsea.py \\
        --result_prefix  Results_figure5 \\
        --num_runs       10 \\
        --section        all \\
        --threshold      0.9 \\
        --normal_clone   0 \\
        --gene_sets      h.all c2.cp.kegg c5.go.bp \\
        --outdir         /Volumes/LenovoPS8/ClonalGE/gsea_results \\
        --output         plots_paper/gsea_results.png

Requirements:
    pip install gseapy
    MSigDB gene-set files downloaded from https://www.gsea-msigdb.org/gsea/msigdb/
    or let gseapy fetch them automatically (requires internet access).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clonalge import run_selection as rs

try:
    import gseapy as gp
    GSEAPY_OK = True
except ImportError:
    GSEAPY_OK = False
    print('WARNING: gseapy not installed. Run: pip install gseapy')

CHAIN_COUNT = 10

# ── data loading (mirrors plot_figure3.py) ────────────────────────────────────

def _align_perm(H_ref, H_other):
    K    = H_ref.shape[1]
    cost = np.array([[-pearsonr(H_ref[:, i], H_other[:, j])[0]
                       for j in range(K)] for i in range(K)])
    _, perm = linear_sum_assignment(cost)
    return perm


def load_chain(prefix, run, section, cc):
    path = f'{prefix}_{run}/inferred_vars_{section}_chain_{cc}'
    return pickle.load(open(path, 'rb'))


def load_genes(prefix, run):
    path = f'{prefix}_{run}/genes_order.txt'
    lines = open(path).read().strip().split('\n')
    return [l.split()[0] for l in lines]


def _estimate_batch_n(chain):
    if hasattr(chain, 'batch_n') and chain.batch_n > 1:
        return int(chain.batch_n)
    nz       = np.where(chain.B_sum.sum(axis=(1, 2)) > 0)[0]
    if len(nz) == 0:
        return 1
    ref_mean = float(chain.inferred_B.mean())
    nz_mean  = float(chain.B_sum[nz].mean())
    if ref_mean > 0 and nz_mean > 0:
        return max(1, int(round(nz_mean / ref_mean)))
    return 1


def select_within_run(prefix, run, section, threshold):
    chains  = [load_chain(prefix, run, section, cc) for cc in range(CHAIN_COUNT)]
    lls     = [c.last_loglik for c in chains]
    best    = int(np.argmax(lls))
    ref_H   = chains[best].inferred_H.flatten()
    sel     = [i for i, c in enumerate(chains)
               if i == best or pearsonr(ref_H, c.inferred_H.flatten())[0] > threshold]
    H_mean  = np.mean([chains[i].inferred_H for i in sel], axis=0)
    B_mean  = np.mean([chains[i].inferred_B for i in sel], axis=0)
    batch_n = _estimate_batch_n(chains[best])

    def _nonzero_batch_means(chain):
        bm   = chain.B_sum / batch_n
        mask = bm.sum(axis=(1, 2)) > 0
        return bm[mask]

    B_samp  = np.concatenate([_nonzero_batch_means(chains[i]) for i in sel], axis=0)
    return H_mean, B_mean, B_samp


def load_consensus_B(prefix, num_runs, start_seed, section, threshold):
    H_runs, B_runs, Bsamp_runs = [], [], []
    for run in range(start_seed, start_seed + num_runs):
        try:
            H, B, Bs = select_within_run(prefix, run, section, threshold)
        except FileNotFoundError:
            continue
        H_runs.append(H); B_runs.append(B); Bsamp_runs.append(Bs)

    n = len(H_runs)
    best_ref, best_count, best_mean_r = 0, -1, -1.0
    best_sel = best_perms = None

    for ref in range(n):
        perms = [None] * n
        H_al  = list(H_runs)
        for i in range(n):
            if i != ref:
                p       = _align_perm(H_runs[ref], H_runs[i])
                H_al[i] = H_runs[i][:, p]
                perms[i] = p
            else:
                perms[i] = np.arange(H_runs[ref].shape[1])
        ref_flat = H_al[ref].flatten()
        r_vals   = [pearsonr(ref_flat, H_al[i].flatten())[0] for i in range(n)]
        sel      = [i for i, r in enumerate(r_vals) if r >= threshold]
        mean_r   = float(np.mean([r_vals[i] for i in sel]))
        if len(sel) > best_count or (len(sel) == best_count and mean_r > best_mean_r):
            best_ref, best_count, best_mean_r = ref, len(sel), mean_r
            best_sel   = sel
            best_perms = perms

    B_aligned    = [B_runs[i][best_perms[i], :]             for i in best_sel]
    Bsamp_aligned = [Bsamp_runs[i][:, best_perms[i], :]    for i in best_sel]
    B_final   = np.mean(B_aligned, axis=0)
    B_samples = np.concatenate(Bsamp_aligned, axis=0)
    print(f'Consensus: {best_count}/{n} runs | B_samples shape {B_samples.shape}')
    return B_final, B_samples


# ── ranking ───────────────────────────────────────────────────────────────────

def compute_clone_scores(B_samples, gene_names, normal_clone):
    """
    For each clone k (excluding normal_clone), rank genes by posterior mean
    effect size relative to all other non-normal clones:

        score_k(g) = mean_{k' != k} E[B_{k,g} - B_{k',g} | Data]

    This captures both direction and magnitude of differential expression.
    Returns dict {clone_idx: pd.Series(index=gene_names, values=scores)}
    """
    N, K, G = B_samples.shape
    B_mean  = B_samples.mean(axis=0)   # (K, G)  — posterior mean per clone/gene
    scores  = {}
    for k in range(K):
        if k == normal_clone:
            continue
        other = [k2 for k2 in range(K) if k2 != k and k2 != normal_clone]
        if not other:
            other = [k2 for k2 in range(K) if k2 != k]
        mean_others = B_mean[other, :].mean(axis=0)   # (G,)
        score_g     = B_mean[k, :] - mean_others       # (G,)
        scores[k]   = pd.Series(score_g, index=gene_names).sort_values(ascending=False)
    return scores


# ── GSEA ─────────────────────────────────────────────────────────────────────

MSIGDB_NAMES = {
    'MSigDB_Hallmark_2020': 'MSigDB Hallmark',
    'KEGG_2021_Human':      'KEGG',
    'GO_Biological_Process_2023': 'GO BP',
}


def run_preranked_gsea(scores_series, gene_set_name, outdir, clone_label,
                       min_size=10, max_size=500, permutation_num=1000):
    """Run gseapy prerank for one clone and one gene set collection."""
    if not GSEAPY_OK:
        return None
    rnk = scores_series.reset_index()
    rnk.columns = ['gene', 'score']
    out_sub = os.path.join(outdir, f'gsea_{clone_label}_{gene_set_name}')
    os.makedirs(out_sub, exist_ok=True)
    try:
        res = gp.prerank(
            rnk=rnk, gene_sets=gene_set_name,
            outdir=out_sub, min_size=min_size, max_size=max_size,
            permutation_num=permutation_num, verbose=False,
            seed=42, no_plot=True)
        return res
    except Exception as e:
        print(f'  GSEA error ({clone_label}, {gene_set_name}): {e}')
        return None


# ── summary figure ────────────────────────────────────────────────────────────

def plot_gsea_summary(summary_df, output):
    """Bar chart of top NES per clone, coloured by gene set."""
    clones = sorted(summary_df['clone'].unique())
    K = len(clones)
    fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 5), sharey=False)
    if K == 1:
        axes = [axes]

    cmap = plt.get_cmap('tab10')
    gs_colors = {gs: cmap(i) for i, gs in
                 enumerate(summary_df['gene_set'].unique())}

    for ax, clone in zip(axes, clones):
        sub = (summary_df[summary_df['clone'] == clone]
               .sort_values('NES', ascending=False)
               .head(15))
        colors = [gs_colors.get(gs, 'grey') for gs in sub['gene_set']]
        ax.barh(sub['Term'], sub['NES'], color=colors, edgecolor='white',
                height=0.7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f'Clone {clone + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('NES', fontsize=10)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)

    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=MSIGDB_NAMES.get(gs, gs))
               for gs, c in gs_colors.items()]
    fig.legend(handles=handles, loc='lower center', ncol=len(gs_colors),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight', dpi=300)
    print(f'Saved: {output}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_prefix', default='Results_figure5')
    ap.add_argument('--num_runs',      type=int, default=10)
    ap.add_argument('--start_seed',    type=int, default=1)
    ap.add_argument('--section',       default='all')
    ap.add_argument('--threshold',     type=float, default=0.9)
    ap.add_argument('--normal_clone',  type=int, default=0)
    ap.add_argument('--gene_sets',     nargs='+',
                    default=['MSigDB_Hallmark_2020', 'KEGG_2021_Human',
                             'GO_Biological_Process_2023'])
    ap.add_argument('--outdir',        default='gsea_results')
    ap.add_argument('--output',        default='plots_paper/gsea_results.png')
    ap.add_argument('--permutations',  type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('Loading consensus B samples ...')
    B_final, B_samples = load_consensus_B(
        args.result_prefix, args.num_runs, args.start_seed,
        args.section, args.threshold)
    gene_names = load_genes(args.result_prefix, args.start_seed)

    K = B_final.shape[0]
    print(f'K={K}  G={len(gene_names)}  N_samples={B_samples.shape[0]}')

    print('Computing clone-specific ranking scores ...')
    scores = compute_clone_scores(B_samples, gene_names, args.normal_clone)

    # save ranked gene lists
    for k, s in scores.items():
        rnk_path = os.path.join(args.outdir, f'ranked_genes_clone{k+1}.csv')
        s.reset_index().to_csv(rnk_path, index=False,
                               header=['gene', 'posterior_effect_size'])
        print(f'  Clone {k+1}: top genes = {list(s.index[:10])}')

    summary_rows = []
    for k, s in scores.items():
        for gs in args.gene_sets:
            print(f'GSEA: clone {k+1} / {gs} ...')
            res = run_preranked_gsea(s, gs, args.outdir,
                                     clone_label=f'clone{k+1}',
                                     permutation_num=args.permutations)
            if res is None:
                continue
            try:
                df = res.res2d if hasattr(res, 'res2d') else res.results
                df = df[df['FDR q-val'].astype(float) < 0.25].copy()
                df['clone']    = k
                df['gene_set'] = gs
                summary_rows.append(df[['Term', 'NES', 'NOM p-val',
                                         'FDR q-val', 'clone', 'gene_set']])
            except Exception as e:
                print(f'  Could not parse results: {e}')

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        summary_path = os.path.join(args.outdir, 'gsea_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f'Summary saved: {summary_path}')
        plot_gsea_summary(summary, args.output)
    else:
        print('No significant GSEA results (FDR < 0.25) found.')


if __name__ == '__main__':
    main()
