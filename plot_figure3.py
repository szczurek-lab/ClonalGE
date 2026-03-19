"""
Figure 3 — Genes are expressed differently in various cancer clones.

Panel a: Normalised expression (min-max scaled across clones) of the top 30
         genes ranked by variance of B_kg across clones.  Rows and columns
         are clustered hierarchically.

Panel b: Posterior probability of differential gene expression between every
         pair of clones, P(B_k1,g > B_k2,g | Data), computed from pooled
         MCMC samples across consensus-selected chains and runs.

Usage:
    python plot_figure3.py [options]

Options:
    --result_prefix   Directory prefix     (default: Results_figure5)
    --num_runs        Number of runs       (default: 10)
    --start_seed      Starting seed        (default: 1)
    --section         Section name         (default: all)
    --threshold       Pearson r threshold  (default: 0.9)
    --top_genes       Genes to show        (default: 30)
    --output          Output path          (default: plots_paper/figure3.png)
"""

import argparse
import os
import pickle

import run_selection as rs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

# ── constants ──────────────────────────────────────────────────────────────────
CHAIN_COUNT = 10

CMAP_A   = 'magma'         # black → purple → orange → yellow
CMAP_B   = 'Blues'         # white → dark blue

FS_TITLE  = 7
FS_LABEL  = 7
FS_TICK   = 7
FS_PANEL  = 7


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_chain(prefix, run, section, cc):
    path = f'{prefix}_{run}/inferred_vars_{section}_chain_{cc}'
    return pickle.load(open(path, 'rb'))


def load_genes(prefix, run):
    path = f'{prefix}_{run}/genes_order.txt'
    lines = open(path).read().strip().split('\n')
    return [l.split()[0] for l in lines]   # first token = gene symbol


# ══════════════════════════════════════════════════════════════════════════════
# Consensus pipeline — returns posterior mean B AND pooled MCMC B samples
# ══════════════════════════════════════════════════════════════════════════════

def _align_perm(H_ref, H_other):
    """Return permutation of clones in H_other that best matches H_ref."""
    K    = H_ref.shape[1]
    cost = np.array([[-pearsonr(H_ref[:, i], H_other[:, j])[0]
                       for j in range(K)] for i in range(K)])
    _, perm = linear_sum_assignment(cost)
    return perm


def select_within_run(prefix, run, section, threshold):
    """Within-run chain selection. Returns (H_mean, B_mean, B_samples, perm_identity)."""
    chains = [load_chain(prefix, run, section, cc) for cc in range(CHAIN_COUNT)]
    lls    = [c.last_loglik for c in chains]
    best   = int(np.argmax(lls))
    ref_H  = chains[best].inferred_H.flatten()
    sel    = [i for i, c in enumerate(chains)
              if i == best or pearsonr(ref_H, c.inferred_H.flatten())[0] > threshold]
    H_mean = np.mean([chains[i].inferred_H for i in sel], axis=0)
    B_mean = np.mean([chains[i].inferred_B for i in sel], axis=0)
    # pool raw MCMC B samples from selected chains  (n_samples, K, g)
    B_samp = np.concatenate([chains[i].B_sum for i in sel], axis=0)
    return H_mean, B_mean, B_samp


def consensus_B(prefix, num_runs, start_seed, section, threshold):
    """
    Full two-stage consensus pipeline.
    Returns:
        B_final  : (K, g) posterior mean B
        B_samples: (N, K, g) pooled MCMC samples from consensus runs (clone-aligned)
    """
    H_runs, B_runs, Bsamp_runs = [], [], []
    for run in range(start_seed, start_seed + num_runs):
        try:
            H, B, Bs = select_within_run(prefix, run, section, threshold)
        except FileNotFoundError:
            continue
        H_runs.append(H)
        B_runs.append(B)
        Bsamp_runs.append(Bs)

    n = len(H_runs)
    best_ref, best_count, best_mean_r = 0, -1, -1.0
    best_sel = None
    best_H_al = best_perms = None

    for ref in range(n):
        perms  = [None] * n
        H_al   = list(H_runs)
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
            best_H_al  = H_al

    print(f'Consensus run idx {best_ref}  ({best_count}/{n} runs selected)')

    # align B and B_samples with the winning permutations
    B_aligned    = [B_runs[i][best_perms[i], :] for i in best_sel]
    Bsamp_aligned = [Bsamp_runs[i][:, best_perms[i], :] for i in best_sel]

    B_final   = np.mean(B_aligned, axis=0)          # (K, g)
    B_samples = np.concatenate(Bsamp_aligned, axis=0)  # (N_total, K, g)
    return B_final, B_samples


# ══════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def top_genes_by_variance(B, gene_names, n=30):
    """Select top-n genes by variance of B_kg across clones."""
    var_g   = np.var(B, axis=0)          # (g,)
    idx     = np.argsort(var_g)[::-1][:n]
    return idx, [gene_names[i] for i in idx]


def minmax_scale_genes(B_sub):
    """Min-max scale each gene (row) across clones (columns). Shape: (n_genes, K)."""
    B_T   = B_sub.T                       # (n_genes, K)
    lo    = B_T.min(axis=1, keepdims=True)
    hi    = B_T.max(axis=1, keepdims=True)
    denom = np.where(hi - lo == 0, 1, hi - lo)
    return (B_T - lo) / denom


def cluster_order(mat):
    """Return row and column reordering from hierarchical clustering."""
    row_link = linkage(mat,             method='average', metric='euclidean')
    col_link = linkage(mat.T,           method='average', metric='euclidean')
    return leaves_list(row_link), leaves_list(col_link)


def posterior_diff_prob(B_samples, gene_idx, K):
    """
    Compute P(B_k1,g > B_k2,g | Data) for all clone pairs and selected genes.

    Returns dict {(k1,k2): array of shape (n_genes,)} for k1 < k2.
    """
    B_sub = B_samples[:, :, :][:, :, gene_idx]   # (N, K, n_genes)
    pairs = {}
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            prob = np.mean(B_sub[:, k1, :] > B_sub[:, k2, :], axis=0)
            pairs[(k1, k2)] = prob
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_prefix', default='Results_figure5')
    ap.add_argument('--num_runs',      type=int, default=10)
    ap.add_argument('--start_seed',    type=int, default=1)
    ap.add_argument('--section',       default='all')
    ap.add_argument('--threshold',     type=float, default=0.9)
    ap.add_argument('--top_genes',     type=int, default=30)
    ap.add_argument('--approach',      type=int, default=2, choices=[1, 2, 3],
                    help='1=highest loglik  2=consensus  3=consensus anchored to HL')
    ap.add_argument('--outdir',        default='plots_paper',
                    help='Output directory (figure3.png placed inside)')
    args = ap.parse_args()

    output = os.path.join(args.outdir, 'figure3.png')

    # ── run selection ─────────────────────────────────────────────────────────
    res        = rs.select_runs(args.approach, args.result_prefix, args.num_runs,
                                args.start_seed, args.section, args.threshold)
    B_final    = res['B_final']
    B_samples  = res['B_samples']
    K          = B_final.shape[0]
    gene_names = load_genes(args.result_prefix, args.start_seed)
    print(f'B shape: {B_final.shape}  MCMC samples: {B_samples.shape[0]}  genes: {len(gene_names)}')

    # ── select top genes ──────────────────────────────────────────────────────
    gene_idx, top_names = top_genes_by_variance(B_final, gene_names, args.top_genes)

    # panel a: clones 2,3,4 only (drop Clone 1 = index 0)
    clone_keep   = [k for k in range(K) if k != 0]
    clone_labels = [f'Clone {k+1}' for k in clone_keep]

    # panel b: pairs that do NOT involve Clone 1 (index 0)
    pair_keep   = [(k1, k2) for k1 in range(K) for k2 in range(k1+1, K)
                   if k1 != 0 and k2 != 0]
    pair_labels = [f'C{k1+1} vs C{k2+1}' for k1, k2 in pair_keep]
    n_pairs     = len(pair_labels)

    # ── panel a: min-max scaled B, cluster genes (cols) ───────────────────────
    B_sub    = B_final[clone_keep, :][:, gene_idx]   # (n_clones, n_genes)
    B_scaled = minmax_scale_genes(B_sub)              # (n_genes, n_clones)
    # cluster only genes (rows of B_scaled = cols of heatmap)
    gene_link = linkage(B_scaled, method='average', metric='euclidean')
    gene_ord  = leaves_list(gene_link)
    gene_names_ord = [top_names[g] for g in gene_ord]

    # heatmap: rows=clones, cols=genes  → B_scaled.T reordered
    B_plot = B_scaled[gene_ord, :].T     # (n_clones, n_genes)

    # ── panel b: differential expression probabilities ────────────────────────
    diff_probs = posterior_diff_prob(B_samples, gene_idx, K)
    # matrix: rows=pairs, cols=genes (same gene order as panel a)
    prob_mat = np.zeros((n_pairs, args.top_genes))
    for j, (k1, k2) in enumerate(pair_keep):
        prob_mat[j, :] = diff_probs[(k1, k2)][gene_ord]

    # ── figure layout ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':   'sans-serif',
        'font.size':      FS_TICK,
        'axes.linewidth': 0.5,
        'pdf.fonttype':   42,
        'svg.fonttype':   'none',
    })

    n_clones_show = len(clone_keep)
    n_genes       = args.top_genes
    DENDRO_H      = 0.35  # inches for dendrogram row
    CELL_H        = 0.18  # inches per heatmap row
    LABEL_H       = 0.45  # inches for gene name labels between panels
    fig_h = DENDRO_H + n_clones_show * CELL_H + LABEL_H + n_pairs * CELL_H + 0.3
    fig_w = 10.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # manually placed axes (all in figure-fraction coordinates)
    left, right = 0.06, 0.87
    top_y   = 1.0 - 0.25 / fig_h          # small top margin
    bot_y   = 0.02

    total_h   = top_y - bot_y
    dendro_fh = DENDRO_H / fig_h
    label_fh  = LABEL_H  / fig_h
    a_fh      = n_clones_show * CELL_H / fig_h
    b_fh      = n_pairs       * CELL_H / fig_h

    # y positions (bottom of each axes, counting from top)
    y_dendro = top_y - dendro_fh
    y_a      = y_dendro - a_fh
    y_label  = y_a - label_fh          # invisible axes just for gene labels
    y_b      = y_label - b_fh

    w = right - left

    ax_dendro = fig.add_axes([left, y_dendro, w, dendro_fh])
    ax_a      = fig.add_axes([left, y_a,      w, a_fh])
    ax_lbl    = fig.add_axes([left, y_label,  w, label_fh])   # label spacer
    ax_b      = fig.add_axes([left, y_b,      w, b_fh])

    # colorbars
    cax_a = fig.add_axes([0.895, y_a + a_fh * 0.1,      0.018, a_fh * 0.8])
    cax_b = fig.add_axes([0.895, y_b + b_fh * 0.1,      0.018, b_fh * 0.8])

    # ── dendrogram ────────────────────────────────────────────────────────────
    dend = dendrogram(gene_link, ax=ax_dendro, no_labels=True, color_threshold=0,
                      above_threshold_color='#444444',
                      link_color_func=lambda _: '#444444')
    # align dendrogram x-axis with heatmap columns:
    # scipy places leaf i at 10*i+5; heatmap col i is at x=i (xlim -0.5..n-0.5)
    ax_dendro.set_xlim(-0.5 * 10 + 5, (n_genes - 0.5) * 10 + 5)
    ax_dendro.set_ylim(bottom=0)
    ax_dendro.axis('off')
    ax_dendro.set_title('a', fontsize=FS_PANEL, fontweight='bold',
                        loc='left', pad=2, x=-0.005)

    # — panel a heatmap ────────────────────────────────────────────────────────
    im_a = ax_a.imshow(B_plot, aspect='auto', cmap=CMAP_A,
                       vmin=0, vmax=1, interpolation='nearest')
    ax_a.set_yticks(range(n_clones_show))
    ax_a.set_yticklabels(clone_labels, fontsize=FS_TICK)
    ax_a.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False)
    for sp in ax_a.spines.values():
        sp.set_linewidth(0.4)
    cb_a = fig.colorbar(im_a, cax=cax_a)
    cb_a.set_label('Normalised\nexpression', fontsize=FS_LABEL, labelpad=3)
    cb_a.ax.tick_params(labelsize=FS_TICK - 0.5, length=2)
    cb_a.outline.set_linewidth(0.4)

    # ── gene name labels — drawn in the spacer axes ───────────────────────────
    ax_lbl.set_xlim(-0.5, n_genes - 0.5)
    ax_lbl.set_ylim(0, 1)
    ax_lbl.axis('off')
    for i, name in enumerate(gene_names_ord):
        ax_lbl.text(i, 1.0, name, ha='center', va='top',
                    fontsize=FS_TICK - 0.5, rotation=90)

    # — panel b heatmap ────────────────────────────────────────────────────────
    im_b = ax_b.imshow(prob_mat, aspect='auto', cmap=CMAP_B,
                       vmin=0, vmax=1, interpolation='nearest')
    ax_b.set_xticks([])
    ax_b.set_yticks(range(n_pairs))
    ax_b.set_yticklabels(pair_labels, fontsize=FS_TICK)
    ax_b.tick_params(bottom=False, labelbottom=False)
    ax_b.set_title('b', fontsize=FS_PANEL, fontweight='bold',
                   loc='left', pad=2, x=-0.005)
    for sp in ax_b.spines.values():
        sp.set_linewidth(0.4)
    cb_b = fig.colorbar(im_b, cax=cax_b)
    cb_b.set_label('P(B$_{k_1}$ > B$_{k_2}$\n| Data)', fontsize=FS_LABEL, labelpad=3)
    cb_b.ax.tick_params(labelsize=FS_TICK - 0.5, length=2)
    cb_b.outline.set_linewidth(0.4)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    plt.savefig(output, dpi=400, bbox_inches='tight')
    print(f'Saved → {output}')
    plt.close()


if __name__ == '__main__':
    main()
