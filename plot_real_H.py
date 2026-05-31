"""
Plot the inferred clonal composition H for the prostate cancer dataset.

Shows one heatmap panel per tissue section (P1.2, P2.4, P3.3).
Each panel: rows = spots (sorted by dominant clone), columns = clones.
Colour = inferred proportion H[s,k].

Usage:
    python plot_real_H.py [options]

Options:
    --result_prefix   Directory prefix            (default: Results_figure5)
    --num_runs        Number of runs              (default: 10)
    --start_seed      Starting seed               (default: 1)
    --section         Config section name         (default: all)
    --threshold       Pearson r threshold         (default: 0.9)
    --spots_run       Run to load spots_order.txt (default: 1)
    --output          Output path                 (default: plots_paper/real_H_consensus.png)
    --sort            Sort spots by dominant clone within each section
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

# ── constants ──────────────────────────────────────────────────────────────────
CHAIN_COUNT = 10
SECTIONS    = ['P1.2', 'P2.4', 'P3.3']          # display order
SEC_LABELS  = ['Section P1.2', 'Section P2.4', 'Section P3.3']

# Clone colours — one per clone, used for the column header stripe
CLONE_COLS  = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

FS_BASE  = 7
FS_TITLE = 7
FS_TICK  = 6
FS_LABEL = 6.5


# ══════════════════════════════════════════════════════════════════════════════
# Consensus H computation  (same pipeline as plot_figure5.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_chains(prefix, run, section, n=CHAIN_COUNT):
    base = f'{prefix}_{run}/inferred_vars_{section}'
    return [pickle.load(open(base + f'_chain_{cc}', 'rb')) for cc in range(n)]


def select_within(chains, threshold):
    lls  = [c.last_loglik for c in chains]
    best = int(np.argmax(lls))
    ref  = chains[best].inferred_H.flatten()
    sel  = {i for i, c in enumerate(chains)
            if i == best or pearsonr(ref, c.inferred_H.flatten())[0] > threshold}
    return (np.mean([chains[i].inferred_H for i in sel], axis=0),
            np.mean([chains[i].inferred_B for i in sel], axis=0))


def _align(H_ref, H_other, B_other):
    K    = H_ref.shape[1]
    cost = np.array([[-pearsonr(H_ref[:, i], H_other[:, j])[0]
                       for j in range(K)] for i in range(K)])
    _, perm = linear_sum_assignment(cost)
    return H_other[:, perm], B_other[perm, :]


def consensus_H(prefix, num_runs, start_seed, section, threshold):
    H_runs, B_runs = [], []
    for run in range(start_seed, start_seed + num_runs):
        try:
            chains = load_chains(prefix, run, section)
        except FileNotFoundError:
            continue
        H, B = select_within(chains, threshold)
        H_runs.append(H)
        B_runs.append(B)

    n = len(H_runs)
    best_ref, best_count, best_mean_r = 0, -1, -1.0
    best_sel = best_H_al = None

    for ref in range(n):
        H_al = list(H_runs)
        B_al = list(B_runs)
        for i in range(n):
            if i != ref:
                H_al[i], B_al[i] = _align(H_runs[ref], H_runs[i], B_runs[i])
        ref_flat = H_al[ref].flatten()
        r_vals   = [pearsonr(ref_flat, H_al[i].flatten())[0] for i in range(n)]
        sel      = [i for i, r in enumerate(r_vals) if r >= threshold]
        mean_r   = float(np.mean([r_vals[i] for i in sel]))
        if len(sel) > best_count or (len(sel) == best_count and mean_r > best_mean_r):
            best_ref, best_count, best_mean_r = ref, len(sel), mean_r
            best_sel  = sel
            best_H_al = H_al

    H_final = np.mean([best_H_al[i] for i in best_sel], axis=0)
    print(f'Consensus run idx {best_ref}  ({best_count}/{n} runs selected)')
    return H_final


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

def draw_section(ax, H_sec, section_label, K, sort, first_col):
    """Draw one heatmap panel for a single section."""
    n_spots = H_sec.shape[0]

    if sort:
        order = np.lexsort((-H_sec[:, np.argmax(H_sec.mean(axis=0))],
                            H_sec.argmax(axis=1)))
        H_sec = H_sec[order]

    im = ax.imshow(H_sec, cmap='Blues', vmin=0, vmax=1,
                   aspect='auto', interpolation='none')

    # clone colour stripe at the bottom (x-tick replacement)
    for k in range(K):
        ax.add_patch(plt.Rectangle(
            (k - 0.5, n_spots - 0.5), 1, 0.04 * n_spots,
            facecolor=CLONE_COLS[k], edgecolor='none',
            transform=ax.transData, clip_on=False, zorder=5))

    ax.set_xticks(range(K))
    ax.set_xticklabels([f'Clone {k+1}' for k in range(K)],
                       fontsize=FS_TICK, rotation=30, ha='right')
    for k, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(CLONE_COLS[k])
        lbl.set_fontweight('bold')

    ax.set_yticks([0, n_spots - 1])
    ax.set_yticklabels([1, n_spots], fontsize=FS_TICK)
    if first_col:
        ax.set_ylabel('Spot', fontsize=FS_LABEL, labelpad=2)

    ax.set_title(f'{section_label}\n({n_spots} spots)',
                 fontsize=FS_TITLE, pad=4)
    ax.tick_params(length=2, width=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_prefix', default='Results_figure5')
    ap.add_argument('--num_runs',      type=int, default=10)
    ap.add_argument('--start_seed',    type=int, default=1)
    ap.add_argument('--section',       default='all')
    ap.add_argument('--threshold',     type=float, default=0.9)
    ap.add_argument('--spots_run',     type=int, default=1)
    ap.add_argument('--output',        default='plots_paper/real_H_consensus.png')
    ap.add_argument('--sort',          action='store_true', default=True)
    args = ap.parse_args()

    # ── compute consensus H ───────────────────────────────────────────────────
    print('Computing consensus H ...')
    H = consensus_H(args.result_prefix, args.num_runs, args.start_seed,
                    args.section, args.threshold)
    K = H.shape[1]
    print(f'H shape: {H.shape}  K={K}')
    print(f'Mean proportion per clone: {H.mean(axis=0).round(3)}')

    # ── load spots order ──────────────────────────────────────────────────────
    spots_file = f'{args.result_prefix}_{args.spots_run}/spots_order.txt'
    spots = pd.read_csv(spots_file, header=None)[0].tolist()
    spots_sec = pd.Series([s.split('_')[0] for s in spots])

    # ── split H by section ────────────────────────────────────────────────────
    sec_data = {}
    for sec in SECTIONS:
        mask = (spots_sec == sec).values
        sec_data[sec] = H[mask]
        print(f'  {sec}: {mask.sum()} spots')

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':   'sans-serif',
        'font.size':      FS_BASE,
        'axes.linewidth': 0.5,
        'pdf.fonttype':   42,
        'svg.fonttype':   'none',
    })

    # widths proportional to spot counts for each section
    counts  = [sec_data[s].shape[0] for s in SECTIONS]
    fig_w   = 5.6
    fig_h   = 3.2
    fig     = plt.figure(figsize=(fig_w, fig_h))
    gs      = gridspec.GridSpec(
        1, len(SECTIONS),
        width_ratios=counts,
        left=0.09, right=0.88, top=0.88, bottom=0.22,
        wspace=0.22)

    axes = [fig.add_subplot(gs[i]) for i in range(len(SECTIONS))]

    im = None
    for i, (sec, label) in enumerate(zip(SECTIONS, SEC_LABELS)):
        im = draw_section(axes[i], sec_data[sec], label, K,
                          sort=args.sort, first_col=(i == 0))

    # ── shared colourbar ──────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.905, 0.22, 0.018, 0.66])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Clone proportion', fontsize=FS_BASE, labelpad=4)
    cb.ax.tick_params(labelsize=FS_TICK, length=2, width=0.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(0.5)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=400, bbox_inches='tight')
    print(f'Saved → {args.output}')
    plt.close()


if __name__ == '__main__':
    main()
