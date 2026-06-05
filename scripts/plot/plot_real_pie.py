"""
Plot inferred clonal composition H as spatial pie charts for each tissue section.

One panel per tissue section (P1.2, P2.4, P3.3).  Each pie is placed at the
physical (x, y) spot coordinate; the wedge sizes are the consensus-inferred
clone proportions H[s,:].

Usage:
    python plot_real_pie.py [options]

Options:
    --result_prefix   Directory prefix           (default: Results_figure5)
    --num_runs        Number of runs             (default: 10)
    --start_seed      Starting seed              (default: 1)
    --section         Config section name        (default: all)
    --threshold       Pearson r threshold        (default: 0.9)
    --spots_run       Run to load spots_order    (default: 1)
    --n_s_data        Path to n_s_data CSV       (default: prostate_data_configs/...)
    --radius          Pie radius in data units   (default: 0.45)
    --output          Output path               (default: plots_paper/real_pie_consensus.png)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

# ── constants ──────────────────────────────────────────────────────────────────
CHAIN_COUNT = 10
SECTIONS    = ['P1.2', 'P2.4', 'P3.3']
SEC_LABELS  = ['Section P1.2', 'Section P2.4', 'Section P3.3']
CLONE_COLS  = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

FS_BASE  = 7
FS_TITLE = 8
FS_TICK  = 6

N_S_DATA_DEFAULT = ('prostate_data_configs/'
                    'prostate_cell_count_annotation_any_fixed_margin.txt')


# ══════════════════════════════════════════════════════════════════════════════
# Consensus H (same pipeline as plot_figure5.py / plot_real_H.py)
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
# Pie-chart drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_pie(ax, x, y, values, colors, radius):
    """Draw a single pie chart centred at (x, y) using wedge patches."""
    total = values.sum()
    if total == 0:
        fracs = np.ones(len(values)) / len(values)
    else:
        fracs = values / total
    start = 0.0
    for frac, color in zip(fracs, colors):
        angle = frac * 360.0
        wedge = mpatches.Wedge(
            center=(x, y), r=radius,
            theta1=start, theta2=start + angle,
            facecolor=color, edgecolor='white', linewidth=0.15)
        ax.add_patch(wedge)
        start += angle


def draw_section_pie(ax, H_sec, coords, section_label, K, radius):
    """Draw all pie charts for one section."""
    x_vals = coords['x'].values
    y_vals = coords['y'].values

    for s in range(H_sec.shape[0]):
        vals = H_sec[s]
        if vals.sum() == 0:
            vals = np.ones(K) / K
        draw_pie(ax, x_vals[s], y_vals[s], vals, CLONE_COLS[:K], radius)

    # axis cosmetics
    x_pad = radius * 1.2
    y_pad = radius * 1.2
    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)
    ax.set_aspect('equal')
    ax.set_title(f'{section_label}\n({H_sec.shape[0]} spots)',
                 fontsize=FS_TITLE, pad=4)
    ax.tick_params(labelsize=FS_TICK, length=2, width=0.4)
    ax.set_xlabel('x', fontsize=FS_TICK, labelpad=1)
    ax.set_ylabel('y', fontsize=FS_TICK, labelpad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


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
    ap.add_argument('--spots_run',     type=int, default=1)
    ap.add_argument('--n_s_data',      default=N_S_DATA_DEFAULT)
    ap.add_argument('--radius',        type=float, default=0.45)
    ap.add_argument('--output',        default='plots_paper/real_pie_consensus.png')
    args = ap.parse_args()

    # ── consensus H ───────────────────────────────────────────────────────────
    print('Computing consensus H ...')
    H = consensus_H(args.result_prefix, args.num_runs, args.start_seed,
                    args.section, args.threshold)
    K = H.shape[1]
    print(f'H shape: {H.shape}  K={K}')

    # ── load spot ordering and spatial coords ──────────────────────────────────
    spots_file = f'{args.result_prefix}_{args.spots_run}/spots_order.txt'
    spots = pd.read_csv(spots_file, header=None)[0].tolist()

    n_s_data = pd.read_csv(args.n_s_data, sep=',')
    n_s_data = n_s_data[~n_s_data.isin([float('nan'), float('inf'), float('-inf')]).any(axis=1)]
    n_s_data = n_s_data.drop_duplicates()
    n_s_data = n_s_data[n_s_data['type'].str.contains('Cancer')]
    n_s_data['x'] = n_s_data['x'].astype(float)
    n_s_data['y'] = n_s_data['y'].astype(float)

    # align n_s_data to the spots_order used during inference
    n_s_data = n_s_data[n_s_data['barcode'].isin(spots)].drop_duplicates('barcode')
    spot_order = pd.DataFrame({'barcode': spots})
    n_s_data = spot_order.merge(n_s_data, on='barcode', how='left').dropna(subset=['x', 'y'])

    # build H DataFrame aligned to spots
    H_df = pd.DataFrame(H, index=spots, columns=[f'C{k+1}' for k in range(K)])

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':   'sans-serif',
        'font.size':      FS_BASE,
        'axes.linewidth': 0.5,
        'pdf.fonttype':   42,
        'svg.fonttype':   'none',
    })

    fig, axes = plt.subplots(1, len(SECTIONS), figsize=(10, 3.8))

    for ax, sec, label in zip(axes, SECTIONS, SEC_LABELS):
        mask    = n_s_data['section'] == sec
        coords  = n_s_data[mask].reset_index(drop=True)
        barcodes = coords['barcode'].tolist()
        H_sec   = H_df.loc[barcodes].values
        draw_section_pie(ax, H_sec, coords, label, K, args.radius)

    # ── shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=CLONE_COLS[k], edgecolor='grey',
                       linewidth=0.3, label=f'Clone {k+1}')
        for k in range(K)
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=K, fontsize=FS_BASE, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Inferred clonal composition per spot', fontsize=FS_TITLE + 1,
                 y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=400, bbox_inches='tight')
    print(f'Saved → {args.output}')
    plt.close()


if __name__ == '__main__':
    main()
