"""
Figure 4 — Inferred clonal composition per spot across tissue sections.

Layout: 3 rows (sections SP1/SP2/SP3) × 4 columns (clones).
Each panel shows spots as filled circles whose colour intensity encodes
the inferred proportion of that clone at that spot (white = 0, full colour = 1).

Usage:
    python plot_real_clone_maps.py [options]

Options:
    --result_prefix   Directory prefix           (default: Results_figure5)
    --num_runs        Number of runs             (default: 10)
    --start_seed      Starting seed              (default: 1)
    --section         Config section name        (default: all)
    --threshold       Pearson r threshold        (default: 0.9)
    --n_s_data        Path to n_s_data CSV
    --radius          Spot radius in data units  (default: 0.45)
    --output          Output path
"""

import argparse
import os
import pickle

from clonalge import run_selection as rs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────
CHAIN_COUNT = 10
SECTIONS    = ['P3.3', 'P2.4', 'P1.2']   # display order: SP1, SP2, SP3
SEC_NAMES   = ['SP1',  'SP2',  'SP3']

# One base colour per clone; spots are shaded white→colour by proportion
CLONE_COLS  = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

FS_TITLE = 7
FS_TICK  = 6
FS_LABEL = 6.5

N_S_DATA_DEFAULT = ('prostate_data_configs/'
                    'prostate_cell_count_annotation_any_fixed_margin.txt')


# ══════════════════════════════════════════════════════════════════════════════
# Consensus H  (identical pipeline to other plotting scripts)
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
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_clone_section(ax, x_vals, y_vals, proportions, cmap, radius, vmax,
                        xlim, ylim):
    """Draw spots as filled circles; xlim/ylim are shared across all panels."""
    for x, y, p in zip(x_vals, y_vals, proportions):
        color = cmap(p / vmax)
        circle = mpatches.Circle(
            (x, y), radius=radius,
            facecolor=color, edgecolor='#888888', linewidth=0.25)
        ax.add_patch(circle)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=FS_TICK, length=2, width=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)


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
    ap.add_argument('--approach',      type=int, default=2, choices=[1, 2, 3],
                    help='1=highest loglik  2=consensus  3=consensus anchored to HL')
    ap.add_argument('--outdir',        default='plots_paper')
    args = ap.parse_args()

    output = os.path.join(args.outdir, 'figure4.png')

    # ── run selection ─────────────────────────────────────────────────────────
    res = rs.select_runs(args.approach, args.result_prefix, args.num_runs,
                         args.start_seed, args.section, args.threshold)
    H   = res['H_final']
    K   = H.shape[1]
    print(f'H shape: {H.shape}  K={K}')

    # ── spatial coords ────────────────────────────────────────────────────────
    spots_file = f'{args.result_prefix}_{args.spots_run}/spots_order.txt'
    spots = pd.read_csv(spots_file, header=None)[0].tolist()

    n_s_data = pd.read_csv(args.n_s_data, sep=',')
    n_s_data = n_s_data[~n_s_data.isin([float('nan'), float('inf'), float('-inf')]).any(axis=1)]
    n_s_data = n_s_data.drop_duplicates()
    n_s_data = n_s_data[n_s_data['type'].str.contains('Cancer')]
    n_s_data['x'] = n_s_data['x'].astype(float)
    n_s_data['y'] = n_s_data['y'].astype(float)
    n_s_data = n_s_data[n_s_data['barcode'].isin(spots)].drop_duplicates('barcode')
    n_s_data = (pd.DataFrame({'barcode': spots})
                  .merge(n_s_data, on='barcode', how='left')
                  .dropna(subset=['x', 'y']))

    H_df = pd.DataFrame(H, index=spots, columns=[f'C{k+1}' for k in range(K)])

    # shared colormap (white → pale yellow → dark brown-red)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'wylrd', ['#ffffff', '#ffffb2', '#fecc5c', '#fd8d3c', '#e31a1c', '#800026'])
    vmax = H.max()

    # shared axis limits: global extent + padding so all panels are same size
    pad = args.radius * 1.8
    gx_min = n_s_data['x'].min() - pad
    gx_max = n_s_data['x'].max() + pad
    gy_min = n_s_data['y'].min() - pad
    gy_max = n_s_data['y'].max() + pad
    # make the range square so circles look the same in both axes
    x_span = gx_max - gx_min
    y_span = gy_max - gy_min
    span   = max(x_span, y_span)
    gx_mid = (gx_min + gx_max) / 2
    gy_mid = (gy_min + gy_max) / 2
    xlim   = (gx_mid - span / 2, gx_mid + span / 2)
    ylim   = (gy_mid - span / 2, gy_mid + span / 2)

    # ── figure: 3 rows (sections) × 4 cols (clones) ───────────────────────────
    plt.rcParams.update({
        'font.family':   'sans-serif',
        'font.size':      FS_TICK,
        'axes.linewidth': 0.4,
        'pdf.fonttype':   42,
        'svg.fonttype':   'none',
    })

    fig = plt.figure(figsize=(2.5 * K + 0.6, 2.8 * len(SECTIONS)))
    gs  = fig.add_gridspec(len(SECTIONS), K,
                           wspace=0.0, hspace=0.0,
                           left=0.08, right=0.88, top=0.93, bottom=0.02)
    axes = np.array([[fig.add_subplot(gs[i, k])
                      for k in range(K)]
                     for i in range(len(SECTIONS))])
    # small colorbar axes: vertically centred, sitting just right of the grid
    cax = fig.add_axes([0.91, 0.38, 0.025, 0.30])

    cx = (xlim[0] + xlim[1]) / 2   # centre of shared frame
    cy = (ylim[0] + ylim[1]) / 2

    for i, (sec, sname) in enumerate(zip(SECTIONS, SEC_NAMES)):
        mask     = n_s_data['section'] == sec
        coords   = n_s_data[mask].reset_index(drop=True)
        barcodes = coords['barcode'].tolist()

        # shift this section's spots to be centred in the shared frame
        x_raw = coords['x'].values
        y_raw = coords['y'].values
        x_c   = (x_raw.min() + x_raw.max()) / 2
        y_c   = (y_raw.min() + y_raw.max()) / 2
        x_shifted = x_raw - x_c + cx
        y_shifted = y_raw - y_c + cy

        for k in range(K):
            ax = axes[i, k]
            proportions = H_df.loc[barcodes, f'C{k+1}'].values

            draw_clone_section(ax, x_shifted, y_shifted,
                               proportions, cmap, args.radius, vmax,
                               xlim, ylim)

            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)

            # column header (clone) — top row only
            if i == 0:
                ax.set_title(f'Clone {k+1}', fontsize=FS_TITLE,
                             fontweight='bold', pad=3)

            # row label (section) — left column only
            if k == 0:
                ax.set_ylabel(f'{sname}\n({len(barcodes)} spots)',
                              fontsize=FS_LABEL, labelpad=4)

    # ── single shared colorbar in dedicated axes ───────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label('Clone proportion', fontsize=FS_LABEL, labelpad=3)

    os.makedirs(args.outdir, exist_ok=True)
    plt.savefig(output, dpi=400, bbox_inches='tight')
    print(f'Saved → {output}')
    plt.close()


if __name__ == '__main__':
    main()
