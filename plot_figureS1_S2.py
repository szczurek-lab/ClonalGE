"""
Supplementary figure: within-run chain agreement for H and B.

Shows pairwise Pearson r between all 10 MCMC chains within one run as two
heatmaps (clonal composition H, gene expression B).  Chains are reordered so
selected chains (r > threshold against the best-loglik chain) appear first,
with a dividing line separating them from excluded chains.

Usage:
    # preview with synthetic data (no results files needed)
    python plot_chain_agreement.py --synthetic

    # real data
    python plot_chain_agreement.py --result_prefix Results_figure5 --run 1 --section all

Options:
    --result_prefix   Directory prefix used in run scripts  (default: Results_figure5)
    --run             Which run index to display            (default: 1)
    --section         Config section name                   (default: all)
    --threshold       Pearson r threshold for selection     (default: 0.9)
    --output          Output path                           (default: plots_paper/chain_agreement.png)
    --synthetic       Use synthetic data for layout preview
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import pearsonr

# ── layout constants ───────────────────────────────────────────────────────────
CHAIN_COUNT  = 10
FS_BASE      = 7        # base font size (pt) — matches rest of paper
FS_ANNOT     = 5.0      # cell annotation
FS_TICK      = 6.0      # tick labels
FS_TITLE     = 7.0      # panel title
FIG_W        = 5.6      # total figure width (in)  — fits one journal column
FIG_H        = 2.55     # total figure height (in)

# ── colour palette ─────────────────────────────────────────────────────────────
# Sequential blue — clean, print-safe, colourblind-friendly
CMAP         = 'Blues'
VMIN, VMAX   = 0.0, 1.0

COL_SEL      = '#1f4e79'   # dark navy  — selected chain tick labels
COL_EXC      = '#c0392b'   # deep red   — excluded chain tick labels
COL_BEST     = '#1f4e79'   # same as selected (best is bold)
COL_DIAG     = '#e8e8e8'   # very light grey for diagonal cells
COL_DIV      = '#555555'   # dividing line colour


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_chains(result_prefix, run_seed, section, chain_count=CHAIN_COUNT):
    base = f'{result_prefix}_{run_seed}/inferred_vars_{section}'
    chains = []
    for cc in range(chain_count):
        chains.append(pickle.load(open(base + f'_chain_{cc}', 'rb')))
    return chains


def select_within_run(chains, threshold=0.9):
    """Return (selected_indices, best_idx, logliks)."""
    logliks  = [c.last_loglik for c in chains]
    best_idx = int(np.argmax(logliks))
    ref      = chains[best_idx].inferred_H.flatten()

    selected = {best_idx}
    for cc, c in enumerate(chains):
        if cc == best_idx:
            continue
        r, _ = pearsonr(ref, c.inferred_H.flatten())
        if r > threshold:
            selected.add(cc)
    return selected, best_idx, logliks


def pairwise_r(matrices):
    """(n × n) symmetric Pearson r matrix from a list of flattened arrays."""
    n    = len(matrices)
    R    = np.ones((n, n))
    flat = [m.flatten() for m in matrices]
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(flat[i], flat[j])
            R[i, j] = R[j, i] = r
    return R


def reorder(R, selected, best_idx, n):
    """Reorder rows/cols: selected chains first, excluded last."""
    sel_sorted = sorted(selected)
    exc_sorted = sorted(set(range(n)) - selected)
    order      = sel_sorted + exc_sorted
    R_ord      = R[np.ix_(order, order)]
    new_best   = order.index(best_idx)
    new_sel    = set(range(len(sel_sorted)))   # first len(sel) indices
    return R_ord, order, new_best, new_sel, len(sel_sorted)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic test data
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic(n_chains=CHAIN_COUNT, S=55, K=4, g=150, seed=0):
    rng = np.random.default_rng(seed)

    class FakeChain:
        pass

    H_true = rng.dirichlet(np.ones(K), size=S)
    B_true = rng.gamma(2.0, 1.0, size=(K, g))
    chains = []

    for i in range(n_chains):
        c = FakeChain()
        if i < 7:                         # converged chains
            noise_H = rng.normal(0, 0.025, H_true.shape)
            H = np.clip(H_true + noise_H, 1e-9, None)
            H /= H.sum(axis=1, keepdims=True)
            noise_B = rng.normal(0, 0.12, B_true.shape)
            B = np.clip(B_true + noise_B, 0, None)
            c.last_loglik = -1200 + rng.uniform(-40, 40)
        else:                             # diverged chains (clone-swapped)
            perm = rng.permutation(K)
            H = H_true[:, perm]
            B = B_true[perm, :]
            c.last_loglik = -1750 + rng.uniform(-40, 40)
        c.inferred_H = H
        c.inferred_B = B
        chains.append(c)

    return chains


# ══════════════════════════════════════════════════════════════════════════════
# Panel rendering
# ══════════════════════════════════════════════════════════════════════════════

def draw_heatmap(ax, R_ord, order, new_best, new_sel, n_sel, n,
                 title, threshold, show_ylabel=True):
    """
    Draw one 10×10 pairwise Pearson r heatmap on `ax`.
    Chains are pre-reordered: first n_sel rows/cols are selected.
    """
    im = ax.imshow(R_ord, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                   aspect='equal', interpolation='none')

    # ── cell annotations ──
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val      = R_ord[i, j]
            txt_col  = 'white' if val > 0.62 else '#333333'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=FS_ANNOT, color=txt_col, zorder=3)

    # ── diagonal — light grey patch + em-dash ──
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            facecolor=COL_DIAG, edgecolor='none', zorder=2))
        ax.text(i, i, '—', ha='center', va='center',
                fontsize=FS_ANNOT, color='#888888', zorder=3)

    # ── dividing line between selected / excluded ──
    if 0 < n_sel < n:
        cut = n_sel - 0.5
        ax.axhline(cut, color=COL_DIV, linewidth=0.6, linestyle='--', zorder=4)
        ax.axvline(cut, color=COL_DIV, linewidth=0.6, linestyle='--', zorder=4)

    # ── tick labels coloured by selection status ──
    tick_labels = [str(orig + 1) for orig in order]   # 1-based chain numbers
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    xlbls = ax.set_xticklabels(tick_labels, fontsize=FS_TICK)
    ylbls = ax.set_yticklabels(tick_labels, fontsize=FS_TICK)

    for pos, (xl, yl) in enumerate(zip(xlbls, ylbls)):
        is_sel    = pos in new_sel
        is_best   = (pos == new_best)
        col       = COL_SEL if is_sel else COL_EXC
        weight    = 'bold' if is_best else 'normal'
        for lbl in (xl, yl):
            lbl.set_color(col)
            lbl.set_fontweight(weight)

    ax.set_xlabel('Chain', fontsize=FS_BASE, labelpad=2)
    if show_ylabel:
        ax.set_ylabel('Chain', fontsize=FS_BASE, labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return im


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def load_all_runs(result_prefix, num_runs, start_seed, section,
                  chain_count=CHAIN_COUNT):
    """Return list of (seed, chains) for every run that loads successfully."""
    runs = []
    for seed in range(start_seed, start_seed + num_runs):
        try:
            chains = load_chains(result_prefix, seed, section, chain_count)
            runs.append((seed, chains))
            print(f'  Run {seed}: loaded')
        except FileNotFoundError:
            print(f'  Run {seed}: not found, skipping')
    return runs


def _make_rcparams():
    plt.rcParams.update({
        'font.family':   'sans-serif',
        'font.size':      FS_BASE,
        'axes.linewidth': 0.5,
        'pdf.fonttype':   42,
        'svg.fonttype':   'none',
    })


def _add_shared_colorbar(fig, im, rect, threshold):
    cbar_ax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Pearson  r', fontsize=FS_BASE, labelpad=4)
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, length=2, width=0.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(0.5)


def _add_legend(fig, threshold, anchor):
    legend_els = [
        mpatches.Patch(color=COL_SEL,
                       label=f'Selected  (r > {threshold})'),
        mpatches.Patch(color=COL_EXC, label='Excluded'),
        mpatches.Patch(facecolor='none', edgecolor=COL_DIV,
                       linestyle='--', linewidth=0.8,
                       label='Selection boundary'),
    ]
    fig.legend(handles=legend_els, fontsize=FS_BASE - 1,
               loc='lower center', ncol=3, bbox_to_anchor=anchor,
               frameon=False, handlelength=1.0,
               handletextpad=0.5, columnspacing=1.2)


# ── single-run figure ─────────────────────────────────────────────────────────

def plot_single_run(chains, run_label, threshold, output, approach_label=None):
    n = len(chains)
    selected, best_idx, logliks = select_within_run(chains, threshold)
    print(f'  Best chain: {best_idx + 1}  loglik={logliks[best_idx]:.1f}')
    print(f'  Selected {len(selected)}/{n}: {sorted(c + 1 for c in selected)}')

    R_H = pairwise_r([c.inferred_H for c in chains])
    R_B = pairwise_r([c.inferred_B for c in chains])
    R_H_ord, order, new_best, new_sel, n_sel = reorder(R_H, selected, best_idx, n)
    R_B_ord, *_ = reorder(R_B, selected, best_idx, n)

    _make_rcparams()
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H + 0.25),
        gridspec_kw=dict(wspace=0.38, left=0.08, right=0.88,
                         top=0.82, bottom=0.18),
    )
    im = draw_heatmap(axes[0], R_H_ord, order, new_best, new_sel, n_sel, n,
                      title='Clonal composition  (H)',
                      threshold=threshold, show_ylabel=True)
    draw_heatmap(axes[1], R_B_ord, order, new_best, new_sel, n_sel, n,
                 title='Gene expression  (B)',
                 threshold=threshold, show_ylabel=False)
    _add_shared_colorbar(fig, im, [0.905, 0.18, 0.018, 0.70], threshold)
    _add_legend(fig, threshold, (0.46, -0.02))

    # run label (top right, italic grey) — sits just below the suptitle
    fig.text(0.895, 0.91, run_label, fontsize=FS_BASE - 1.5,
             color='#888888', ha='right', va='top', style='italic')
    # approach-specific reference run label — suptitle above everything
    if approach_label is not None:
        fig.suptitle(approach_label, fontsize=FS_BASE - 0.5,
                     color='#333333', style='italic', y=1.01)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    plt.savefig(output, dpi=400, bbox_inches='tight')
    print(f'  Saved → {output}')
    plt.close()


# ── all-runs supplementary grid ───────────────────────────────────────────────

def plot_all_runs(runs, threshold, output):
    """
    Layout: H and B side-by-side per run, RUNS_PER_ROW runs per row.
    With 20 runs and RUNS_PER_ROW=4 this gives 5 rows — each panel is
    tall enough for legible annotations.
    """
    RUNS_PER_ROW = 2          # runs per row; 2 panels (H,B) each → 4 cols
    PAIR_GAP     = 0.10       # gap (inches) between H and B within a run
    RUN_GAP      = 0.28       # gap (inches) between adjacent run-pairs

    FS_A2 = 4.2    # annotation font
    FS_T2 = 6.5    # tick labels
    FS_TT = 7.0    # panel title

    panel_w = 1.85  # inches per individual heatmap
    panel_h = 1.80  # inches per individual heatmap

    n_runs   = len(runs)
    n_rows   = int(np.ceil(n_runs / RUNS_PER_ROW))
    n_cols   = RUNS_PER_ROW * 2   # H + B per run

    # total figure dimensions
    fig_w = (RUNS_PER_ROW * (2 * panel_w + PAIR_GAP)
             + (RUNS_PER_ROW - 1) * RUN_GAP
             + 0.65)                             # + colorbar
    fig_h = n_rows * panel_h + 0.45             # + legend

    _make_rcparams()
    fig = plt.figure(figsize=(fig_w, fig_h))

    # pre-compute left edges for each panel column
    xs = []   # xs[run_in_row][var] = left edge (figure fraction)
    for r in range(RUNS_PER_ROW):
        left = (r * (2 * panel_w + PAIR_GAP + RUN_GAP)) / fig_w
        xs.append([left,
                   (left * fig_w + panel_w + PAIR_GAP) / fig_w])

    im_ref = None

    for run_idx, (seed, chains) in enumerate(runs):
        row     = run_idx // RUNS_PER_ROW
        col_run = run_idx %  RUNS_PER_ROW

        n_chains = len(chains)
        selected, best_idx, logliks = select_within_run(chains, threshold)
        n_sel = len(selected)

        R_H = pairwise_r([c.inferred_H for c in chains])
        R_B = pairwise_r([c.inferred_B for c in chains])
        R_H_ord, order, new_best, new_sel, ns = reorder(
            R_H, selected, best_idx, n_chains)
        R_B_ord, *_ = reorder(R_B, selected, best_idx, n_chains)

        for vi, (R_ord, var_label) in enumerate(
                [(R_H_ord, 'H'), (R_B_ord, 'B')]):

            x0 = xs[col_run][vi]
            y0 = 1.0 - (row * panel_h + panel_h) / fig_h + 0.03
            w  = (panel_w - 0.26) / fig_w
            h  = (panel_h - 0.32) / fig_h
            ax = fig.add_axes([x0, y0, w, h])

            n  = R_ord.shape[0]
            im = ax.imshow(R_ord, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                           aspect='equal', interpolation='none')
            if im_ref is None:
                im_ref = im

            # cell annotations
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    val = R_ord[i, j]
                    tc  = 'white' if val > 0.62 else '#333333'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=FS_A2, color=tc, zorder=3)
            for i in range(n):
                ax.add_patch(plt.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    facecolor=COL_DIAG, edgecolor='none', zorder=2))
                ax.text(i, i, '—', ha='center', va='center',
                        fontsize=FS_A2, color='#888888', zorder=3)

            # selection divider
            if 0 < ns < n:
                cut = ns - 0.5
                ax.axhline(cut, color=COL_DIV, lw=0.5, linestyle='--', zorder=4)
                ax.axvline(cut, color=COL_DIV, lw=0.5, linestyle='--', zorder=4)

            # tick labels
            ordered_labels = [str(order[i] + 1) for i in range(n)]
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            xlbls = ax.set_xticklabels(ordered_labels, fontsize=FS_T2)
            ylbls = ax.set_yticklabels(ordered_labels, fontsize=FS_T2)
            for pos, (xl, yl) in enumerate(zip(xlbls, ylbls)):
                c  = COL_SEL if pos in new_sel else COL_EXC
                fw = 'bold' if pos == new_best else 'normal'
                for lbl in (xl, yl):
                    lbl.set_color(c)
                    lbl.set_fontweight(fw)

            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)

            ax.set_title(
                f'Run {seed} — {var_label}  ({n_sel}/{n_chains})',
                fontsize=FS_TT, pad=2)

            if vi == 0:
                ax.set_ylabel('Chain', fontsize=FS_T2, labelpad=2)
            ax.set_xlabel('Chain', fontsize=FS_T2, labelpad=1)

    # shared colorbar
    cbar_x = 1.0 - 0.50 / fig_w
    cbar_ax = fig.add_axes([cbar_x, 0.06, 0.018, 0.88])
    cb = fig.colorbar(im_ref, cax=cbar_ax)
    cb.set_label('Pearson  r', fontsize=FS_BASE, labelpad=4)
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, length=2, width=0.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(0.5)

    # legend
    legend_els = [
        mpatches.Patch(color=COL_SEL,
                       label=f'Selected  (r > {threshold})'),
        mpatches.Patch(color=COL_EXC, label='Excluded'),
        mpatches.Patch(facecolor='none', edgecolor=COL_DIV,
                       linestyle='--', linewidth=0.7,
                       label='Selection boundary'),
    ]
    fig.legend(handles=legend_els, fontsize=FS_BASE - 1,
               loc='lower center', ncol=3,
               bbox_to_anchor=(0.46, -0.005),
               frameon=False, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.0)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Saved → {output}')
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_prefix', default='Results_figure5')
    ap.add_argument('--num_runs',      type=int, default=10)
    ap.add_argument('--start_seed',    type=int, default=1)
    ap.add_argument('--section',       default='all')
    ap.add_argument('--threshold',     type=float, default=0.9)
    ap.add_argument('--output',        default='plots_paper/chain_agreement.png')
    ap.add_argument('--output_supp',   default='plots_paper/chain_agreement_supp.png')
    ap.add_argument('--ref_run',       type=int, default=None,
                    help='Force this run as the single-run reference (default: '
                         'run with highest best-chain loglik)')
    ap.add_argument('--approach',      type=int, default=None, choices=[1, 2, 3],
                    help='Approach number — adds "approach-specific reference run" '
                         'annotation to the single-run figure')
    ap.add_argument('--synthetic',     action='store_true')
    args = ap.parse_args()

    _make_rcparams()

    if args.synthetic:
        print('Generating synthetic data...')
        chains = make_synthetic()
        print('\n=== Single-run figure (synthetic) ===')
        plot_single_run(chains, 'synthetic', args.threshold, args.output)

        # wrap as runs list for the all-runs figure
        runs = [(1, make_synthetic(seed=s)) for s in range(10)]
        print('\n=== All-runs supplementary figure (synthetic) ===')
        plot_all_runs(runs, args.threshold, args.output_supp)
        return

    print(f'Loading runs from {args.result_prefix}_* ...')
    runs = load_all_runs(args.result_prefix, args.num_runs,
                         args.start_seed, args.section)
    if not runs:
        raise RuntimeError('No runs found.')

    # single-run figure: use --ref_run if given, else highest best-chain loglik
    if args.ref_run is not None:
        matched = [(s, c) for s, c in runs if s == args.ref_run]
        if not matched:
            raise RuntimeError(f'--ref_run {args.ref_run} not found in loaded runs.')
        ref_seed, ref_chains = matched[0]
    else:
        ref_seed, ref_chains = max(
            runs, key=lambda sc: max(c.last_loglik for c in sc[1]))

    approach_label = None
    if args.approach is not None:
        approach_label = f'Approach {args.approach} — approach-specific reference run'

    print(f'\n=== Single-run figure (run {ref_seed}) ===')
    plot_single_run(ref_chains, f'run {ref_seed}',
                    args.threshold, args.output,
                    approach_label=approach_label)

    print('\n=== All-runs supplementary figure ===')
    plot_all_runs(runs, args.threshold, args.output_supp)


if __name__ == '__main__':
    main()
