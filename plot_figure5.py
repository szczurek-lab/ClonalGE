"""
Figure 5 and supplementary figure generator for ClonalGE real prostate data.

Steps:
  1. Within-run chain selection (Pearson r > 0.9 against best-loglik chain)
  2. Cross-run selection (keep runs whose averaged H agrees with the best run)
  3. Average selected chains → main Figure 5 scatter plot
  4. Per-run scatter plots → supplementary figure

Usage:
    python plot_figure5.py [options]

Options:
    --result_prefix     Directory prefix used in run_figure5.sh  (default: Results_figure5)
    --num_runs          Number of independent runs                (default: 10)
    --start_seed        Starting seed                             (default: 1)
    --section           Section name in result filenames          (default: all)
    --cross_run_thresh  Pearson r threshold for cross-run filter  (default: 0.9)
    --tum_h             Tumoroscope H .npy path (optional, panel a)
    --tum_n             Tumoroscope N .npy path (optional, panel a)
    --output            Main figure output path      (default: plots_paper/figure5.png)
    --output_supp       Supp figure output path      (default: plots_paper/figure5_supp.png)
    --force_reselect    Re-run chain selection even if cache exists

Expected inputs (produced by run_figure5.sh):
    <result_prefix>_<i>/inferred_vars_<section>_chain_0 ... _chain_9
    <result_prefix>_1/spots_order.txt
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr

CHAIN_COUNT = 10


# ── data loading ──────────────────────────────────────────────────────────────

def load_chains(result_prefix, run_seed, section):
    result_obj = f'{result_prefix}_{run_seed}/inferred_vars_{section}'
    chains = []
    for cc in range(CHAIN_COUNT):
        chains.append(pickle.load(open(result_obj + f'_chain_{cc}', 'rb')))
    return chains


# ── within-run chain selection ────────────────────────────────────────────────

def select_within_run(chains, threshold=0.9):
    """Keep chains correlated with the best-loglik chain within a single run."""
    logliks = [c.last_loglik for c in chains]
    best_idx = int(np.argmax(logliks))
    best_H_flat = chains[best_idx].inferred_H.flatten()

    similar = [chains[best_idx]]
    for cc, c in enumerate(chains):
        if cc == best_idx:
            continue
        r, _ = pearsonr(best_H_flat, c.inferred_H.flatten())
        if r > threshold:
            similar.append(c)
    return similar, best_idx, logliks


# ── per-run processing ────────────────────────────────────────────────────────

def process_all_runs(result_prefix, num_runs, start_seed, section, threshold=0.9):
    """
    Load and within-run-select chains for every run.
    Returns a dict keyed by run index with fields:
        chains      — selected chain objects
        H, B, n     — averaged inferred variables
        best_loglik — best loglik within the run
        n_kept      — number of chains kept
    """
    per_run = {}
    for run in range(start_seed, start_seed + num_runs):
        try:
            chains = load_chains(result_prefix, run, section)
        except FileNotFoundError as e:
            print(f'Run {run}: SKIPPED ({e})')
            continue

        similar, best_idx, logliks = select_within_run(chains, threshold)
        n_kept = len(similar)
        status = ('OMITTED (no similar chains)' if n_kept == 1
                  else f'kept {n_kept} chains')
        print(f'Run {run}: {status}  '
              f'(best idx={best_idx}, loglik={logliks[best_idx]:.2f})')

        if n_kept == 1:
            continue  # drop runs where no chain agrees with the best

        per_run[run] = dict(
            chains=similar,
            H=np.mean([c.inferred_H for c in similar], axis=0),
            B=np.mean([c.inferred_B for c in similar], axis=0),
            n=np.mean([c.inferred_n for c in similar], axis=0),
            best_loglik=logliks[best_idx],
            n_kept=n_kept,
        )

    if not per_run:
        sys.exit('ERROR: no valid runs found.')
    return per_run


# ── cross-run selection ───────────────────────────────────────────────────────

def cross_run_selection(per_run, threshold=0.9):
    """
    Find the run with the highest best_loglik, then keep only runs whose
    averaged H correlates with the best run's H at r > threshold.
    Returns (selected_run_ids, best_run_id, corr_per_run).
    """
    best_run = max(per_run, key=lambda r: per_run[r]['best_loglik'])
    ref_H = per_run[best_run]['H'].flatten()

    selected = []
    corrs = {}
    for run, data in per_run.items():
        r, _ = pearsonr(ref_H, data['H'].flatten())
        corrs[run] = r
        if r >= threshold:
            selected.append(run)

    return selected, best_run, corrs


# ── Y_pred computation ────────────────────────────────────────────────────────

def compute_Y_pred(H, B, n, scaling_factors_array):
    N_mat = n * np.eye(len(n))
    X = np.matmul(N_mat, H)
    return np.matmul(X * scaling_factors_array, B)


# ── scatter panel ─────────────────────────────────────────────────────────────

def scatter_panel(ax, Y_true, Y_pred, p_y, title, letter=None, fontsize=7):
    r_nb = np.matmul(Y_pred, np.diag(p_y / (1 - p_y)))
    ll   = scipy.stats.nbinom.logpmf(Y_true, r_nb, p=p_y)

    sc = ax.scatter(Y_true[:, ::-1], Y_pred[:, ::-1],
                    s=4, c=ll[:, ::-1], cmap='viridis',
                    vmin=ll.min() - 1, vmax=0)
    lims = ax.get_xlim()
    ax.plot(lims, lims, '--', color='gray', alpha=0.5, linewidth=0.7)
    ax.set_xlim(-100, np.max(Y_true) + 100)
    ax.set_ylim(-100, np.max(Y_true) + 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.set_title(title, fontsize=fontsize)

    corr = np.round(np.corrcoef(Y_true.flatten(), Y_pred.flatten())[0, 1], 3)
    ax.text(0.97, 0.97, f'r={corr}',
            transform=ax.transAxes, fontsize=fontsize - 1,
            ha='right', va='top',
            bbox=dict(boxstyle='square', facecolor='white',
                      edgecolor='black', lw=0.3))
    if letter:
        ax.text(-0.25, 1.1, letter, transform=ax.transAxes,
                fontsize=fontsize + 1, va='top', fontweight='bold')
    return sc, corr


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_prefix',    default='Results_figure5')
    parser.add_argument('--num_runs',         type=int, default=10)
    parser.add_argument('--start_seed',       type=int, default=1)
    parser.add_argument('--section',          default='all')
    parser.add_argument('--cross_run_thresh', type=float, default=0.9)
    parser.add_argument('--tum_h',            default=None)
    parser.add_argument('--tum_n',            default=None)
    parser.add_argument('--output',           default='plots_paper/figure5.png')
    parser.add_argument('--output_supp',      default='plots_paper/figure5_supp.png')
    parser.add_argument('--force_reselect',   action='store_true')
    args = parser.parse_args()

    # ── step 1: within-run selection ─────────────────────────────────────────
    print('=== Within-run chain selection ===')
    per_run = process_all_runs(
        args.result_prefix, args.num_runs, args.start_seed,
        args.section, threshold=args.cross_run_thresh)

    # ── step 2: cross-run selection ───────────────────────────────────────────
    print('\n=== Cross-run selection ===')
    selected_runs, best_run, corrs = cross_run_selection(per_run, args.cross_run_thresh)
    print(f'Best run: {best_run}  (loglik={per_run[best_run]["best_loglik"]:.2f})')
    for run in sorted(per_run):
        status = 'SELECTED' if run in selected_runs else 'EXCLUDED'
        print(f'  Run {run:2d}: r={corrs[run]:.4f}  [{status}]')
    print(f'\n{len(selected_runs)} runs selected: {selected_runs}')

    # ── step 3: average across selected runs ──────────────────────────────────
    all_selected_chains = []
    for run in selected_runs:
        all_selected_chains.extend(per_run[run]['chains'])

    H_pred = np.mean([c.inferred_H for c in all_selected_chains], axis=0)
    B_pred = np.mean([c.inferred_B for c in all_selected_chains], axis=0)
    N_pred = np.mean([c.inferred_n for c in all_selected_chains], axis=0)

    # ── load observed data ────────────────────────────────────────────────────
    ref_chain_path = (f'{args.result_prefix}_{args.start_seed}/'
                      f'inferred_vars_{args.section}_chain_0')
    print(f'\nLoading observed data from {ref_chain_path}')
    t = pickle.load(open(ref_chain_path, 'rb'))

    spots_file = f'{args.result_prefix}_{args.start_seed}/spots_order.txt'
    spots = pd.read_csv(spots_file, header=None).loc[:, 0]

    # ── per-section scaling ───────────────────────────────────────────────────
    sections = ['P1.2', 'P2.4', 'P3.3']
    means = np.array([t.Y[spots.str.startswith(s)].mean() for s in sections])
    scaling_factors = np.mean(means) / means

    Y_scaled = t.Y.astype(float).copy()
    for sec, s in zip(sections, scaling_factors):
        Y_scaled[spots.str.startswith(sec)] *= s

    sfa = np.zeros(t.Y.shape[0])
    for sec, s in zip(sections, scaling_factors):
        sfa[spots.str.startswith(sec)] = s
    sfa = (1.0 / sfa).reshape(-1, 1)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_supp) or '.', exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN FIGURE 5
    # ══════════════════════════════════════════════════════════════════════════
    print('\n=== Generating main Figure 5 ===')
    panels = []

    if args.tum_h and args.tum_n:
        print('Computing Tumoroscope + LR panel...')
        H_tum = np.load(args.tum_h)
        N_tum = np.load(args.tum_n)
        N_mat = N_tum * np.eye(len(N_tum))
        X_tum = np.matmul(N_mat, H_tum)
        g_count = t.Y.shape[1]
        B_tum = np.zeros((H_tum.shape[0], g_count))
        for g in range(g_count):
            if Y_scaled[:, g].sum() > 0:
                B_tum[:, g] = lsq_linear(X_tum, Y_scaled[:, g], bounds=(0, np.inf)).x
        panels.append(('Tumoroscope + LR', np.matmul(X_tum * sfa, B_tum)))

    print('Computing ClonalGE panel...')
    panels.append(('TumoroscopeGE', compute_Y_pred(H_pred, B_pred, N_pred, sfa)))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3))
    if n_panels == 1:
        axes = [axes]

    for i, (title, Y_pred) in enumerate(panels):
        sc, corr = scatter_panel(axes[i], t.Y, Y_pred, t.p_y,
                                 title=title, letter='ab'[i])
        axes[i].set_xlabel('True gene expression')
        axes[i].set_ylabel('Calculated gene expression' if i == 0 else '')
        cb = plt.colorbar(sc, label='Log-likelihood', ax=axes[i])
        cb.outline.set_edgecolor('k')
        cb.outline.set_linewidth(0.5)

    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig(args.output, dpi=400, bbox_inches='tight')
    print(f'Main figure saved to {args.output}')
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # SUPPLEMENTARY: one panel per run
    # ══════════════════════════════════════════════════════════════════════════
    print('\n=== Generating supplementary figure (per-run) ===')
    run_ids = sorted(per_run.keys())
    n_runs  = len(run_ids)
    ncols   = 5
    nrows   = int(np.ceil(n_runs / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    for idx, run in enumerate(run_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        Y_pred_run = compute_Y_pred(
            per_run[run]['H'], per_run[run]['B'], per_run[run]['n'], sfa)

        is_selected = run in selected_runs
        r_cross = corrs[run]
        ll_best = per_run[run]['best_loglik']
        title = (f'Run {run}  [{"✓" if is_selected else "✗"}]\n'
                 f'loglik={ll_best:.0f}  r_cross={r_cross:.3f}')

        sc, corr = scatter_panel(ax, t.Y, Y_pred_run, t.p_y,
                                 title=title, fontsize=6)
        ax.set_xlabel('True Y', fontsize=6)
        ax.set_ylabel('Pred Y', fontsize=6)
        ax.tick_params(labelsize=5)
        edge_color = '#2ca02c' if is_selected else '#d62728'
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(1.5)

    # hide unused axes
    for idx in range(n_runs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # shared colorbar
    fig.subplots_adjust(right=0.88, wspace=0.45, hspace=0.55)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sc, cax=cbar_ax, label='Log-likelihood')
    cb.outline.set_edgecolor('k')

    fig.suptitle(
        'Per-run Y prediction  '
        '(green border = selected by cross-run filter, red = excluded)',
        fontsize=8, y=1.01)

    plt.savefig(args.output_supp, dpi=350, bbox_inches='tight')
    print(f'Supplementary figure saved to {args.output_supp}')
    plt.close()


if __name__ == '__main__':
    main()
