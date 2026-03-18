"""
Standalone Figure 5 generator for ClonalGE real prostate data results.

Runs chain selection (if not already done) then plots the Y_pred vs Y
scatter panels saved by the notebook as Figure 5.

Usage:
    python plot_figure5.py [options]

Options:
    --result_prefix   Directory prefix used in run_figure5.sh   (default: Results_figure5)
    --num_runs        Number of independent runs                 (default: 10)
    --start_seed      Starting seed                              (default: 1)
    --section         Config section name in result filenames    (default: P1.2)
    --tum_h           Path to Tumoroscope H .npy (optional, for panel a comparison)
    --tum_n           Path to Tumoroscope N .npy (optional, for panel a comparison)
    --output          Output figure filename                     (default: figure5.png)

Expected inputs (produced by run_figure5.sh):
    <result_prefix>_1/results_<section>_chain_0 ... _chain_9
    ...
    <result_prefix>_<num_runs>/results_<section>_chain_0 ... _chain_9
    <result_prefix>_1/spots_order.txt   (written by the first run)
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

# ── chain selection (mirrors select_chains.py) ───────────────────────────────

PEARSON_THRESHOLD = 0.9
CHAIN_COUNT = 10


def load_chains(result_prefix, run_seed, section):
    result_obj = f'{result_prefix}_{run_seed}/results_{section}'
    chains = []
    for cc in range(CHAIN_COUNT):
        path = result_obj + f'_chain_{cc}'
        chains.append(pickle.load(open(path, 'rb')))
    return chains


def select_chains_for_run(chains):
    logliks = [c.last_loglik for c in chains]
    best_idx = int(np.argmax(logliks))
    best_H_flat = chains[best_idx].inferred_H.flatten()

    similar = [chains[best_idx]]
    for cc, c in enumerate(chains):
        if cc == best_idx:
            continue
        r, _ = pearsonr(best_H_flat, c.inferred_H.flatten())
        if r > PEARSON_THRESHOLD:
            similar.append(c)
    return similar, best_idx, logliks


def run_chain_selection(result_prefix, num_runs, start_seed, section):
    out_dir = f'{result_prefix}_selected'
    selected = []

    for run in range(start_seed, start_seed + num_runs):
        try:
            chains = load_chains(result_prefix, run, section)
        except FileNotFoundError as e:
            print(f'Run {run}: SKIPPED (file not found: {e})')
            continue

        similar, best_idx, logliks = select_chains_for_run(chains)
        if len(similar) == 1:
            print(f'Run {run}: OMITTED — best chain (idx={best_idx}, '
                  f'loglik={logliks[best_idx]:.2f}) had no similar chains')
        else:
            selected.extend(similar)
            print(f'Run {run}: kept {len(similar)} chains '
                  f'(best idx={best_idx}, loglik={logliks[best_idx]:.2f})')

    if not selected:
        sys.exit('ERROR: no chains selected across all runs.')

    print(f'\nAveraging over {len(selected)} chains...')
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/inferred_H.npy',   np.mean([c.inferred_H   for c in selected], axis=0))
    np.save(f'{out_dir}/inferred_B.npy',   np.mean([c.inferred_B   for c in selected], axis=0))
    np.save(f'{out_dir}/inferred_n.npy',   np.mean([c.inferred_n   for c in selected], axis=0))
    np.save(f'{out_dir}/inferred_phi.npy', np.mean([c.inferred_phi for c in selected], axis=0))
    np.save(f'{out_dir}/inferred_pi.npy',  np.mean([c.inferred_pi  for c in selected], axis=0))
    np.save(f'{out_dir}/inferred_P_Z.npy', np.mean([c.inferred_P_Z for c in selected], axis=0))
    np.save(f'{out_dir}/inferred_Z.npy',   np.mean([c.inferred_Z   for c in selected], axis=0))
    print(f'Chain selection results saved to {out_dir}/')
    return selected


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 5 from ClonalGE prostate results.')
    parser.add_argument('--result_prefix', default='Results_figure5')
    parser.add_argument('--num_runs',      type=int, default=10)
    parser.add_argument('--start_seed',    type=int, default=1)
    parser.add_argument('--section',       default='P1.2')
    parser.add_argument('--tum_h',         default=None,
                        help='Path to Tumoroscope inferred H .npy (for panel a)')
    parser.add_argument('--tum_n',         default=None,
                        help='Path to Tumoroscope inferred N .npy (for panel a)')
    parser.add_argument('--output',        default='figure5.png')
    args = parser.parse_args()

    sel_dir = f'{args.result_prefix}_selected'

    # --- chain selection (skip if already done) ---
    if os.path.exists(f'{sel_dir}/inferred_H.npy'):
        print(f'Found existing chain selection in {sel_dir}/, skipping re-selection.')
    else:
        run_chain_selection(args.result_prefix, args.num_runs, args.start_seed, args.section)

    # --- load averaged ClonalGE estimates ---
    H_pred = np.load(f'{sel_dir}/inferred_H.npy')
    B_pred = np.load(f'{sel_dir}/inferred_B.npy')
    N_pred = np.load(f'{sel_dir}/inferred_n.npy')

    # --- load observed data (Y, p_y) from the first chain of the first run ---
    first_chain_path = (f'{args.result_prefix}_{args.start_seed}/'
                        f'results_{args.section}_chain_0')
    print(f'Loading observed data from {first_chain_path}')
    t = pickle.load(open(first_chain_path, 'rb'))

    # --- load spot barcodes ---
    spots_file = f'{args.result_prefix}_{args.start_seed}/spots_order.txt'
    spots = pd.read_csv(spots_file, header=None).loc[:, 0]

    # --- per-section expression scaling (as in the notebook) ---
    sections = ['P1.2', 'P2.4', 'P3.3']
    means = np.array([t.Y[spots.str.startswith(sec)].mean() for sec in sections])
    scaling_factors = np.mean(means) / means

    Y_scaled = t.Y.astype(float).copy()
    for sec, s in zip(sections, scaling_factors):
        Y_scaled[spots.str.startswith(sec)] *= s

    scaling_factors_array = np.zeros(t.Y.shape[0])
    for sec, s in zip(sections, scaling_factors):
        scaling_factors_array[spots.str.startswith(sec)] = s
    scaling_factors_array = (1.0 / scaling_factors_array).reshape(-1, 1)

    # --- build panels ---
    panels = []

    # Panel a: Tumoroscope + linear regression (optional)
    if args.tum_h and args.tum_n:
        print('Computing Tumoroscope + LR panel...')
        H_tum = np.load(args.tum_h)
        N_tum = np.load(args.tum_n)
        N_mat = N_tum * np.eye(len(N_tum))
        X_tum = np.matmul(N_mat, H_tum)
        g_count = t.Y.shape[1]
        B_tum = np.zeros((H_tum.shape[1], g_count))
        for g in range(g_count):
            if Y_scaled[:, g].sum() > 0:
                B_tum[:, g] = lsq_linear(X_tum, Y_scaled[:, g], bounds=(0, np.inf)).x
        Y_pred_tum = np.matmul(X_tum * scaling_factors_array, B_tum)
        panels.append(('Tumoroscope + LR', Y_pred_tum))

    # Panel b: ClonalGE
    print('Computing ClonalGE panel...')
    N_mat = N_pred * np.eye(len(N_pred))
    X_ge  = np.matmul(N_mat, H_pred)
    Y_pred_ge = np.matmul(X_ge * scaling_factors_array, B_pred)
    panels.append(('TumoroscopeGE', Y_pred_ge))

    # --- plot ---
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels / 2, 3))
    if n_panels == 1:
        axes = [axes]

    letters = list('abcdefgh')
    for i, (title, Y_pred) in enumerate(panels):
        r_nb = np.matmul(Y_pred, np.diag(t.p_y / (1 - t.p_y)))
        ll   = scipy.stats.nbinom.logpmf(t.Y, r_nb, p=t.p_y)

        sc = axes[i].scatter(
            t.Y[:, ::-1], Y_pred[:, ::-1],
            s=10, c=ll[:, ::-1], cmap='viridis',
            vmin=ll.min() - 1, vmax=0
        )
        lims = axes[i].get_xlim()
        axes[i].plot(lims, lims, '--', color='gray', alpha=0.5, linewidth=0.7)
        axes[i].set_xlabel('True gene expression')
        axes[i].set_ylabel('Calculated gene expression')
        axes[i].set_xlim(-100, np.max(t.Y) + 100)
        axes[i].set_ylim(-100, np.max(t.Y) + 100)
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(1000))
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(1000))

        cb = plt.colorbar(sc, label='Log-likelihood', ax=axes[i])
        cb.outline.set_edgecolor('k')
        cb.outline.set_linewidth(0.5)

        corr = np.round(np.corrcoef(t.Y.flatten(), Y_pred.flatten())[0, 1], 3)
        axes[i].text(0.74, 0.97, f'r={corr}',
                     transform=axes[i].transAxes, fontsize=6,
                     verticalalignment='top',
                     bbox=dict(boxstyle='square', facecolor='white',
                               edgecolor='black', lw=0.3))
        axes[i].text(-0.25, 1.1, letters[i],
                     transform=axes[i].transAxes, fontsize=8, va='top')
        axes[i].set_title(title)

    plt.subplots_adjust(wspace=0.40)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=400, bbox_inches='tight')
    print(f'Figure saved to {args.output}')


if __name__ == '__main__':
    main()
