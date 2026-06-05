"""
Plot Figure 2: simulated-data validation comparing three methods.

Methods compared:
  - Tumoroscope          (H, phi, n only — no B)
  - ClonalGE 1-chain
  - ClonalGE 5-chains (best chain by log-likelihood)

Metric: rMAE = SEE / mean(true_value).
Mean(true) is taken from the 5-chain results files (which append it);
since all methods run on the same simulated data, these denominators apply
equally to all three methods.

Usage:
    python plot_simulated_results.py \\
        --results_1chain   /Volumes/LenovoPS8/ClonalGE/Results_simulated_data \\
        --results_5chains  /Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains \\
        --results_tum      /Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope \\
        --num_runs 20 \\
        --output plots_paper/figure2_simulated.png
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import argparse
import os
import re
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CONDITIONS  = ['normal', 'low_variance', 'high_coverage']
COND_LABELS = ['Normal', 'Low variance', 'High coverage']

# Metrics to plot and their display labels
METRICS = ['H_SEE', 'phi_SEE', 'n_SEE', 'B_SEE']
METRIC_LABELS = [
    r'$H$ rMAE',
    r'$\phi$ rMAE',
    r'$N$ rMAE',
    r'$B$ rMAE',
]
MEAN_TRUE_KEYS = {
    'H_SEE':   'mean_H',
    'phi_SEE': 'mean_phi',
    'n_SEE':   'mean_n',
    'B_SEE':   'mean_B',
}

# Method colours — ColorBrewer Pastel1 (no hatch)
METHOD_COLORS  = ['#fbb4ae', '#fed9a6', '#b3cde3', '#ccebc5']   # Tum+LR, Tum+NB, CGE-1, CGE-5
METHOD_HATCHES = ['',        '',        '',        ''      ]
METHOD_LABELS  = ['Tumoroscope+LR', 'Tumoroscope+NB', 'ClonalGE (1 chain)', 'ClonalGE (5 chains)']

FS_TITLE = 9
FS_LABEL = 8
FS_TICK  = 7


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_results_file(path):
    """Return dict of floats from a results_*.txt file."""
    txt = open(path).read()
    out = {}

    patterns = {
        'H_SEE':   r'Standard Error of the Estimate H:\n([\d.eE+\-nan]+)',
        'phi_SEE': r'Standard Error of the Estimate phi:\n([\d.eE+\-nan]+)',
        'pi_SEE':  r'Standard Error of the Estimate pi:\n([\d.eE+\-nan]+)',
        'n_SEE':   r'Standard Error of the Estimate n:\n([\d.eE+\-nan]+)',
        # ClonalGE writes "Standard Error of the Estimate B:"
        # Tumoroscope+LR writes "Standard Error of the Estimate B (Tumoroscope+LR):"
        # Match ClonalGE first (no parentheses); fall through to LR variant below.
        'B_SEE':    r'Standard Error of the Estimate B(?! \():\n([\d.eE+\-nan]+)',
        'B_SEE_LR': r'Standard Error of the Estimate B \(Tumoroscope\+LR\):\n([\d.eE+\-nan]+)',
        'B_SEE_NB': r'Standard Error of the Estimate B \(Tumoroscope\+NB\):\n([\d.eE+\-nan]+)',
        'mean_H':   r'Mean of true H:\n([\d.eE+\-nan]+)',
        'mean_phi': r'Mean of true phi:\n([\d.eE+\-nan]+)',
        'mean_n':   r'Mean of true n:\n([\d.eE+\-nan]+)',
        'mean_B':   r'Mean of true B:\n([\d.eE+\-nan]+)',
    }

    for key, pat in patterns.items():
        m = re.search(pat, txt)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                out[key] = float('nan')
        else:
            out[key] = float('nan')

    # If ClonalGE B_SEE is missing, fall back to LR variant (Tumoroscope results)
    if np.isnan(out.get('B_SEE', float('nan'))):
        out['B_SEE'] = out.get('B_SEE_LR', float('nan'))

    return out


def find_results_file(results_dir, run, cond):
    """Try both {dir}/{run}/results_{cond}.txt and {dir}{run}/results_{cond}.txt."""
    p1 = os.path.join(results_dir, str(run), f'results_{cond}.txt')
    if os.path.exists(p1):
        return p1
    p2 = results_dir.rstrip('/') + str(run) + f'/results_{cond}.txt'
    if os.path.exists(p2):
        return p2
    return None


def load_all(results_dir, num_runs):
    """Return data[cond][metric] = list of raw values (one per run)."""
    all_keys = list({
        'H_SEE', 'phi_SEE', 'n_SEE', 'B_SEE', 'B_SEE_LR', 'B_SEE_NB', 'pi_SEE',
        'mean_H', 'mean_phi', 'mean_n', 'mean_B'
    })
    data = {c: {k: [] for k in all_keys} for c in CONDITIONS}
    missing = 0
    for run in range(1, num_runs + 1):
        for cond in CONDITIONS:
            fpath = find_results_file(results_dir, run, cond)
            if fpath is None:
                print(f'  MISSING: run {run} / {cond} in {results_dir}')
                missing += 1
                for k in all_keys:
                    data[cond][k].append(float('nan'))
                continue
            vals = parse_results_file(fpath)
            for k in all_keys:
                data[cond][k].append(vals.get(k, float('nan')))
    if missing:
        print(f'  ({missing} files missing — filled with NaN)')
    return data


# ── best-chain CSV loader ─────────────────────────────────────────────────────

def load_best_chain_csv(csv_path, num_runs):
    """
    Load best-chain SEE values from extract_best_see.py output CSV.
    Returns data[cond][metric] = list of floats (one per run, NaN if missing).
    SEE metrics only — no mean_true (those come from the txt files).
    """
    all_keys = ['H_SEE', 'phi_SEE', 'n_SEE', 'B_SEE']
    data = {c: {k: [float('nan')] * num_runs for k in all_keys} for c in CONDITIONS}

    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            run  = int(row['run']) - 1   # 0-based index
            cond = row['cond']
            if cond not in data:
                continue
            for k in all_keys:
                try:
                    data[cond][k][run] = float(row[k])
                except (KeyError, ValueError):
                    pass

    found = sum(1 for c in CONDITIONS for v in data[c]['H_SEE'] if not np.isnan(v))
    print(f'  Loaded {found} best-chain rows from {csv_path}')
    return data


# ── rMAE computation ──────────────────────────────────────────────────────────

def compute_rmae(data, ref_data):
    """
    Compute rMAE = SEE / mean(true) for each metric.
    ref_data provides mean(true) denominators; falls back to data itself
    if ref_data has NaN for a given run (e.g. Tumoroscope has its own mean_true).
    Returns rmae[cond][metric] = list of floats.
    """
    rmae = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        for metric, mean_key in MEAN_TRUE_KEYS.items():
            sees  = np.array(data[cond][metric])
            ref   = np.array(ref_data[cond][mean_key])
            own   = np.array(data[cond][mean_key])
            # Use ref denominator where available; fall back to own mean_true
            denom = np.where(np.isnan(ref), own, ref)
            with np.errstate(invalid='ignore', divide='ignore'):
                vals = sees / denom
            rmae[cond][metric] = vals.tolist()
    return rmae


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_box(ax, vals, pos, color, hatch=''):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return
    ax.boxplot(
        vals,
        positions=[pos],
        widths=0.22,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2, markerfacecolor=color,
                        markeredgecolor=color, alpha=0.5),
        medianprops=dict(color='black', linewidth=1.2),
        boxprops=dict(facecolor=color, alpha=0.7, linewidth=0.6, hatch=hatch),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
    )


def make_figure(rmae_tum, rmae_1chain, rmae_5chains, outpath, rmae_tum_nb=None):
    """
    rmae_tum     : rMAE dict for Tumoroscope (B column will be skipped)
    rmae_1chain  : rMAE dict for ClonalGE 1-chain
    rmae_5chains : rMAE dict for ClonalGE 5-chains
    """
    plt.rcParams.update({
        'font.family':   'sans-serif',
        'font.size':      FS_TICK,
        'axes.linewidth': 0.5,
        'pdf.fonttype':   42,
        'svg.fonttype':   'none',
    })

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(1.85 * n_metrics, 2.0))

    n_cond  = len(CONDITIONS)
    group_w = 2.0
    offsets = np.arange(n_cond) * group_w

    has_nb = rmae_tum_nb is not None

    if has_nb:
        # Four methods: Tum+LR, Tum+NB, CGE-1, CGE-5
        dx = [-0.36, -0.12, 0.12, 0.36]
        methods_H = [
            (rmae_tum,    METHOD_COLORS[0], METHOD_HATCHES[0]),
            (rmae_tum_nb, METHOD_COLORS[1], METHOD_HATCHES[1]),
            (rmae_1chain, METHOD_COLORS[2], METHOD_HATCHES[2]),
            (rmae_5chains,METHOD_COLORS[3], METHOD_HATCHES[3]),
        ]
        # For H/phi/n: Tum+LR and Tum+NB show the same Tumoroscope values
        methods_nonB = [
            (rmae_tum,    METHOD_COLORS[0], METHOD_HATCHES[0]),
            (rmae_1chain, METHOD_COLORS[2], METHOD_HATCHES[2]),
            (rmae_5chains,METHOD_COLORS[3], METHOD_HATCHES[3]),
        ]
        dx_nonB = [-0.26, 0.0, 0.26]
    else:
        dx = [-0.26, 0.0, 0.26]
        methods_H = [
            (rmae_tum,    METHOD_COLORS[0], METHOD_HATCHES[0]),
            (rmae_1chain, METHOD_COLORS[2], METHOD_HATCHES[2]),
            (rmae_5chains,METHOD_COLORS[3], METHOD_HATCHES[3]),
        ]
        methods_nonB = methods_H
        dx_nonB = dx

    for col, (metric, ylabel) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes[col]
        is_B = (metric == 'B_SEE')

        cur_methods = methods_H if is_B else methods_nonB
        cur_dx      = dx       if is_B else dx_nonB

        for ci in range(n_cond):
            for mi, (rmae_dict, color, hatch) in enumerate(cur_methods):
                vals = rmae_dict[CONDITIONS[ci]][metric]
                _draw_box(ax, vals, offsets[ci] + cur_dx[mi], color, hatch)

        ax.set_xticks(offsets)
        ax.set_xticklabels(COND_LABELS, rotation=30, ha='right',
                           fontsize=FS_TICK - 0.5)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title('abcd'[col], fontsize=FS_TITLE, fontweight='bold',
                     loc='left', pad=3)
        ax.tick_params(axis='y', labelsize=FS_TICK - 0.5, length=2, width=0.4)
        ax.tick_params(axis='x', length=0)
        ax.axhline(0, color='grey', linewidth=0.4, linestyle='--', alpha=0.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.set_xlim(offsets[0] - 0.7, offsets[-1] + 0.7)

    # Legend — show all 4 methods (or 3 if NB not provided)
    legend_labels  = METHOD_LABELS  if has_nb else [METHOD_LABELS[0]] + METHOD_LABELS[2:]
    legend_colors  = METHOD_COLORS  if has_nb else [METHOD_COLORS[0]] + METHOD_COLORS[2:]
    legend_hatches = METHOD_HATCHES if has_nb else [METHOD_HATCHES[0]] + METHOD_HATCHES[2:]
    legend_handles = [
        mpatches.Patch(facecolor=c, alpha=0.7, edgecolor='grey',
                       linewidth=0.5, hatch=h, label=l)
        for c, h, l in zip(legend_colors, legend_hatches, legend_labels)
    ]
    n_legend = len(legend_labels)
    fig.legend(handles=legend_handles,
               loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=n_legend, fontsize=FS_LABEL, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    print(f'Saved → {outpath}')
    plt.close()


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(rmae, label):
    print(f'\n=== {label} — rMAE median [IQR] ===')
    for cond, clabel in zip(CONDITIONS, COND_LABELS):
        print(f'  {clabel}:')
        for metric in METRICS:
            vals = np.array([v for v in rmae[cond][metric] if not np.isnan(v)])
            if len(vals) == 0:
                print(f'    {metric}: no data')
                continue
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            print(f'    {metric}: {med:.4f}  [{q25:.4f}, {q75:.4f}]  n={len(vals)}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_1chain',
                    default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_data',
                    help='ClonalGE 1-chain results dir')
    ap.add_argument('--results_5chains',
                    default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains',
                    help='ClonalGE 5-chain results dir (used for mean(true) denominators)')
    ap.add_argument('--best_chain_csv',
                    default=None,
                    help='CSV from extract_best_see.py with best-chain SEE values (optional but recommended)')
    ap.add_argument('--results_tum',
                    default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope',
                    help='Tumoroscope results dir (prefix; run number appended)')
    ap.add_argument('--num_runs', type=int, default=20)
    ap.add_argument('--output',   default='plots_paper/figure2_simulated.png')
    args = ap.parse_args()

    print(f'Loading ClonalGE 5-chain results (for mean-true denominators)...')
    data_5 = load_all(args.results_5chains, args.num_runs)

    # Override 5-chain SEE values with best-chain CSV if provided
    if args.best_chain_csv:
        print(f'\nLoading best-chain SEE from {args.best_chain_csv}...')
        best = load_best_chain_csv(args.best_chain_csv, args.num_runs)
        for cond in CONDITIONS:
            for k in ['H_SEE', 'phi_SEE', 'n_SEE', 'B_SEE']:
                data_5[cond][k] = best[cond][k]
    else:
        print('\nWARNING: no --best_chain_csv provided; 5-chain SEE values are from a '
              'random chain (whichever wrote last), not the best chain.')
        print('  Run extract_best_see.py on the cluster to get correct values.')

    print(f'\nLoading ClonalGE 1-chain results...')
    data_1 = load_all(args.results_1chain, args.num_runs)

    print(f'\nLoading Tumoroscope results...')
    data_tum = load_all(args.results_tum, args.num_runs)

    # rMAE — all methods use 5-chain mean(true) as denominator (same simulated data)
    rmae_5   = compute_rmae(data_5,   data_5)
    rmae_1   = compute_rmae(data_1,   data_5)
    rmae_tum = compute_rmae(data_tum, data_5)

    # Build Tumoroscope+NB rMAE: same as rmae_tum but B_SEE uses B_SEE_NB
    data_tum_nb = {c: dict(data_tum[c]) for c in CONDITIONS}
    for c in CONDITIONS:
        data_tum_nb[c]['B_SEE'] = data_tum[c]['B_SEE_NB']
    rmae_tum_nb = compute_rmae(data_tum_nb, data_5)

    has_nb = any(not np.isnan(v)
                 for c in CONDITIONS
                 for v in data_tum[c]['B_SEE_NB'])

    print_summary(rmae_tum,    'Tumoroscope+LR')
    print_summary(rmae_tum_nb, 'Tumoroscope+NB')
    print_summary(rmae_1,      'ClonalGE 1-chain')
    print_summary(rmae_5,      'ClonalGE 5-chains')

    make_figure(rmae_tum, rmae_1, rmae_5, args.output,
                rmae_tum_nb=rmae_tum_nb if has_nb else None)


if __name__ == '__main__':
    main()
