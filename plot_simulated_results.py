"""
Plot simulated-data validation results for ClonalGE (and optionally Tumoroscope).

Reads results_{condition}.txt from each run directory and produces a
boxplot figure showing H_SEE, phi_SEE, pi_SEE, and B_SEE across
three conditions (normal, low_variance, high_coverage).

Usage:
    # ClonalGE only
    python plot_simulated_results.py

    # ClonalGE + Tumoroscope side-by-side
    python plot_simulated_results.py \\
        --tumoroscope_dir /Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope \\
        --output plots_paper/figure_simulated_comparison.png

Options:
    --results_dir       ClonalGE base directory                   (default: /Volumes/LenovoPS8/ClonalGE/Results_simulated_data)
    --tumoroscope_dir   Tumoroscope base directory (optional)
    --num_runs          Number of run folders (1..N)              (default: 20)
    --outdir            Output directory                          (default: plots_paper)
    --output            Output filename                           (default: figure_simulated.png)
"""

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CONDITIONS   = ['normal', 'low_variance', 'high_coverage']
COND_LABELS  = ['Normal', 'Low variance', 'High coverage']
METRICS_ALL      = ['H_SEE', 'phi_SEE', 'pi_SEE', 'B_SEE']
METRICS_SHARED   = ['H_SEE', 'phi_SEE', 'pi_SEE']   # metrics both models have
METRIC_LABELS_ALL = [
    'H Standard Error\nof the Estimate',
    r'$\phi$ Standard Error''\nof the Estimate',
    r'$\pi$ Standard Error''\nof the Estimate',
    'B Standard Error\nof the Estimate',
]
METRIC_LABELS_SHARED = METRIC_LABELS_ALL[:3]

# Colour per condition
COND_COLORS = ['#4e79a7', '#f28e2b', '#59a14f']

FS_TITLE = 8
FS_LABEL = 7
FS_TICK  = 6.5


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_results_file(path):
    """Return dict {metric: float} from a results_*.txt file."""
    txt = open(path).read()
    out = {}
    for metric, pattern in [
        ('H_SEE',   r'Standard Error of the Estimate H:\n([\d.eE+\-nan]+)'),
        ('phi_SEE', r'Standard Error of the Estimate phi:\n([\d.eE+\-nan]+)'),
        ('pi_SEE',  r'Standard Error of the Estimate pi:\n([\d.eE+\-nan]+)'),
        ('B_SEE',   r'Standard Error of the Estimate B:\n([\d.eE+\-nan]+)'),
        ('n_SEE',   r'Standard Error of the Estimate n:\n([\d.eE+\-nan]+)'),
    ]:
        m = re.search(pattern, txt)
        if m:
            try:
                out[metric] = float(m.group(1))
            except ValueError:
                out[metric] = float('nan')
        else:
            out[metric] = float('nan')
    return out


def load_all_results(results_dir, num_runs):
    """
    Returns data[condition][metric] = list of floats (one per run).
    Runs with missing files are skipped.
    """
    data = {c: {m: [] for m in METRICS_ALL + ['n_SEE']} for c in CONDITIONS}

    for run in range(1, num_runs + 1):
        for cond in CONDITIONS:
            fpath = os.path.join(results_dir, str(run), f'results_{cond}.txt')
            if not os.path.exists(fpath):
                print(f'  MISSING: {fpath}')
                continue
            vals = parse_results_file(fpath)
            for m in METRICS_ALL + ['n_SEE']:
                data[cond][m].append(vals.get(m, float('nan')))

    return data


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_boxplot(ax, vals, position, color, hatch=None):
    """Draw a single boxplot at the given position."""
    if not vals:
        return
    bp = ax.boxplot(
        vals,
        positions=[position],
        widths=0.35,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.5,
                        markerfacecolor=color, markeredgecolor=color,
                        alpha=0.5),
        medianprops=dict(color='black', linewidth=1.2),
        boxprops=dict(facecolor=color, alpha=0.65, linewidth=0.6,
                      hatch=hatch or ''),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
    )


def make_figure(clonalge_data, outpath, tumoroscope_data=None):
    """
    clonalge_data     : dict from load_all_results (always present)
    tumoroscope_data  : dict from load_all_results, or None
    """
    plt.rcParams.update({
        'font.family':    'sans-serif',
        'font.size':       FS_TICK,
        'axes.linewidth':  0.5,
        'pdf.fonttype':    42,
        'svg.fonttype':    'none',
    })

    comparison = tumoroscope_data is not None

    # When comparing: only shared metrics (no B_SEE for Tumoroscope).
    # When ClonalGE-only: all 4 metrics.
    metrics       = METRICS_SHARED       if comparison else METRICS_ALL
    metric_labels = METRIC_LABELS_SHARED if comparison else METRIC_LABELS_ALL

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 3.6))
    if n_metrics == 1:
        axes = [axes]

    n_cond   = len(CONDITIONS)
    group_w  = 1.2 if comparison else 1.0
    offsets  = np.arange(n_cond) * group_w   # centre of each condition group

    for col, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]

        for ci, (cond, color) in enumerate(zip(CONDITIONS, COND_COLORS)):
            if comparison:
                # ClonalGE left, Tumoroscope right within each condition group
                pos_cge = offsets[ci] - 0.22
                pos_tum = offsets[ci] + 0.22
                vals_cge = [v for v in clonalge_data[cond][metric]    if not np.isnan(v)]
                vals_tum = [v for v in tumoroscope_data[cond][metric] if not np.isnan(v)]
                _draw_boxplot(ax, vals_cge, pos_cge, color)
                _draw_boxplot(ax, vals_tum, pos_tum, color, hatch='///')
            else:
                vals = [v for v in clonalge_data[cond][metric] if not np.isnan(v)]
                _draw_boxplot(ax, vals, offsets[ci], color)

        ax.set_xticks(offsets)
        ax.set_xticklabels(COND_LABELS, rotation=30, ha='right',
                           fontsize=FS_TICK - 0.5)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title('abcd'[col], fontsize=FS_TITLE, fontweight='bold',
                     loc='left', pad=3)
        ax.tick_params(axis='y', labelsize=FS_TICK - 0.5, length=2, width=0.4)
        ax.tick_params(axis='x', length=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        xlim_pad = 0.7
        ax.set_xlim(offsets[0] - xlim_pad, offsets[-1] + xlim_pad)

    # Legend: conditions (always) + model style (when comparing)
    legend_handles = [
        mpatches.Patch(facecolor=c, alpha=0.65, edgecolor='grey',
                       linewidth=0.4, label=l)
        for c, l in zip(COND_COLORS, COND_LABELS)
    ]
    if comparison:
        legend_handles += [
            mpatches.Patch(facecolor='white', edgecolor='grey',
                           linewidth=0.6, label='ClonalGE'),
            mpatches.Patch(facecolor='white', edgecolor='grey',
                           linewidth=0.6, hatch='///', label='Tumoroscope'),
        ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.04),
               ncol=len(legend_handles),
               fontsize=FS_LABEL, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    print(f'Saved → {outpath}')
    plt.close()


# ── summary stats ─────────────────────────────────────────────────────────────

def print_summary(data, label=''):
    print(f'\n=== Summary{" — " + label if label else ""} (median [IQR]) ===')
    for cond, clabel in zip(CONDITIONS, COND_LABELS):
        print(f'\n  {clabel}:')
        for metric in METRICS_ALL:
            vals = np.array([v for v in data[cond][metric] if not np.isnan(v)])
            if len(vals) == 0:
                print(f'    {metric}: no data')
                continue
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            print(f'    {metric}: median={med:.5f}  IQR=[{q25:.5f}, {q75:.5f}]  '
                  f'n={len(vals)}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir',      default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_data',
                    help='ClonalGE results directory')
    ap.add_argument('--tumoroscope_dir',  default=None,
                    help='Tumoroscope results directory (optional; enables comparison mode)')
    ap.add_argument('--num_runs',         type=int, default=20)
    ap.add_argument('--outdir',           default='plots_paper')
    ap.add_argument('--output',           default=None,
                    help='Full output path (overrides --outdir)')
    args = ap.parse_args()

    comparison = args.tumoroscope_dir is not None
    default_fname = 'figure_simulated_comparison.png' if comparison else 'figure_simulated.png'
    outpath = args.output or os.path.join(args.outdir, default_fname)

    print(f'Loading ClonalGE results from {args.results_dir} ({args.num_runs} runs)...')
    cge_data = load_all_results(args.results_dir, args.num_runs)
    print_summary(cge_data, 'ClonalGE')

    tum_data = None
    if comparison:
        print(f'\nLoading Tumoroscope results from {args.tumoroscope_dir} ({args.num_runs} runs)...')
        tum_data = load_all_results(args.tumoroscope_dir, args.num_runs)
        print_summary(tum_data, 'Tumoroscope')

    make_figure(cge_data, outpath, tumoroscope_data=tum_data)


if __name__ == '__main__':
    main()
