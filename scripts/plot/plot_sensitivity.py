"""
Plot hyperparameter sensitivity results.

Accepts either:
  --csv    a single combined CSV (from run_sensitivity.py)
  --dir    a directory of per-task CSVs (from run_sensitivity_one.py / cluster run)

Usage:
    python plot_sensitivity.py \\
        --csv  /path/to/sensitivity_results/sensitivity_results.csv \\
        --output plots_paper/sensitivity_analysis.png

    python plot_sensitivity.py \\
        --dir  /path/to/sensitivity_results \\
        --output plots_paper/sensitivity_analysis.png
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAM_LABELS = {
    'alpha_g': r'$\alpha_g$ (B shape)',
    'beta':    r'$\beta$ (B rate)',
    'p_g':     r'$p_g$ (NB probability)',
}
METRIC_LABELS = {
    'H_rMAE': 'Relative MAE  ($H$)',
    'B_rMAE': 'Relative MAE  ($B$)',
}
COLOR_H = '#2166ac'
COLOR_B = '#d6604d'
FACTORS = [0.50, 0.75, 1.00, 1.25, 1.50]
FACTOR_LABELS = ['−50%', '−25%', '0%', '+25%', '+50%']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',    default=None, help='Combined CSV from run_sensitivity.py')
    ap.add_argument('--dir',    default=None, help='Directory of per-task CSVs from cluster run')
    ap.add_argument('--output', default='plots_paper/sensitivity_analysis.png')
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.dir:
        parts = [pd.read_csv(f) for f in glob.glob(f'{args.dir}/task*.csv')]
        if not parts:
            print(f'No task*.csv files found in {args.dir}')
            return
        df = pd.concat(parts, ignore_index=True)
    else:
        print('Provide --csv or --dir')
        return
    params = ['alpha_g', 'beta', 'p_g']

    fig, axes = plt.subplots(len(params), 2, figsize=(9, 3.5 * len(params)))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for row_idx, param in enumerate(params):
        sub = df[df['param'] == param]

        for col_idx, metric in enumerate(['H_rMAE', 'B_rMAE']):
            ax = axes[row_idx, col_idx]
            color = COLOR_H if metric == 'H_rMAE' else COLOR_B

            data_by_factor = [sub[sub['factor'] == f][metric].dropna().values
                              for f in FACTORS]

            bp = ax.boxplot(data_by_factor, patch_artist=True,
                            medianprops=dict(color='white', linewidth=2),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker='o', markersize=3.5,
                                            markerfacecolor='gray', alpha=0.6))
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            # highlight baseline
            baseline_idx = FACTORS.index(1.0) + 1  # 1-indexed
            ax.axvline(baseline_idx, color='#555555', linestyle='--',
                       linewidth=0.8, alpha=0.7)

            ax.set_xticks(range(1, len(FACTORS) + 1))
            ax.set_xticklabels(FACTOR_LABELS, fontsize=9)
            ax.set_xlabel('Perturbation factor', fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_title(f'{PARAM_LABELS[param]}', fontsize=10, fontweight='bold')
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
