"""
Plot ClonalGE performance under assumption-violating simulation conditions.

Reads H_rMAE and B_rMAE from result .txt files for three variants:
  normal, zinb, batch

Usage:
    python plot_violated_results.py \\
        --base_dir  /Volumes/LenovoPS8/ClonalGE \\
        --num_runs  20 \\
        --output    plots_paper/violated_simulations.png
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = ['normal', 'zinb', 'batch']
VARIANT_LABELS = {
    'normal': 'Standard\n(no violation)',
    'zinb':   'ZINB\n(10% zero-inflation)',
    'batch':  'Batch effects\n(×0.5–2.0)',
}
RESULTS_DIRS = {
    'normal': 'Results_violated_normal',
    'zinb':   'Results_violated_zinb',
    'batch':  'Results_violated_batch',
}
COLOR_H = '#2166ac'
COLOR_B = '#d6604d'

H_PAT = re.compile(r'H_rMAE \(violated\):\s*([\d.eE+\-nan]+)')
B_PAT = re.compile(r'B_rMAE \(violated\):\s*([\d.eE+\-nan]+)')


def read_results(base_dir, variant, num_runs):
    rdir = RESULTS_DIRS[variant]
    rows = []
    for run in range(1, num_runs + 1):
        txt = os.path.join(base_dir, rdir + str(run), 'results_normal.txt')
        if not os.path.exists(txt):
            continue
        text = open(txt).read()
        hm = H_PAT.search(text)
        bm = B_PAT.search(text)
        if hm and bm:
            rows.append(dict(variant=variant,
                             run=run,
                             H_rMAE=float(hm.group(1)),
                             B_rMAE=float(bm.group(1))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', required=True)
    ap.add_argument('--num_runs', type=int, default=20)
    ap.add_argument('--output',   default='plots_paper/violated_simulations.png')
    args = ap.parse_args()

    rows = []
    for v in VARIANTS:
        rows.extend(read_results(args.base_dir, v, args.num_runs))

    if not rows:
        print('No results found. Have the simulations been run?')
        return

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, metric, color, ylabel in zip(
            axes,
            ['H_rMAE', 'B_rMAE'],
            [COLOR_H, COLOR_B],
            ['Relative MAE  (H)', 'Relative MAE  (B)']):

        data = [df[df['variant'] == v][metric].dropna().values for v in VARIANTS]
        positions = range(1, len(VARIANTS) + 1)

        bp = ax.boxplot(data, positions=list(positions), widths=0.5,
                        patch_artist=True,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=3.5,
                                        markerfacecolor='gray', alpha=0.6))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks(list(positions))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('ClonalGE robustness under assumption violations',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
