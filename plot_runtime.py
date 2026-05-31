#!/usr/bin/env python
"""
plot_runtime.py
---------------
Generates the runtime / scalability figure from run_scalability.py output.

Panel A: bar chart — wall-clock time for the three baseline configs
         (normal, low_variance, high_coverage), in seconds / iteration.

Panel B–D: line plots — seconds / iteration vs S, g, I independently
           (baseline value marked with a dashed vertical line).

Usage:
    python plot_runtime.py [--csv PATH] [--output PATH]

Options:
    --csv       Path to runtime_results.csv  (default: results/scalability/runtime_results.csv)
    --output    Output figure path           (default: plots_paper/runtime.png)
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── layout ────────────────────────────────────────────────────────────────────
FS_BASE   = 7
FS_TICK   = 6
FS_LABEL  = 7
FS_TITLE  = 7.5
FIG_W     = 7.0
FIG_H     = 3.2

# baselines (dashed line positions on sweep plots)
S_BASE, G_BASE, I_BASE = 300, 50, 200

# colours
COL_BAR   = '#2c7bb6'
COL_LINE  = {'S': '#d7191c', 'g': '#1a9641', 'I': '#2c7bb6'}
COL_BASE  = '#888888'

CONFIG_LABELS = {
    'normal':        'Normal',
    'low_variance':  'Low variance',
    'high_coverage': 'High coverage',
}


def load(csv_path):
    df = pd.read_csv(csv_path)
    df['sec_per_iter'] = pd.to_numeric(df['sec_per_iter'], errors='coerce')
    df['wall_time_s']  = pd.to_numeric(df['wall_time_s'],  errors='coerce')
    return df


def panel_baselines(ax, df):
    rows = df[df['param_name'] == 'config'].copy()
    rows['config_label'] = rows['param_value'].map(CONFIG_LABELS)
    rows = rows.sort_values('param_value')

    x     = np.arange(len(rows))
    bars  = ax.bar(x, rows['sec_per_iter'], color=COL_BAR,
                   width=0.5, edgecolor='none', zorder=3)

    # value labels on bars
    for bar, val in zip(bars, rows['sec_per_iter']):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + max(rows['sec_per_iter'].dropna()) * 0.02,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=FS_TICK, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(rows['config_label'], fontsize=FS_TICK, rotation=10, ha='right')
    ax.set_ylabel('Seconds / iteration', fontsize=FS_LABEL)
    ax.set_title('Baseline configs', fontsize=FS_TITLE)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.4, color='#dddddd', zorder=0)


def panel_sweep(ax, df, param, baseline_val, xlabel):
    rows = df[df['param_name'] == param].copy()
    rows['param_value'] = pd.to_numeric(rows['param_value'])
    rows = rows.sort_values('param_value').dropna(subset=['sec_per_iter'])

    ax.plot(rows['param_value'], rows['sec_per_iter'],
            color=COL_LINE[param], marker='o', markersize=3.5,
            linewidth=1.2, zorder=3)
    ax.axvline(baseline_val, color=COL_BASE, linewidth=0.8,
               linestyle='--', zorder=2, label=f'baseline ({baseline_val})')

    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel('Seconds / iteration', fontsize=FS_LABEL)
    ax.set_title(f'Scaling with {param}', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.4, color='#dddddd', zorder=0)
    ax.legend(fontsize=FS_TICK - 0.5, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',    default='results/scalability/runtime_results.csv')
    ap.add_argument('--output', default='plots_paper/runtime.png')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f'CSV not found: {args.csv}\n'
                                f'Run run_scalability.py first.')

    df = load(args.csv)

    plt.rcParams.update({
        'font.family':    'sans-serif',
        'font.size':       FS_BASE,
        'axes.linewidth':  0.5,
        'pdf.fonttype':    42,
        'svg.fonttype':    'none',
    })

    fig, axes = plt.subplots(
        1, 4, figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(wspace=0.45, left=0.08, right=0.97,
                         top=0.88, bottom=0.20),
    )

    panel_baselines(axes[0], df)
    panel_sweep(axes[1], df, 'S', S_BASE, 'Number of spots (S)')
    panel_sweep(axes[2], df, 'g', G_BASE, 'Number of genes (g)')
    panel_sweep(axes[3], df, 'I', I_BASE, 'Number of mutations (I)')

    # panel letters
    for ax, letter in zip(axes, 'ABCD'):
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
                fontsize=FS_BASE + 1, fontweight='bold', va='top')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=400, bbox_inches='tight')
    print(f'Saved → {args.output}')
    plt.close()


if __name__ == '__main__':
    main()
