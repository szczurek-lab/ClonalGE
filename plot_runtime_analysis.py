"""
Runtime analysis for ClonalGE vs Tumoroscope across simulated and real data.

Collects Time: fields from results files and inferred_vars_all.txt (real data),
then produces:
  1. Boxplot: simulated data runtimes by method x condition (seconds)
  2. Bar chart: real-data (figure5) runtimes across 20 runs
  3. Summary statistics printed to stdout

Usage:
    python plot_runtime_analysis.py \\
        --results_1chain   /Volumes/LenovoPS8/ClonalGE/Results_simulated_data \\
        --results_5chains  /Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains \\
        --results_tum      /Volumes/LenvoPS8/ClonalGE/Results_simulated_tumoroscope \\
        --results_real     /Volumes/LenovoPS8/ClonalGE/Results_figure5 \\
        --num_runs 20 \\
        --output plots_paper/figure_runtime.png
"""

import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CONDITIONS  = ['normal', 'low_variance', 'high_coverage']
COND_LABELS = ['Normal', 'Low variance', 'High coverage']


def parse_time(path):
    """Return the Time: value from a results file, or NaN."""
    if not os.path.isfile(path):
        return np.nan
    for line in open(path):
        m = re.match(r'^Time:\s*([\d.eE+\-]+)', line)
        if m:
            return float(m.group(1))
    return np.nan


def load_simulated(base_dir, num_runs):
    """Return dict {condition: [time_run1, ..., time_runN]} (seconds).
    Expects base_dir/{run_num}/results_{cond}.txt layout."""
    data = {c: [] for c in CONDITIONS}
    for run in range(1, num_runs + 1):
        for cond in CONDITIONS:
            path = os.path.join(base_dir, str(run), f'results_{cond}.txt')
            data[cond].append(parse_time(path))
    return data


def load_simulated_flat(base_dir, num_runs):
    """Return dict {condition: [time_run1, ..., time_runN]} (seconds).
    Expects base_dir + str(run_num) /results_{cond}.txt layout
    (e.g. Results_simulated_tumoroscope1/results_normal.txt)."""
    data = {c: [] for c in CONDITIONS}
    for run in range(1, num_runs + 1):
        for cond in CONDITIONS:
            path = os.path.join(f'{base_dir}{run}', f'results_{cond}.txt')
            data[cond].append(parse_time(path))
    return data


def load_real(base_dir, num_runs):
    """Return list of runtimes from inferred_vars_all.txt (real prostate data)."""
    times = []
    for run in range(1, num_runs + 1):
        path = os.path.join(f'{base_dir}_{run}', 'inferred_vars_all.txt')
        times.append(parse_time(path))
    return times


def minutes(seconds_array):
    return np.array(seconds_array, dtype=float) / 60.0


def print_stats(label, times_sec):
    arr = np.array(times_sec, dtype=float)
    arr = arr[~np.isnan(arr)]
    print(f"\n{label}")
    if len(arr) == 0:
        print("  no data")
        return
    arr_min = arr / 60.0
    print(f"  n={len(arr_min)}, "
          f"mean={arr_min.mean():.1f} min, "
          f"median={np.median(arr_min):.1f} min, "
          f"std={arr_min.std():.1f} min, "
          f"min={arr_min.min():.1f} min, "
          f"max={arr_min.max():.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_1chain',  default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_data')
    ap.add_argument('--results_5chains', default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains')
    ap.add_argument('--results_tum',     default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope')
    ap.add_argument('--results_real',    default='/Volumes/LenovoPS8/ClonalGE/Results_figure5')
    ap.add_argument('--num_runs',        type=int, default=20)
    ap.add_argument('--output',          default='plots_paper/figure_runtime.png')
    args = ap.parse_args()

    # ── load data ────────────────────────────────────────────────────────────────
    data_1c  = load_simulated(args.results_1chain,  args.num_runs)
    data_5c  = load_simulated(args.results_5chains,  args.num_runs)
    data_tum = load_simulated_flat(args.results_tum,  args.num_runs)
    data_real = load_real(args.results_real, args.num_runs)

    # ── print summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("RUNTIME SUMMARY (minutes)")
    print("=" * 60)
    for cond, label in zip(CONDITIONS, COND_LABELS):
        print_stats(f"Tumoroscope — {label}",       data_tum[cond])
        print_stats(f"ClonalGE 1-chain — {label}",  data_1c[cond])
        print_stats(f"ClonalGE 5-chains — {label}", data_5c[cond])
    print_stats("ClonalGE real data (figure5, 10 chains)", data_real)

    # ── figure setup ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('ClonalGE Runtime Analysis', fontsize=12, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    METHOD_COLORS  = ['#e15759', '#76b7b2', '#4e79a7']
    METHOD_LABELS  = ['Tumoroscope', 'ClonalGE (1 chain)', 'ClonalGE (5 chains)']

    # ── row 1: per-condition boxplots ─────────────────────────────────────────
    for ci, (cond, clabel) in enumerate(zip(CONDITIONS, COND_LABELS)):
        ax = fig.add_subplot(gs[0, ci])

        plot_data = [
            minutes(data_tum[cond]),
            minutes(data_1c[cond]),
            minutes(data_5c[cond]),
        ]

        bp = ax.boxplot(plot_data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1),
                        flierprops=dict(marker='o', markersize=4, alpha=0.6))

        for patch, color in zip(bp['boxes'], METHOD_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_title(clabel, fontsize=9, fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Tum.', 'CGE-1', 'CGE-5'], fontsize=8)
        ax.set_ylabel('Runtime (min)', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ── row 2 left: all conditions combined per method ────────────────────────
    ax_all = fig.add_subplot(gs[1, 0])
    all_tum = minutes([v for c in CONDITIONS for v in data_tum[c]])
    all_1c  = minutes([v for c in CONDITIONS for v in data_1c[c]])
    all_5c  = minutes([v for c in CONDITIONS for v in data_5c[c]])

    bp2 = ax_all.boxplot([all_tum, all_1c, all_5c], patch_artist=True, widths=0.5,
                         medianprops=dict(color='black', linewidth=1.5),
                         whiskerprops=dict(linewidth=1),
                         capprops=dict(linewidth=1),
                         flierprops=dict(marker='o', markersize=4, alpha=0.6))
    for patch, color in zip(bp2['boxes'], METHOD_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_all.set_title('All conditions combined', fontsize=9, fontweight='bold')
    ax_all.set_xticks([1, 2, 3])
    ax_all.set_xticklabels(['Tum.', 'CGE-1', 'CGE-5'], fontsize=8)
    ax_all.set_ylabel('Runtime (min)', fontsize=8)
    ax_all.tick_params(axis='y', labelsize=7)
    ax_all.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ── row 2 middle: real data per-run bar chart ─────────────────────────────
    ax_real = fig.add_subplot(gs[1, 1])
    real_min = minutes(data_real)
    run_ids = list(range(1, args.num_runs + 1))
    colors_bar = ['#4e79a7' if not np.isnan(t) else '#cccccc' for t in real_min]
    ax_real.bar(run_ids, np.nan_to_num(real_min), color=colors_bar, alpha=0.8, edgecolor='white', linewidth=0.5)
    valid_real = real_min[~np.isnan(real_min)]
    if len(valid_real):
        ax_real.axhline(np.mean(valid_real), color='#e15759', linewidth=1.5,
                        linestyle='--', label=f'Mean: {np.mean(valid_real):.0f} min')
        ax_real.legend(fontsize=7)
    ax_real.set_title('Real data (prostate, 10 chains)', fontsize=9, fontweight='bold')
    ax_real.set_xlabel('Run', fontsize=8)
    ax_real.set_ylabel('Runtime (min)', fontsize=8)
    ax_real.set_xticks(run_ids)
    ax_real.set_xticklabels([str(r) for r in run_ids], fontsize=6)
    ax_real.tick_params(axis='y', labelsize=7)
    ax_real.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ── row 2 right: mean runtime bar with error bars ─────────────────────────
    ax_mean = fig.add_subplot(gs[1, 2])

    all_groups = []
    all_labels_bar = []
    all_colors_bar = []

    for cond, clabel in zip(CONDITIONS, COND_LABELS):
        for arr, mlabel, color in zip(
            [data_tum[cond], data_1c[cond], data_5c[cond]],
            ['Tum', 'CGE-1', 'CGE-5'],
            METHOD_COLORS
        ):
            a = minutes(arr)
            a = a[~np.isnan(a)]
            all_groups.append((np.mean(a), np.std(a), color))
            all_labels_bar.append(f'{mlabel}\n{clabel[:4]}')

    xs = np.arange(len(all_groups))
    means = [g[0] for g in all_groups]
    stds  = [g[1] for g in all_groups]
    cols  = [g[2] for g in all_groups]

    ax_mean.bar(xs, means, yerr=stds, color=cols, alpha=0.8,
                capsize=3, error_kw=dict(linewidth=1), edgecolor='white', linewidth=0.5)
    ax_mean.set_xticks(xs)
    ax_mean.set_xticklabels(all_labels_bar, fontsize=5.5, rotation=45, ha='right')
    ax_mean.set_title('Mean ± SD runtime\nby method & condition', fontsize=9, fontweight='bold')
    ax_mean.set_ylabel('Runtime (min)', fontsize=8)
    ax_mean.tick_params(axis='y', labelsize=7)
    ax_mean.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(facecolor=c, alpha=0.8, label=l)
               for c, l in zip(METHOD_COLORS, METHOD_LABELS)]
    fig.legend(handles=handles, loc='upper right', fontsize=8,
               bbox_to_anchor=(0.97, 0.96), framealpha=0.9)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {args.output}")


if __name__ == '__main__':
    main()
