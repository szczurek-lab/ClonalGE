"""
Publication-quality figure and LaTeX table comparing:
  Tumoroscope+LR | Tumoroscope+NB | ClonalGE (1-chain) | ClonalGE (5-chain best)
on B rMAE across three simulation conditions.

Usage:
    python plot_nb_comparison.py \
        --tum_dir   /Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope \
        --cge1_dir  /Volumes/LenovoPS8/ClonalGE/Results_simulated_data \
        --cge5_dir  /Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains \
        --best_csv  /Volumes/LenovoPS8/ClonalGE/best_chain_see_5chains.csv \
        --num_runs  20 \
        --out_fig   plots_paper/figure_nb_comparison.pdf \
        --out_tex   ClonalGE_revision/supplementary_table_nb.tex
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import argparse, csv, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── constants ─────────────────────────────────────────────────────────────────

CONDITIONS  = ['normal', 'low_variance', 'high_coverage']
COND_LABELS = ['Normal', 'Low variance', 'High coverage']
NUM_RUNS    = 20

# Palette: colourblind-safe (Wong 2011)
C_LR  = '#D55E00'   # vermillion
C_NB  = '#E69F00'   # orange
C_1   = '#0072B2'   # blue
C_5   = '#009E73'   # teal-green

METHODS = [
    ('Tum+LR',  C_LR,  '////',  r'Tum$+$LR'),
    ('Tum+NB',  C_NB,  'xxxx',  r'Tum$+$NB'),
    ('CGE-1',   C_1,   '',      r'\textsc{ClonalGE} (1-chain)'),
    ('CGE-5',   C_5,   '....',  r'\textsc{ClonalGE} (5-chain)'),
]

# ── I/O helpers ───────────────────────────────────────────────────────────────

def _grep(txt, pattern):
    m = re.search(pattern + r'\n([\d.eE+\-nan]+)', txt)
    return float(m.group(1)) if m else float('nan')


def load_tum(tum_dir, num_runs):
    """Returns data[cond] = dict of lists: B_LR, B_NB, mean_B."""
    out = {c: {'B_LR': [], 'B_NB': [], 'mean_B': []} for c in CONDITIONS}
    for run in range(1, num_runs + 1):
        for cond in CONDITIONS:
            f = f'{tum_dir.rstrip("/")}{run}/results_{cond}.txt'
            if not os.path.exists(f):
                for k in out[cond]: out[cond][k].append(float('nan'))
                continue
            txt = open(f).read()
            out[cond]['B_LR'  ].append(_grep(txt, r'Standard Error of the Estimate B \(Tumoroscope\+LR\):'))
            out[cond]['B_NB'  ].append(_grep(txt, r'Standard Error of the Estimate B \(Tumoroscope\+NB\):'))
            out[cond]['mean_B'].append(_grep(txt, r'Mean of true B:'))
    return out


def load_cge1(cge1_dir, num_runs):
    out = {c: {'B_SEE': [], 'mean_B': []} for c in CONDITIONS}
    for run in range(1, num_runs + 1):
        for cond in CONDITIONS:
            f = os.path.join(cge1_dir, str(run), f'results_{cond}.txt')
            if not os.path.exists(f):
                for k in out[cond]: out[cond][k].append(float('nan'))
                continue
            txt = open(f).read()
            out[cond]['B_SEE' ].append(_grep(txt, r'Standard Error of the Estimate B:'))
            out[cond]['mean_B'].append(_grep(txt, r'Mean of true B:'))
    return out


def load_cge5_txt(cge5_dir, num_runs):
    out = {c: {'mean_B': []} for c in CONDITIONS}
    for run in range(1, num_runs + 1):
        for cond in CONDITIONS:
            f = os.path.join(cge5_dir, str(run), f'results_{cond}.txt')
            if not os.path.exists(f):
                out[cond]['mean_B'].append(float('nan'))
                continue
            txt = open(f).read()
            out[cond]['mean_B'].append(_grep(txt, r'Mean of true B:'))
    return out


def load_best_csv(csv_path, num_runs):
    out = {c: {'B_SEE': [float('nan')] * num_runs} for c in CONDITIONS}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            run  = int(row['run']) - 1
            cond = row['cond']
            if cond in out:
                try:
                    out[cond]['B_SEE'][run] = float(row['B_SEE'])
                except (KeyError, ValueError):
                    pass
    return out


# ── rMAE ─────────────────────────────────────────────────────────────────────

def rmae(see_list, denom_list):
    see   = np.array(see_list,   dtype=float)
    denom = np.array(denom_list, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (see / denom).tolist()


# ── plotting ──────────────────────────────────────────────────────────────────

def _box(ax, data, pos, color, hatch, alpha=0.82, width=0.18):
    data = [v for v in data if np.isfinite(v)]
    if not data:
        return
    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=width,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.2, alpha=0.55,
                        markerfacecolor=color, markeredgecolor=color,
                        linewidth=0.4),
        medianprops=dict(color='black', linewidth=1.4),
        boxprops=dict(facecolor=color, alpha=alpha, linewidth=0.6, hatch=hatch),
        whiskerprops=dict(linewidth=0.6, linestyle='--', color='#555555'),
        capprops=dict(linewidth=0.6, color='#555555'),
    )
    return bp


def make_figure(rmae_data, outpath):
    """
    rmae_data[cond][method_key] = list of rMAE values.
    method_key ∈ {'Tum+LR', 'Tum+NB', 'CGE-1', 'CGE-5'}
    """
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':           7,
        'axes.linewidth':      0.5,
        'xtick.major.width':   0.5,
        'ytick.major.width':   0.5,
        'pdf.fonttype':        42,
        'svg.fonttype':        'none',
        'figure.dpi':          300,
    })

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.4))

    n_cond  = len(CONDITIONS)
    group_w = 1.0
    offsets = np.arange(n_cond, dtype=float) * group_w

    # 4 boxes per group, centred
    dx = np.array([-0.285, -0.095, 0.095, 0.285])
    mkeys = [m[0] for m in METHODS]

    for ci, cond in enumerate(CONDITIONS):
        for mi, (mkey, col, hat, _) in enumerate(METHODS):
            vals = rmae_data[cond][mkey]
            _box(ax, vals, offsets[ci] + dx[mi], col, hat)

    # axis cosmetics
    ax.set_xticks(offsets)
    ax.set_xticklabels(COND_LABELS, fontsize=7.5)
    ax.set_ylabel(r'$B$ rMAE', fontsize=8)
    ax.set_xlim(offsets[0] - 0.55, offsets[-1] + 0.55)
    ax.tick_params(axis='y', labelsize=7, length=2.5, width=0.5)
    ax.tick_params(axis='x', length=0)
    ax.axhline(0, color='#aaaaaa', linewidth=0.4, linestyle=':', zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_linewidth(0.5)

    # group separators
    for x in offsets[:-1] + group_w / 2:
        ax.axvline(x, color='#dddddd', linewidth=0.5, zorder=0)

    # legend
    handles = [
        mpatches.Patch(facecolor=col, alpha=0.82, edgecolor='#444444',
                       linewidth=0.5, hatch=hat,
                       label={'Tum+LR':  r'Tum$+$LR',
                              'Tum+NB':  r'Tum$+$NB',
                              'CGE-1':   r'ClonalGE (1-chain)',
                              'CGE-5':   r'ClonalGE (5-chain)'}[mkey])
        for mkey, col, hat, _ in METHODS
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=6.2,
              frameon=True, framealpha=0.9, edgecolor='#cccccc',
              borderpad=0.5, labelspacing=0.3, handlelength=1.4,
              handletextpad=0.4, ncol=2)

    plt.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    print(f'Figure saved → {outpath}')
    plt.close()


# ── LaTeX table ───────────────────────────────────────────────────────────────

def wilcoxon_p(a, b):
    """One-sided Wilcoxon signed-rank: P(a < b), i.e. a is better."""
    a, b = np.array(a), np.array(b)
    mask = np.isfinite(a) & np.isfinite(b) & (a != b)
    if mask.sum() < 5:
        return float('nan')
    _, p = stats.wilcoxon(a[mask], b[mask], alternative='less')
    return p


def fmt_median_iqr(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) == 0:
        return '---'
    med        = np.median(v)
    q1, q3     = np.percentile(v, [25, 75])
    return rf'{med:.3f} [{q1:.3f}, {q3:.3f}]'


def star(p):
    if np.isnan(p): return ''
    if p < 0.001:   return r'$^{***}$'
    if p < 0.01:    return r'$^{**}$'
    if p < 0.05:    return r'$^{*}$'
    return r'$^{\mathrm{ns}}$'


def make_table(rmae_data, outpath):
    mkeys  = [m[0] for m in METHODS]
    mlabel = {m[0]: m[3] for m in METHODS}

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(
        r'\caption{Comparison of $B$ rMAE (median [IQR], $n = 20$ simulations) '
        r'across simulation conditions. '
        r'$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (one-sided Wilcoxon signed-rank '
        r'test, ClonalGE vs.\ Tum$+$NB).}'
    )
    lines.append(r'\label{tab:nb_comparison}')
    lines.append(r'\setlength{\tabcolsep}{5pt}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'Method & Normal & Low variance & High coverage \\')
    lines.append(r'\midrule')

    for mkey, _, _, mlbl in METHODS:
        row_cells = [mlbl]
        for cond in CONDITIONS:
            cell = fmt_median_iqr(rmae_data[cond][mkey])
            # add significance star vs NB for CGE methods
            if mkey in ('CGE-1', 'CGE-5'):
                p = wilcoxon_p(rmae_data[cond][mkey], rmae_data[cond]['Tum+NB'])
                cell += star(p)
            row_cells.append(cell)
        lines.append(' & '.join(row_cells) + r' \\')
        if mkey == 'Tum+NB':
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'LaTeX table saved → {outpath}')
    print('\n' + '\n'.join(lines))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tum_dir',  default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope')
    ap.add_argument('--cge1_dir', default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_data')
    ap.add_argument('--cge5_dir', default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains')
    ap.add_argument('--best_csv', default='/Volumes/LenovoPS8/ClonalGE/best_chain_see_5chains.csv')
    ap.add_argument('--num_runs', type=int, default=20)
    ap.add_argument('--out_fig',  default='plots_paper/figure_nb_comparison.pdf')
    ap.add_argument('--out_tex',  default='ClonalGE_revision/supplementary_table_nb.tex')
    args = ap.parse_args()

    print('Loading Tumoroscope results ...')
    tum   = load_tum(args.tum_dir, args.num_runs)
    print('Loading ClonalGE 1-chain results ...')
    cge1  = load_cge1(args.cge1_dir, args.num_runs)
    print('Loading ClonalGE 5-chain (mean_B denominators) ...')
    cge5t = load_cge5_txt(args.cge5_dir, args.num_runs)
    print('Loading best-chain SEE CSV ...')
    cge5b = load_best_csv(args.best_csv, args.num_runs)

    # Use 5-chain mean_B as canonical denominator (same simulation data)
    rmae_data = {}
    for cond in CONDITIONS:
        denom = cge5t[cond]['mean_B']          # from 5-chain txt files
        rmae_data[cond] = {
            'Tum+LR': rmae(tum[cond]['B_LR'],      denom),
            'Tum+NB': rmae(tum[cond]['B_NB'],      denom),
            'CGE-1':  rmae(cge1[cond]['B_SEE'],    denom),
            'CGE-5':  rmae(cge5b[cond]['B_SEE'],   denom),
        }

    print('\n=== rMAE summary ===')
    for cond, clbl in zip(CONDITIONS, COND_LABELS):
        print(f'\n{clbl}:')
        for mkey, _, _, _ in METHODS:
            v = np.array([x for x in rmae_data[cond][mkey] if np.isfinite(x)])
            print(f'  {mkey:8s}: median={np.median(v):.4f}  IQR=[{np.percentile(v,25):.4f}, {np.percentile(v,75):.4f}]  n={len(v)}')

    make_figure(rmae_data, args.out_fig)
    make_table(rmae_data,  args.out_tex)


if __name__ == '__main__':
    main()
