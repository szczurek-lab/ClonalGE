"""
Generate LaTeX table + figure caption for Figure 2.

Usage:
    python generate_fig2_latex.py \
        --tum_dir   /Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope \
        --cge1_dir  /Volumes/LenovoPS8/ClonalGE/Results_simulated_data \
        --cge5_dir  /Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains \
        --best_csv  /Volumes/LenovoPS8/ClonalGE/best_chain_see_5chains.csv \
        --num_runs  20 \
        --out       ClonalGE_revision/fig2_table_caption.tex
"""

import argparse, csv, os, re
import numpy as np
from scipy import stats

CONDITIONS  = ['normal', 'low_variance', 'high_coverage']
COND_LABELS = ['Normal', 'Low variance', 'High coverage']
METRICS     = ['H_SEE', 'phi_SEE', 'n_SEE', 'B_SEE']
METRIC_LABELS = [r'$H$', r'$\phi$', r'$N$', r'$B$']
NUM_RUNS    = 20

# ── loaders (same as plot_simulated_results.py) ───────────────────────────────

def _grep(txt, pat):
    m = re.search(pat + r'\n([\d.eE+\-nan]+)', txt)
    return float(m.group(1)) if m else float('nan')


def load_tum(tum_dir):
    data = {c: {k: [] for k in ['H_SEE','phi_SEE','n_SEE','B_SEE','B_SEE_NB','mean_H','mean_phi','mean_n','mean_B']}
            for c in CONDITIONS}
    for run in range(1, NUM_RUNS + 1):
        for cond in CONDITIONS:
            f = f'{tum_dir.rstrip("/")}{run}/results_{cond}.txt'
            txt = open(f).read() if os.path.exists(f) else ''
            data[cond]['H_SEE'   ].append(_grep(txt, r'Standard Error of the Estimate H:'))
            data[cond]['phi_SEE' ].append(_grep(txt, r'Standard Error of the Estimate phi:'))
            data[cond]['n_SEE'   ].append(_grep(txt, r'Standard Error of the Estimate n:'))
            data[cond]['B_SEE'   ].append(_grep(txt, r'Standard Error of the Estimate B \(Tumoroscope\+LR\):'))
            data[cond]['B_SEE_NB'].append(_grep(txt, r'Standard Error of the Estimate B \(Tumoroscope\+NB\):'))
            data[cond]['mean_H'  ].append(_grep(txt, r'Mean of true H:'))
            data[cond]['mean_phi'].append(_grep(txt, r'Mean of true phi:'))
            data[cond]['mean_n'  ].append(_grep(txt, r'Mean of true n:'))
            data[cond]['mean_B'  ].append(_grep(txt, r'Mean of true B:'))
    return data


def load_cge1(cge1_dir):
    data = {c: {k: [] for k in ['H_SEE','phi_SEE','n_SEE','B_SEE']} for c in CONDITIONS}
    for run in range(1, NUM_RUNS + 1):
        for cond in CONDITIONS:
            f = os.path.join(cge1_dir, str(run), f'results_{cond}.txt')
            txt = open(f).read() if os.path.exists(f) else ''
            data[cond]['H_SEE'  ].append(_grep(txt, r'Standard Error of the Estimate H:'))
            data[cond]['phi_SEE'].append(_grep(txt, r'Standard Error of the Estimate phi:'))
            data[cond]['n_SEE'  ].append(_grep(txt, r'Standard Error of the Estimate n:'))
            data[cond]['B_SEE'  ].append(_grep(txt, r'Standard Error of the Estimate B:'))
    return data


def load_cge5_mean(cge5_dir):
    data = {c: {k: [] for k in ['mean_H','mean_phi','mean_n','mean_B']} for c in CONDITIONS}
    for run in range(1, NUM_RUNS + 1):
        for cond in CONDITIONS:
            f = os.path.join(cge5_dir, str(run), f'results_{cond}.txt')
            txt = open(f).read() if os.path.exists(f) else ''
            data[cond]['mean_H'  ].append(_grep(txt, r'Mean of true H:'))
            data[cond]['mean_phi'].append(_grep(txt, r'Mean of true phi:'))
            data[cond]['mean_n'  ].append(_grep(txt, r'Mean of true n:'))
            data[cond]['mean_B'  ].append(_grep(txt, r'Mean of true B:'))
    return data


def load_best_csv(csv_path):
    data = {c: {k: [float('nan')] * NUM_RUNS for k in ['H_SEE','phi_SEE','n_SEE','B_SEE']}
            for c in CONDITIONS}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            run = int(row['run']) - 1
            cond = row['cond']
            if cond in data:
                for k in ['H_SEE','phi_SEE','n_SEE','B_SEE']:
                    try: data[cond][k][run] = float(row[k])
                    except (KeyError, ValueError): pass
    return data


# ── rMAE & stats ──────────────────────────────────────────────────────────────

def rmae(see, denom):
    s, d = np.array(see, float), np.array(denom, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s / d


def wilcoxon_less(a, b):
    """P(a < b) one-sided signed-rank."""
    a, b = np.array(a), np.array(b)
    mask = np.isfinite(a) & np.isfinite(b) & (a != b)
    if mask.sum() < 5:
        return float('nan')
    _, p = stats.wilcoxon(a[mask], b[mask], alternative='less')
    return p


def star(p):
    if np.isnan(p): return ''
    if p < 0.001:   return r'^{***}'
    if p < 0.01:    return r'^{**}'
    if p < 0.05:    return r'^{*}'
    return ''


def fmt(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v): return '---'
    med     = np.median(v)
    q1, q3  = np.percentile(v, [25, 75])
    return rf'{med:.3f} [{q1:.3f}, {q3:.3f}]'


def fmt_bold(vals, is_best):
    s = fmt(vals)
    return rf'\textbf{{{s}}}' if is_best else s


# ── LaTeX builder ─────────────────────────────────────────────────────────────

def build_latex(rmae_all, out_path):
    """
    rmae_all[method][cond][metric] = array of rMAE values
    methods: 'LR', 'NB', 'CGE1', 'CGE5'
    """
    methods   = ['LR',  'NB',  'CGE1', 'CGE5']
    m_labels  = [r'Tum$+$LR', r'Tum$+$NB',
                 r'\textsc{ClonalGE} (1-chain)',
                 r'\textsc{ClonalGE} (5-chain)']

    lines = []

    # ── Figure caption ────────────────────────────────────────────────────────
    lines += [
        r'% ── Figure 2 caption ──────────────────────────────────────────────',
        r'\begin{figure}[htbp]',
        r'\centering',
        r'\includegraphics[width=\linewidth]{figures/Fig2_simulation.pdf}',
        r'\caption{\textbf{Validation on simulated data.}',
        r'Relative mean absolute error (rMAE) of inferred model variables across',
        r'20 independent simulation replicates under three conditions (Normal,',
        r'Low variance, High coverage). Each panel shows rMAE for one variable:',
        r'(\textbf{a})~clonal composition $H$,',
        r'(\textbf{b})~overdispersion $\phi$,',
        r'(\textbf{c})~cell count $N$, and',
        r'(\textbf{d})~clone-specific gene expression $B$.',
        r'Four methods are compared: Tumoroscope with linear-regression',
        r'post-processing (Tum$+$LR), Tumoroscope with Negative Binomial',
        r'regression post-processing (Tum$+$NB), \textsc{ClonalGE} run as a',
        r'single Markov chain, and \textsc{ClonalGE} run as five independent',
        r'chains with the best chain selected by log-likelihood.',
        r'Tum$+$NB is only evaluated on $B$ (panels a--c are identical to',
        r'Tum$+$LR). Boxes show the interquartile range; horizontal lines',
        r'indicate the median; whiskers extend to 1.5\,IQR.',
        r'}',
        r'\label{fig:simulated}',
        r'\end{figure}',
        '',
    ]

    # ── Table ─────────────────────────────────────────────────────────────────
    lines += [
        r'% ── Table: Figure 2 summary statistics ────────────────────────────',
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{4pt}',
        r'\caption{%',
        r'  Median rMAE [IQR] for all inferred variables across 20 simulation',
        r'  replicates ($n=20$ per condition). Bold: lowest median in each',
        r'  condition--metric group.',
        r'  Significance of \textsc{ClonalGE} improvement over Tum$+$LR',
        r'  (one-sided Wilcoxon signed-rank test):',
        r'  $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.',
        r'  Tum$+$NB is evaluated on $B$ only; $H$, $\phi$, $N$ are',
        r'  identical to Tum$+$LR (---).%',
        r'}',
        r'\label{tab:simulated_summary}',
        r'\begin{tabular}{ll' + 'c' * len(METRICS) + '}',
        r'\toprule',
        r'Condition & Method & '
            + ' & '.join(METRIC_LABELS) + r' \\',
        r'\midrule',
    ]

    for ci, (cond, clbl) in enumerate(zip(CONDITIONS, COND_LABELS)):
        if ci > 0:
            lines.append(r'\midrule')
        for mi, (mkey, mlbl) in enumerate(zip(methods, m_labels)):
            # find best (lowest median) per metric for this condition
            best = {}
            for metric in METRICS:
                medians = {m: np.nanmedian(rmae_all[m][cond][metric]) for m in methods}
                best[metric] = min(medians, key=medians.get)

            cells = []
            for metric in METRICS:
                vals = rmae_all[mkey][cond][metric]
                is_nb_nonB = (mkey == 'NB') and (metric != 'B_SEE')
                if is_nb_nonB:
                    cells.append('---')
                    continue
                is_best = (best[metric] == mkey)
                cell = fmt_bold(vals, is_best)
                # significance vs Tum+LR for CGE methods
                if mkey in ('CGE1', 'CGE5'):
                    p = wilcoxon_less(vals, rmae_all['LR'][cond][metric])
                    s = star(p)
                    if s:
                        cell += f'${s}$'
                cells.append(cell)

            cond_col = (rf'\multirow{{4}}{{*}}{{{clbl}}}' if mi == 0 else '')
            lines.append(f'{cond_col} & {mlbl} & ' + ' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'LaTeX saved → {out_path}')
    print('\n' + '\n'.join(lines))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tum_dir',  default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_tumoroscope')
    ap.add_argument('--cge1_dir', default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_data')
    ap.add_argument('--cge5_dir', default='/Volumes/LenovoPS8/ClonalGE/Results_simulated_5chains')
    ap.add_argument('--best_csv', default='/Volumes/LenovoPS8/ClonalGE/best_chain_see_5chains.csv')
    ap.add_argument('--num_runs', type=int, default=20)
    ap.add_argument('--out',      default='ClonalGE_revision/fig2_table_caption.tex')
    args = ap.parse_args()

    global NUM_RUNS
    NUM_RUNS = args.num_runs

    tum   = load_tum(args.tum_dir)
    cge1  = load_cge1(args.cge1_dir)
    cge5m = load_cge5_mean(args.cge5_dir)
    cge5b = load_best_csv(args.best_csv)

    # canonical denominators from 5-chain txt
    denom = {cond: {
        'H_SEE':   cge5m[cond]['mean_H'],
        'phi_SEE': cge5m[cond]['mean_phi'],
        'n_SEE':   cge5m[cond]['mean_n'],
        'B_SEE':   cge5m[cond]['mean_B'],
    } for cond in CONDITIONS}

    rmae_all = {}
    for mkey, see_src, metric_map in [
        ('LR',   tum,   {'H_SEE':'H_SEE','phi_SEE':'phi_SEE','n_SEE':'n_SEE','B_SEE':'B_SEE'}),
        ('NB',   tum,   {'H_SEE':'H_SEE','phi_SEE':'phi_SEE','n_SEE':'n_SEE','B_SEE':'B_SEE_NB'}),
        ('CGE1', cge1,  {'H_SEE':'H_SEE','phi_SEE':'phi_SEE','n_SEE':'n_SEE','B_SEE':'B_SEE'}),
        ('CGE5', cge5b, {'H_SEE':'H_SEE','phi_SEE':'phi_SEE','n_SEE':'n_SEE','B_SEE':'B_SEE'}),
    ]:
        rmae_all[mkey] = {}
        for cond in CONDITIONS:
            rmae_all[mkey][cond] = {}
            for metric, src_key in metric_map.items():
                rmae_all[mkey][cond][metric] = rmae(
                    see_src[cond][src_key], denom[cond][metric])

    build_latex(rmae_all, args.out)


if __name__ == '__main__':
    main()
