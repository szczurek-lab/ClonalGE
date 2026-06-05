"""
Run selection and loglik trace figures for all three approaches,
prostate 20-run analysis.

Approach 1 — highest log-likelihood run only
Approach 2 — consensus: reference that maximises agreement count
Approach 3 — consensus anchored to highest-loglik run

Outputs per approach in plots_paper/prostate_20runs/approach_{1,2,3}/:
    run_selection.png   — (a) best-chain logliks  (b) Pearson r vs ref run
    loglik_traces.png   — 5×4 grid, all chains per run, selected runs highlighted

Usage:
    python plot_prostate_run_selection.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import os, sys, types, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

# ── numpy 2.x stub so chain pickles load on numpy 1.x ────────────────────────
import numpy.core.multiarray as _ma
import numpy.core.numeric    as _num
_fake = types.ModuleType('numpy._core')
_fake.multiarray = _ma
_fake.numeric    = _num
_fake.umath      = sys.modules.get('numpy.core.umath', np)
sys.modules.update({'numpy._core': _fake,
                    'numpy._core.multiarray': _ma,
                    'numpy._core.numeric':    _num})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── constants ─────────────────────────────────────────────────────────────────
PREFIX      = '/Volumes/LenovoPS8/ClonalGE/Results_figure5_'
RUNS        = list(range(1, 21))
CHAIN_COUNT = 10
SECTION     = 'all'
THRESHOLD   = 0.9
BASE_OUTDIR = 'plots_paper/prostate_20runs'

CHAIN_COLORS = list(plt.cm.tab10.colors)   # 10 distinct colours, index = chain id
COLOR_BEST   = '#e15759'                   # red  — best chain within a run
COLOR_REF    = '#e15759'                   # red  — reference run bar
COLOR_SEL    = '#59a14f'                   # green — selected run bar
COLOR_REJ    = '#bab0ac'                   # grey  — rejected run bar

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7,
                     'axes.linewidth': 0.5, 'pdf.fonttype': 42})


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (done once, shared across approaches)
# ══════════════════════════════════════════════════════════════════════════════

print('Loading chain objects...')
all_logliks = {}   # run → [last_loglik × CHAIN_COUNT]
all_traces  = {}   # run → [loglik_trace × CHAIN_COUNT]
all_H       = {}   # run → [inferred_H   × CHAIN_COUNT]

for run in RUNS:
    lls, traces, Hs = [], [], []
    for cc in range(CHAIN_COUNT):
        path = f'{PREFIX}{run}/inferred_vars_{SECTION}_chain_{cc}'
        try:
            c = pickle.load(open(path, 'rb'))
            lls.append(c.last_loglik)
            traces.append(c.loglik[:c.iter_current])
            Hs.append(c.inferred_H)
        except FileNotFoundError:
            print(f'  Run {run} chain {cc}: NOT FOUND')
    all_logliks[run] = lls
    all_traces[run]  = traces
    all_H[run]       = Hs
    bc = int(np.argmax(lls))
    print(f'  Run {run:2d}: best_chain={bc}  loglik={lls[bc]:.2f}')

# Within-run selection: average H over chains that agree with best-loglik chain
def within_run_H(run):
    lls  = all_logliks[run]
    Hs   = all_H[run]
    best = int(np.argmax(lls))
    ref  = Hs[best].flatten()
    sel  = [i for i, H in enumerate(Hs)
            if i == best or pearsonr(ref, H.flatten())[0] > THRESHOLD]
    return np.mean([Hs[i] for i in sel], axis=0), best, len(sel)

H_avg       = {}
best_chain  = {}
n_kept      = {}
for run in RUNS:
    H_avg[run], best_chain[run], n_kept[run] = within_run_H(run)

logliks_best = {r: max(all_logliks[r]) for r in RUNS}


# ══════════════════════════════════════════════════════════════════════════════
# Alignment helper
# ══════════════════════════════════════════════════════════════════════════════

def align_H(H_ref, H_other):
    K    = H_ref.shape[1]
    cost = np.array([[-pearsonr(H_ref[:, i], H_other[:, j])[0]
                      for j in range(K)] for i in range(K)])
    _, perm = linear_sum_assignment(cost)
    return H_other[:, perm]

def pearson_vs_ref(ref_run):
    """Pearson r of every run's averaged-H (aligned) vs ref_run."""
    r_vals = {}
    for r in RUNS:
        if r == ref_run:
            r_vals[r] = 1.0
        else:
            H_al = align_H(H_avg[ref_run], H_avg[r])
            r_vals[r], _ = pearsonr(H_avg[ref_run].flatten(), H_al.flatten())
    return r_vals


# ══════════════════════════════════════════════════════════════════════════════
# Selection logic for each approach
# ══════════════════════════════════════════════════════════════════════════════

def approach1_selection():
    """Single best-loglik run; no similarity threshold needed."""
    ref = max(logliks_best, key=lambda r: logliks_best[r])
    return ref, [ref], pearson_vs_ref(ref)

def approach2_selection():
    """Reference that maximises number of runs with r >= THRESHOLD."""
    best_ref, best_count, best_mean_r = None, -1, -1.0
    best_r_vals = None
    for cand in RUNS:
        r_vals  = pearson_vs_ref(cand)
        sel     = [r for r in RUNS if r_vals[r] >= THRESHOLD]
        mean_r  = float(np.mean([r_vals[r] for r in sel]))
        if (len(sel) > best_count or
                (len(sel) == best_count and mean_r > best_mean_r)):
            best_ref, best_count, best_mean_r = cand, len(sel), mean_r
            best_r_vals = r_vals
    selected = [r for r in RUNS if best_r_vals[r] >= THRESHOLD]
    return best_ref, selected, best_r_vals

def approach3_selection():
    """Highest-loglik run as reference; select all with r >= THRESHOLD."""
    ref    = max(logliks_best, key=lambda r: logliks_best[r])
    r_vals = pearson_vs_ref(ref)
    selected = [r for r in RUNS if r_vals[r] >= THRESHOLD]
    return ref, selected, r_vals


# ══════════════════════════════════════════════════════════════════════════════
# Figure generators
# ══════════════════════════════════════════════════════════════════════════════

def plot_run_selection(approach_num, ref_run, selected, r_vals, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    x = np.arange(len(RUNS))

    def bar_color(r):
        if r == ref_run:   return COLOR_REF
        if r in selected:  return COLOR_SEL
        return COLOR_REJ

    # panel a: best-chain logliks
    ax = axes[0]
    ax.bar(x, [logliks_best[r] for r in RUNS],
           color=[bar_color(r) for r in RUNS],
           width=0.7, linewidth=0.4, edgecolor='white')
    best_pos = RUNS.index(ref_run)
    ax.annotate(f'Run {ref_run}',
                xy=(best_pos, logliks_best[ref_run]),
                xytext=(best_pos + 1.8, logliks_best[ref_run] + 200),
                fontsize=6, color=COLOR_REF,
                arrowprops=dict(arrowstyle='->', color=COLOR_REF, lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RUNS], fontsize=6)
    ax.set_xlabel('Run', fontsize=7)
    ax.set_ylabel('Best-chain log-likelihood', fontsize=7)
    ax.set_title('a', fontsize=8, fontweight='bold', loc='left', pad=3)
    ax.tick_params(labelsize=6, length=2, width=0.4); ax.tick_params(axis='x', length=0)
    for sp in ax.spines.values(): sp.set_linewidth(0.5)

    # panel b: Pearson r vs reference run
    ax = axes[1]
    ax.bar(x, [r_vals[r] for r in RUNS],
           color=[bar_color(r) for r in RUNS],
           width=0.7, linewidth=0.4, edgecolor='white')
    ax.axhline(THRESHOLD, color='black', lw=0.9, linestyle='--',
               label=f'threshold = {THRESHOLD}')
    for r in selected:
        pos = RUNS.index(r)
        ax.text(pos, r_vals[r] + 0.02, f'{r_vals[r]:.2f}',
                ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RUNS], fontsize=6)
    ax.set_xlabel('Run', fontsize=7)
    ax.set_ylabel(f'Pearson r  vs  run {ref_run}  (averaged H, aligned)', fontsize=7)
    ax.set_title('b', fontsize=8, fontweight='bold', loc='left', pad=3)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=6, frameon=False, loc='lower right')
    ax.tick_params(labelsize=6, length=2, width=0.4); ax.tick_params(axis='x', length=0)
    for sp in ax.spines.values(): sp.set_linewidth(0.5)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_REF,
                       label=f'Reference run {ref_run}  (approach {approach_num})'),
        mpatches.Patch(facecolor=COLOR_SEL, label=f'Selected  (r ≥ {THRESHOLD})'),
        mpatches.Patch(facecolor=COLOR_REJ, label='Rejected'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=6.5, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(outdir, 'run_selection.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f'  Saved → {out}')


def plot_loglik_traces(approach_num, ref_run, selected, outdir):
    fig, axes = plt.subplots(5, 4, figsize=(18, 14),
                             sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for idx, run in enumerate(RUNS):
        ax  = axes_flat[idx]
        lls = all_logliks[run]
        bc  = best_chain[run]
        is_selected = run in selected
        is_ref      = run == ref_run

        # Highlight selected runs with a coloured background
        if is_ref:
            ax.set_facecolor('#ffeaea')   # light red for reference
        elif is_selected:
            ax.set_facecolor('#eaffea')   # light green for selected

        for cc, trace in enumerate(all_traces[run]):
            is_best = (cc == bc)
            color   = COLOR_BEST if is_best else CHAIN_COLORS[cc % len(CHAIN_COLORS)]
            ax.plot(np.arange(len(trace)), trace,
                    lw    = 1.5 if is_best else 0.7,
                    alpha = 1.0 if is_best else 0.55,
                    color = color,
                    zorder= 3   if is_best else 2)

        status      = ' [REF]' if is_ref else (' [SEL]' if is_selected else '')
        title_color = COLOR_REF if is_ref else (COLOR_SEL if is_selected else 'black')
        ax.set_title(f'Run {run}{status}', fontsize=8,
                     color=title_color,
                     fontweight='bold' if (is_ref or is_selected) else 'normal', pad=3)
        ax.tick_params(labelsize=7, length=2, width=0.5)
        ax.set_xlabel('Iteration', fontsize=7)
        ax.set_ylabel('Log-likelihood', fontsize=7)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
        for sp in ax.spines.values(): sp.set_linewidth(0.5)

    # Legend: best chain + all 10 chain indices with their colours
    legend_handles = [
        plt.Line2D([0],[0], color=COLOR_BEST, lw=2.0,
                   label='Best chain (per run)'),
    ]
    for cc in range(CHAIN_COUNT):
        legend_handles.append(
            plt.Line2D([0],[0], color=CHAIN_COLORS[cc], lw=1.2, alpha=0.8,
                       label=f'Chain {cc}'))
    # Background patches for selection status
    legend_handles += [
        mpatches.Patch(facecolor='#ffeaea', edgecolor='grey', lw=0.5,
                       label=f'Reference run ({ref_run})'),
        mpatches.Patch(facecolor='#eaffea', edgecolor='grey', lw=0.5,
                       label='Selected run'),
    ]
    fig.legend(handles=legend_handles,
               loc='upper center', bbox_to_anchor=(0.5, 1.005),
               ncol=7, fontsize=7.5, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0, w_pad=2.5)
    out = os.path.join(outdir, 'loglik_traces.png')
    plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close()
    print(f'  Saved → {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Run all three approaches
# ══════════════════════════════════════════════════════════════════════════════

APPROACH_FUNCS = {
    1: ('Highest log-likelihood',                approach1_selection),
    2: ('Consensus (max agreement)',              approach2_selection),
    3: ('Consensus anchored to highest loglik',   approach3_selection),
}

for ap_num, (ap_name, ap_func) in APPROACH_FUNCS.items():
    print(f'\n{"="*60}')
    print(f'Approach {ap_num}: {ap_name}')
    print('='*60)
    ref_run, selected, r_vals = ap_func()
    print(f'  Reference run: {ref_run}  '
          f'(loglik={logliks_best[ref_run]:.2f})')
    print(f'  Selected runs: {selected}')

    outdir = os.path.join(BASE_OUTDIR, f'approach_{ap_num}')
    os.makedirs(outdir, exist_ok=True)

    plot_run_selection(ap_num, ref_run, selected, r_vals, outdir)
    plot_loglik_traces(ap_num, ref_run, selected, outdir)

print('\nDone.')
