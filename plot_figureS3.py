"""
Supplementary figure: across-run agreement for H and B.

For each of the 10 independent runs, chains are first within-run selected
(Pearson r > threshold against the best-loglik chain), then averaged to give
one H and one B per run.  The figure shows the 10×10 pairwise Pearson r
matrix between per-run estimates, separately for H and B.

Usage:
    python plot_run_agreement.py [options]

Options:
    --result_prefix   Directory prefix                (default: Results_figure5)
    --num_runs        Number of runs                  (default: 10)
    --start_seed      Starting seed                   (default: 1)
    --section         Config section name             (default: all)
    --threshold       Pearson r threshold             (default: 0.9)
    --output          Output path                     (default: plots_paper/run_agreement.png)
    --synthetic       Use synthetic data (layout preview)
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

# ── layout constants ───────────────────────────────────────────────────────────
CHAIN_COUNT = 10
FS_BASE     = 7
FS_ANNOT    = 5.0
FS_TICK     = 6.0
FS_TITLE    = 7.0
FIG_W       = 5.6
FIG_H       = 2.55

CMAP        = 'Blues'
VMIN, VMAX  = 0.0, 1.0

COL_SEL     = '#1f4e79'
COL_EXC     = '#c0392b'
COL_DIAG    = '#e8e8e8'
COL_DIV     = '#555555'


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_chains(result_prefix, run_seed, section, chain_count=CHAIN_COUNT):
    base = f'{result_prefix}_{run_seed}/inferred_vars_{section}'
    return [pickle.load(open(base + f'_chain_{cc}', 'rb'))
            for cc in range(chain_count)]


def select_within_run(chains, threshold=0.9):
    logliks  = [c.last_loglik for c in chains]
    best_idx = int(np.argmax(logliks))
    ref      = chains[best_idx].inferred_H.flatten()
    selected = {best_idx}
    for cc, c in enumerate(chains):
        if cc == best_idx:
            continue
        r, _ = pearsonr(ref, c.inferred_H.flatten())
        if r > threshold:
            selected.add(cc)
    return selected, best_idx, logliks


def load_all_runs(result_prefix, num_runs, start_seed, section, threshold):
    """
    Returns:
        run_ids   — list of run seed indices (in load order)
        H_runs    — list of per-run averaged H arrays
        B_runs    — list of per-run averaged B arrays
        best_lls  — list of best-chain log-likelihoods per run
        n_kept    — list of how many chains were kept per run
    """
    run_ids, H_runs, B_runs, best_lls, n_kept = [], [], [], [], []
    for seed in range(start_seed, start_seed + num_runs):
        try:
            chains = load_chains(result_prefix, seed, section)
        except FileNotFoundError:
            print(f'  Run {seed}: not found, skipping')
            continue
        selected, best_idx, logliks = select_within_run(chains, threshold)
        H_avg = np.mean([chains[i].inferred_H for i in selected], axis=0)
        B_avg = np.mean([chains[i].inferred_B for i in selected], axis=0)
        run_ids.append(seed)
        H_runs.append(H_avg)
        B_runs.append(B_avg)
        best_lls.append(logliks[best_idx])
        n_kept.append(len(selected))
        print(f'  Run {seed}: loglik={logliks[best_idx]:.1f}  '
              f'chains kept={len(selected)}/{len(chains)}')
    return run_ids, H_runs, B_runs, best_lls, n_kept


def align_to_ref(H_ref, H_other, B_other):
    """Permute clones of H_other/B_other to best match H_ref (Hungarian)."""
    K    = H_ref.shape[1]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = -pearsonr(H_ref[:, i], H_other[:, j])[0]
    _, perm = linear_sum_assignment(cost)
    return H_other[:, perm], B_other[perm, :]


def align_all_runs(H_runs, B_runs, best_pos):
    """Align all runs to the best run using Hungarian clone matching."""
    H_ref  = H_runs[best_pos]
    B_ref  = B_runs[best_pos]
    H_al   = list(H_runs)
    B_al   = list(B_runs)
    perms  = [None] * len(H_runs)
    for i in range(len(H_runs)):
        if i == best_pos:
            perms[i] = list(range(H_ref.shape[1]))
            continue
        H_al[i], B_al[i] = align_to_ref(H_ref, H_runs[i], B_runs[i])
        K = H_ref.shape[1]
        cost = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                cost[a, b] = -pearsonr(H_ref[:, a], H_runs[i][:, b])[0]
        _, perm = linear_sum_assignment(cost)
        perms[i] = perm.tolist()
    return H_al, B_al, perms


def pairwise_r(matrices):
    n    = len(matrices)
    R    = np.ones((n, n))
    flat = [m.flatten() for m in matrices]
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(flat[i], flat[j])
            R[i, j] = R[j, i] = r
    return R


def find_consensus(H_runs, B_runs, threshold=0.9):
    """
    For each candidate reference run, align all others to it (Hungarian) and
    count how many agree (r >= threshold).  Pick the reference that maximises
    agreement count, breaking ties by highest mean r among agreeing runs.
    Returns:
        ref_pos   — position of consensus run in H_runs list
        selected  — sorted list of agreeing positions (includes ref_pos)
        H_al      — clone-aligned H for all runs (aligned to ref_pos)
        B_al      — clone-aligned B for all runs
        perms     — clone permutations applied per run
    """
    n = len(H_runs)
    best_ref, best_count, best_mean_r = 0, -1, -1.0
    best_selected = best_H_al = best_B_al = best_perms = None

    for ref_pos in range(n):
        H_al, B_al, perms = align_all_runs(H_runs, B_runs, ref_pos)
        ref_flat = H_al[ref_pos].flatten()
        r_vals   = [pearsonr(ref_flat, H_al[i].flatten())[0] for i in range(n)]
        selected = [i for i, r in enumerate(r_vals) if r >= threshold]
        mean_r   = float(np.mean([r_vals[i] for i in selected]))
        if (len(selected) > best_count or
                (len(selected) == best_count and mean_r > best_mean_r)):
            best_ref, best_count, best_mean_r = ref_pos, len(selected), mean_r
            best_selected = selected
            best_H_al, best_B_al, best_perms = H_al, B_al, perms

    return best_ref, best_selected, best_H_al, best_B_al, best_perms


def reorder(R, selected, ref_pos, n):
    """Selected runs first (ref first within selected), excluded last."""
    sel_sorted = [ref_pos] + sorted(i for i in selected if i != ref_pos)
    exc_sorted = sorted(set(range(n)) - set(selected))
    order      = sel_sorted + exc_sorted
    R_ord      = R[np.ix_(order, order)]
    new_ref    = 0                              # ref is always first after reorder
    new_sel    = set(range(len(sel_sorted)))
    return R_ord, order, new_ref, new_sel, len(sel_sorted)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic fallback
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic(n_runs=10, S=55, K=4, g=150, seed=0):
    rng    = np.random.default_rng(seed)
    H_true = rng.dirichlet(np.ones(K), size=S)
    B_true = rng.gamma(2.0, 1.0, size=(K, g))
    run_ids, H_runs, B_runs, best_lls, n_kept = [], [], [], [], []
    for i in range(n_runs):
        if i < 8:
            noise = rng.normal(0, 0.02, H_true.shape)
            H = np.clip(H_true + noise, 1e-9, None)
            H /= H.sum(axis=1, keepdims=True)
            B = np.clip(B_true + rng.normal(0, 0.1, B_true.shape), 0, None)
            ll = -41900 + rng.uniform(-300, 300)
            nk = 10
        else:
            perm = rng.permutation(K)
            H = H_true[:, perm]
            B = B_true[perm, :]
            ll = -43500 + rng.uniform(-100, 100)
            nk = 6
        run_ids.append(i + 1)
        H_runs.append(H)
        B_runs.append(B)
        best_lls.append(ll)
        n_kept.append(nk)
    return run_ids, H_runs, B_runs, best_lls, n_kept


# ══════════════════════════════════════════════════════════════════════════════
# Panel rendering  (shared with chain agreement style)
# ══════════════════════════════════════════════════════════════════════════════

def draw_heatmap(ax, R_ord, order, new_ref, new_sel, n_sel, tick_labels,
                 title, show_ylabel=True, xlabel='Run', ylabel='Run'):
    n  = R_ord.shape[0]
    im = ax.imshow(R_ord, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                   aspect='equal', interpolation='none')

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val     = R_ord[i, j]
            txt_col = 'white' if val > 0.62 else '#333333'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=FS_ANNOT, color=txt_col, zorder=3)

    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            facecolor=COL_DIAG, edgecolor='none', zorder=2))
        ax.text(i, i, '—', ha='center', va='center',
                fontsize=FS_ANNOT, color='#888888', zorder=3)

    if 0 < n_sel < n:
        cut = n_sel - 0.5
        ax.axhline(cut, color=COL_DIV, linewidth=0.6, linestyle='--', zorder=4)
        ax.axvline(cut, color=COL_DIV, linewidth=0.6, linestyle='--', zorder=4)

    ordered_labels = [tick_labels[orig_pos] for orig_pos in order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    xlbls = ax.set_xticklabels(ordered_labels, fontsize=FS_TICK)
    ylbls = ax.set_yticklabels(ordered_labels, fontsize=FS_TICK)

    for pos, (xl, yl) in enumerate(zip(xlbls, ylbls)):
        is_sel  = pos in new_sel
        is_ref  = (pos == new_ref)
        col     = COL_SEL if is_sel else COL_EXC
        for lbl in (xl, yl):
            lbl.set_color(col)
            lbl.set_fontweight('bold' if is_ref else 'normal')

    ax.set_xlabel(xlabel, fontsize=FS_BASE, labelpad=2)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_BASE, labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return im


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_prefix', default='Results_figure5')
    ap.add_argument('--num_runs',      type=int, default=10)
    ap.add_argument('--start_seed',    type=int, default=1)
    ap.add_argument('--section',       default='all')
    ap.add_argument('--threshold',     type=float, default=0.9)
    ap.add_argument('--output',        default='plots_paper/run_agreement.png')
    ap.add_argument('--synthetic',     action='store_true')
    args = ap.parse_args()

    if args.synthetic:
        print('Generating synthetic runs for layout preview...')
        run_ids, H_runs, B_runs, best_lls, n_kept = make_synthetic()
    else:
        print(f'Loading {args.num_runs} runs from {args.result_prefix}_* ...')
        run_ids, H_runs, B_runs, best_lls, n_kept = load_all_runs(
            args.result_prefix, args.num_runs, args.start_seed,
            args.section, args.threshold)

    n = len(run_ids)

    # ── find consensus run and align clones ───────────────────────────────────
    print(f'\nSearching for consensus run (threshold r > {args.threshold})...')
    ref_pos, selected, H_al, B_al, perms = find_consensus(
        H_runs, B_runs, args.threshold)

    print(f'Consensus run: {run_ids[ref_pos]}  '
          f'({len(selected)}/{n} runs agree)')
    print(f'Selected runs: {[run_ids[i] for i in selected]}')
    for i, p in enumerate(perms):
        tag = ' ← consensus' if i == ref_pos else ''
        print(f'  Run {run_ids[i]:2d}: clone permutation = {p}{tag}')

    R_H = pairwise_r(H_al)
    R_B = pairwise_r(B_al)

    sel_idx = np.array(selected)
    off_H   = [R_H[i, j] for i in sel_idx for j in sel_idx if i < j]
    off_B   = [R_B[i, j] for i in sel_idx for j in sel_idx if i < j]
    if off_H:
        print(f'\nAmong selected — H: mean r={np.mean(off_H):.4f} '
              f'min={np.min(off_H):.4f}  '
              f'B: mean r={np.mean(off_B):.4f} min={np.min(off_B):.4f}')

    # ── average selected runs → consensus estimate ─────────────────────────
    H_consensus = np.mean([H_al[i] for i in selected], axis=0)
    B_consensus = np.mean([B_al[i] for i in selected], axis=0)
    print(f'\nConsensus H shape: {H_consensus.shape}  '
          f'B shape: {B_consensus.shape}')

    R_H_ord, order, new_ref, new_sel, n_sel = reorder(R_H, set(selected), ref_pos, n)
    R_B_ord, *_                             = reorder(R_B, set(selected), ref_pos, n)

    tick_labels = [str(rid) for rid in run_ids]

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':  'sans-serif',
        'font.size':     FS_BASE,
        'axes.linewidth': 0.5,
        'pdf.fonttype':  42,
        'svg.fonttype':  'none',
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(wspace=0.38, left=0.08, right=0.88,
                         top=0.88, bottom=0.14),
    )

    im = draw_heatmap(axes[0], R_H_ord, order, new_ref, new_sel, n_sel,
                      tick_labels, title='Clonal composition  (H)',
                      show_ylabel=True, xlabel='Run', ylabel='Run')
    draw_heatmap(axes[1], R_B_ord, order, new_ref, new_sel, n_sel,
                 tick_labels, title='Gene expression  (B)',
                 show_ylabel=False, xlabel='Run', ylabel='Run')

    # ── shared colourbar ──────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.905, 0.18, 0.018, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Pearson  r', fontsize=FS_BASE, labelpad=4)
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, length=2, width=0.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(0.5)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_els = [
        mpatches.Patch(color=COL_SEL,
                       label=f'Selected  (r\u2009>\u2009{args.threshold})'),
        mpatches.Patch(color=COL_EXC, label='Excluded'),
        mpatches.Patch(facecolor='none', edgecolor=COL_DIV,
                       linestyle='--', linewidth=0.8,
                       label='Selection boundary'),
    ]
    fig.legend(handles=legend_els,
               fontsize=FS_BASE - 1, loc='lower center', ncol=3,
               bbox_to_anchor=(0.46, -0.05), frameon=False,
               handlelength=1.0, handletextpad=0.5, columnspacing=1.2)

    # ── footnote ──────────────────────────────────────────────────────────────
    fig.text(0.46, -0.09,
             'Clones aligned across runs via Hungarian algorithm '
             '(consensus run = most agreeing, bold)',
             ha='center', va='bottom', fontsize=FS_BASE - 1.5,
             color='#555555', style='italic')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=400, bbox_inches='tight')
    print(f'Saved → {args.output}')
    plt.close()


if __name__ == '__main__':
    main()
