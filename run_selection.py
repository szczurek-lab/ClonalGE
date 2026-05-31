"""
run_selection.py — shared run/chain selection module for ClonalGE figure scripts.

Three approaches for selecting and combining MCMC results across runs:

  1  highest_loglik  — use the single run with the highest within-run log-likelihood
  2  consensus       — Hungarian alignment; choose reference that maximises
                       the number of agreeing runs (current reviewer-response method)
  3  consensus_hl    — Hungarian alignment anchored to the highest-loglik run;
                       select runs that agree with it above the threshold

All functions return the same result dict so figure scripts are approach-agnostic.

Result dict keys
----------------
H_final      : (S, K)      posterior mean H
B_final      : (K, g)      posterior mean B
B_samples    : (N, K, g)   pooled MCMC B samples from selected runs/chains
n_final      : (S,)        posterior mean n
per_run      : dict        raw per-run data (for supplementary plots in figure 5)
selected_runs: list        run ids included in the average
ref_run      : int         reference / anchor run id
H_aligned    : dict        {run_id: H aligned to ref_run}
B_aligned    : dict        {run_id: B aligned to ref_run}
perms        : dict        {run_id: clone permutation applied}
approach_name: str         human-readable label
"""

import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

CHAIN_COUNT = 10

APPROACH_NAMES = {
    1: 'Highest log-likelihood',
    2: 'Consensus (max agreement)',
    3: 'Consensus anchored to highest log-likelihood',
}


# ══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_chain(prefix, run, section, cc):
    return pickle.load(open(f'{prefix}_{run}/inferred_vars_{section}_chain_{cc}', 'rb'))


def _within_run_select(prefix, run, section, threshold):
    """Load chains, keep those that agree with the best-loglik chain.

    Returns (selected_chains, H_mean, B_mean, B_samples, n_mean, best_loglik).
    Raises FileNotFoundError if the run directory is missing.
    """
    chains = [_load_chain(prefix, run, section, cc) for cc in range(CHAIN_COUNT)]
    lls    = [c.last_loglik for c in chains]
    best   = int(np.argmax(lls))
    ref    = chains[best].inferred_H.flatten()
    sel    = [i for i, c in enumerate(chains)
              if i == best or pearsonr(ref, c.inferred_H.flatten())[0] > threshold]
    sel_chains = [chains[i] for i in sel]
    return (sel_chains,
            np.mean([chains[i].inferred_H for i in sel], axis=0),
            np.mean([chains[i].inferred_B for i in sel], axis=0),
            np.concatenate([chains[i].B_sum for i in sel], axis=0),
            np.mean([chains[i].inferred_n for i in sel], axis=0),
            lls[best])


def _load_all_runs(prefix, num_runs, start_seed, section, threshold):
    """Apply within-run selection to every run.  Returns per_run dict."""
    per_run = {}
    for run in range(start_seed, start_seed + num_runs):
        try:
            sel_chains, H, B, Bs, n, ll = _within_run_select(
                prefix, run, section, threshold)
        except FileNotFoundError:
            print(f'  Run {run}: SKIPPED (file not found)')
            continue
        n_kept = len(sel_chains)
        print(f'  Run {run}: kept {n_kept}/{CHAIN_COUNT} chains  (best loglik={ll:.2f})')
        per_run[run] = dict(chains=sel_chains, H=H, B=B, B_samples=Bs,
                            n=n, best_loglik=ll, n_kept=n_kept)
    return per_run


def _align(H_ref, H_other, B_other):
    """Return (H_aligned, B_aligned, perm) with clones permuted to match H_ref."""
    K    = H_ref.shape[1]
    cost = np.array([[-pearsonr(H_ref[:, i], H_other[:, j])[0]
                       for j in range(K)] for i in range(K)])
    _, perm = linear_sum_assignment(cost)
    return H_other[:, perm], B_other[perm, :], perm


def _align_all_to_ref(per_run, ref_run):
    """Align every run's H/B to ref_run. Returns H_al, B_al, perms dicts."""
    H_al, B_al, perms = {}, {}, {}
    K = per_run[ref_run]['H'].shape[1]
    for run, data in per_run.items():
        if run == ref_run:
            H_al[run]  = data['H']
            B_al[run]  = data['B']
            perms[run] = list(range(K))
        else:
            Ha, Ba, p  = _align(per_run[ref_run]['H'], data['H'], data['B'])
            H_al[run]  = Ha
            B_al[run]  = Ba
            perms[run] = p.tolist()
    return H_al, B_al, perms


def _pool_B_samples(per_run, selected_runs, perms):
    """Concatenate B_sum samples from selected runs, aligned by perms."""
    parts = []
    for run in selected_runs:
        perm = perms[run]
        Bs   = per_run[run]['B_samples']   # (N, K, g)
        parts.append(Bs[:, perm, :])
    return np.concatenate(parts, axis=0)


def _build_result(per_run, selected_runs, ref_run, H_al, B_al, perms, approach):
    """Assemble the standard result dict."""
    H_final   = np.mean([H_al[r] for r in selected_runs], axis=0)
    B_final   = np.mean([B_al[r] for r in selected_runs], axis=0)
    B_samples = _pool_B_samples(per_run, selected_runs, perms)
    n_final   = np.mean([per_run[r]['n'] for r in selected_runs], axis=0)
    print(f'  → ref_run={ref_run}  selected={selected_runs}  '
          f'B_samples shape={B_samples.shape}')
    return dict(
        H_final=H_final, B_final=B_final, B_samples=B_samples, n_final=n_final,
        per_run=per_run, selected_runs=selected_runs, ref_run=ref_run,
        H_aligned=H_al, B_aligned=B_al, perms=perms,
        approach_name=APPROACH_NAMES[approach],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Approach 1 — highest log-likelihood run
# ══════════════════════════════════════════════════════════════════════════════

def _approach_highest_loglik(per_run):
    ref_run = max(per_run, key=lambda r: per_run[r]['best_loglik'])
    print(f'  Highest loglik run: {ref_run}  (loglik={per_run[ref_run]["best_loglik"]:.2f})')
    H_al, B_al, perms = _align_all_to_ref(per_run, ref_run)
    return _build_result(per_run, [ref_run], ref_run, H_al, B_al, perms, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Approach 2 — consensus (maximise agreement count)
# ══════════════════════════════════════════════════════════════════════════════

def _approach_consensus(per_run, threshold):
    run_ids = list(per_run.keys())
    H_list  = [per_run[r]['H'] for r in run_ids]
    n       = len(run_ids)

    best_ref, best_count, best_mean_r = 0, -1, -1.0
    best_sel = best_H_al = best_B_al = best_perms = None

    for ref_pos in range(n):
        H_al, B_al, perms = _align_all_to_ref(
            per_run, run_ids[ref_pos])
        ref_flat = H_al[run_ids[ref_pos]].flatten()
        r_vals   = [pearsonr(ref_flat, H_al[r].flatten())[0] for r in run_ids]
        sel      = [i for i, r in enumerate(r_vals) if r >= threshold]
        mean_r   = float(np.mean([r_vals[i] for i in sel]))
        if len(sel) > best_count or (len(sel) == best_count and mean_r > best_mean_r):
            best_ref, best_count, best_mean_r = ref_pos, len(sel), mean_r
            best_sel   = sel
            best_H_al  = H_al
            best_B_al  = B_al
            best_perms = perms

    ref_run       = run_ids[best_ref]
    selected_runs = [run_ids[i] for i in best_sel]
    print(f'  Consensus ref run: {ref_run}  ({best_count}/{n} runs agree)')
    return _build_result(per_run, selected_runs, ref_run,
                         best_H_al, best_B_al, best_perms, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Approach 3 — consensus anchored to highest-loglik run
# ══════════════════════════════════════════════════════════════════════════════

def _approach_consensus_hl(per_run, threshold):
    ref_run = max(per_run, key=lambda r: per_run[r]['best_loglik'])
    print(f'  HL anchor run: {ref_run}  (loglik={per_run[ref_run]["best_loglik"]:.2f})')
    H_al, B_al, perms = _align_all_to_ref(per_run, ref_run)
    ref_flat      = H_al[ref_run].flatten()
    selected_runs = [r for r in per_run
                     if pearsonr(ref_flat, H_al[r].flatten())[0] >= threshold]
    n = len(per_run)
    print(f'  Selected {len(selected_runs)}/{n} runs agreeing with HL anchor')
    if not selected_runs:
        selected_runs = [ref_run]
    return _build_result(per_run, selected_runs, ref_run, H_al, B_al, perms, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def select_runs(approach, result_prefix, num_runs, start_seed, section,
                threshold=0.9):
    """
    Run the full selection pipeline for the given approach (1, 2, or 3).
    Returns the standard result dict (see module docstring).
    """
    print(f'\n=== Approach {approach}: {APPROACH_NAMES[approach]} ===')
    print('--- Within-run chain selection ---')
    per_run = _load_all_runs(result_prefix, num_runs, start_seed, section, threshold)
    if not per_run:
        raise RuntimeError('No valid runs found.')
    print('--- Cross-run selection ---')
    if approach == 1:
        return _approach_highest_loglik(per_run)
    elif approach == 2:
        return _approach_consensus(per_run, threshold)
    elif approach == 3:
        return _approach_consensus_hl(per_run, threshold)
    else:
        raise ValueError(f'Unknown approach {approach}. Choose 1, 2, or 3.')
