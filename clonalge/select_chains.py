"""
Post-processing script: select chains across runs and compute final averaged estimates.

Usage:
    python select_chains.py <result_prefix> <num_runs> <start_seed> <section>

Arguments:
    result_prefix  — directory prefix used in run_figure5.sh (e.g. Results_figure5)
    num_runs       — number of runs (e.g. 10)
    start_seed     — starting seed used in run_figure5.sh (e.g. 1)
    section        — config section name used in result filenames (e.g. P1.2)

Outputs saved to <result_prefix>_selected/:
    inferred_H.npy, inferred_B.npy, inferred_phi.npy,
    inferred_pi.npy, inferred_n.npy, inferred_P_Z.npy, inferred_Z.npy
    chain_selection_log.txt
"""

import sys
import pickle
import numpy as np
import os
from scipy.stats import pearsonr

PEARSON_THRESHOLD = 0.9
CHAIN_COUNT = 10


def load_chains(result_prefix, run_seed, section, chain_count):
    result_obj = f'{result_prefix}_{run_seed}/results_{section}'
    chains = []
    for cc in range(chain_count):
        path = result_obj + f'_chain_{cc}'
        chains.append(pickle.load(open(path, 'rb')))
    return chains


def select_chains_for_run(chains, pearson_threshold):
    logliks = [c.last_loglik for c in chains]
    best_idx = int(np.argmax(logliks))
    best_H_flat = chains[best_idx].inferred_H.flatten()

    similar = [chains[best_idx]]
    for cc, c in enumerate(chains):
        if cc == best_idx:
            continue
        r, _ = pearsonr(best_H_flat, c.inferred_H.flatten())
        if r > pearson_threshold:
            similar.append(c)

    return similar, best_idx, logliks


def main():
    result_prefix = sys.argv[1]
    num_runs      = int(sys.argv[2])
    start_seed    = int(sys.argv[3])
    section       = sys.argv[4] if len(sys.argv) > 4 else 'P1.2'

    runs = range(start_seed, start_seed + num_runs)
    selected = []
    log_lines = []

    for run in runs:
        try:
            chains = load_chains(result_prefix, run, section, CHAIN_COUNT)
        except FileNotFoundError as e:
            msg = f'Run {run}: SKIPPED (file not found: {e})'
            print(msg)
            log_lines.append(msg)
            continue

        similar, best_idx, logliks = select_chains_for_run(chains, PEARSON_THRESHOLD)

        if len(similar) == 1:
            msg = (f'Run {run}: OMITTED — best chain (idx={best_idx}, '
                   f'loglik={logliks[best_idx]:.2f}) had no similar chains')
        else:
            selected.extend(similar)
            msg = (f'Run {run}: kept {len(similar)} chains '
                   f'(best idx={best_idx}, loglik={logliks[best_idx]:.2f})')

        print(msg)
        log_lines.append(msg)

    if len(selected) == 0:
        print('ERROR: no chains selected across all runs.')
        sys.exit(1)

    print(f'\nAveraging over {len(selected)} chains from {num_runs} runs...')

    inferred_H   = np.mean([c.inferred_H   for c in selected], axis=0)
    inferred_B   = np.mean([c.inferred_B   for c in selected], axis=0)
    inferred_phi = np.mean([c.inferred_phi for c in selected], axis=0)
    inferred_pi  = np.mean([c.inferred_pi  for c in selected], axis=0)
    inferred_n   = np.mean([c.inferred_n   for c in selected], axis=0)
    inferred_P_Z = np.mean([c.inferred_P_Z for c in selected], axis=0)
    inferred_Z   = np.mean([c.inferred_Z   for c in selected], axis=0)

    out_dir = f'{result_prefix}_selected'
    os.makedirs(out_dir, exist_ok=True)

    np.save(f'{out_dir}/inferred_H.npy',   inferred_H)
    np.save(f'{out_dir}/inferred_B.npy',   inferred_B)
    np.save(f'{out_dir}/inferred_phi.npy', inferred_phi)
    np.save(f'{out_dir}/inferred_pi.npy',  inferred_pi)
    np.save(f'{out_dir}/inferred_n.npy',   inferred_n)
    np.save(f'{out_dir}/inferred_P_Z.npy', inferred_P_Z)
    np.save(f'{out_dir}/inferred_Z.npy',   inferred_Z)

    with open(f'{out_dir}/chain_selection_log.txt', 'w') as f:
        f.write('\n'.join(log_lines))
        f.write(f'\n\nTotal chains selected: {len(selected)}\n')

    print(f'Results saved to {out_dir}/')


if __name__ == '__main__':
    main()
