"""
Extract per-run best-chain logliks and within-run-averaged H for all 20 prostate runs.

Run on the cluster (numpy 2.x) where the chain pickles were created.

Usage:
    python extract_run_logliks.py

Writes two files in the current directory:
    run_logliks.csv          — run, best_loglik, n_chains_kept, best_chain_idx
    run_H_averaged.npy       — (20, S, K)  within-run averaged H per run (aligned to run order)
    run_H_averaged_ids.txt   — run ids corresponding to rows of run_H_averaged.npy
"""

import os, sys, types, pickle, csv
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# numpy 2.x renamed numpy.core → numpy._core; stub it so pickles load on numpy 1.x
import numpy.core.multiarray as _ma
import numpy.core.numeric as _num
_fake = types.ModuleType('numpy._core')
_fake.multiarray = _ma
_fake.numeric    = _num
_fake.umath      = sys.modules.get('numpy.core.umath', np)
sys.modules['numpy._core']            = _fake
sys.modules['numpy._core.multiarray'] = _ma
sys.modules['numpy._core.numeric']    = _num

PREFIX     = '/Volumes/LenovoPS8/ClonalGE/Results_figure5_'
RUNS       = list(range(1, 21))
CHAIN_COUNT = 10
THRESHOLD  = 0.9
SECTION    = 'all'

rows = []
H_averaged = []
run_ids_out = []

for run in RUNS:
    chains = []
    for cc in range(CHAIN_COUNT):
        path = f'{PREFIX}{run}/inferred_vars_{SECTION}_chain_{cc}'
        try:
            chains.append(pickle.load(open(path, 'rb')))
        except FileNotFoundError:
            print(f'  Run {run} chain {cc}: NOT FOUND')

    if not chains:
        print(f'Run {run}: no chains found, skipping')
        continue

    lls   = [c.last_loglik for c in chains]
    best  = int(np.argmax(lls))
    ref_H = chains[best].inferred_H.flatten()

    # within-run selection: keep chains with r > threshold vs best
    sel_idx = [i for i, c in enumerate(chains)
               if i == best or pearsonr(ref_H, c.inferred_H.flatten())[0] > THRESHOLD]
    n_kept = len(sel_idx)

    H_mean = np.mean([chains[i].inferred_H for i in sel_idx], axis=0)

    print(f'Run {run:2d}: best_chain={best}  loglik={lls[best]:.2f}  '
          f'kept {n_kept}/{len(chains)} chains')

    rows.append({
        'run':            run,
        'best_loglik':    lls[best],
        'n_chains_kept':  n_kept,
        'best_chain_idx': best,
        'all_logliks':    ' '.join(f'{ll:.2f}' for ll in lls),
    })
    H_averaged.append(H_mean)
    run_ids_out.append(run)

# Save
with open('run_logliks.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['run', 'best_loglik', 'n_chains_kept',
                                           'best_chain_idx', 'all_logliks'])
    writer.writeheader()
    writer.writerows(rows)

np.save('run_H_averaged.npy', np.array(H_averaged))
with open('run_H_averaged_ids.txt', 'w') as f:
    f.write('\n'.join(str(r) for r in run_ids_out) + '\n')

print(f'\nWrote run_logliks.csv  ({len(rows)} runs)')
print(f'Wrote run_H_averaged.npy  shape={np.array(H_averaged).shape}')
print(f'Wrote run_H_averaged_ids.txt')
