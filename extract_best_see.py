"""
Extract SEE metrics from best-chain pickles (best_oo5_*) for 5-chain ClonalGE runs.

Run on the cluster where numpy 2.x is available and the pickles were created.

Usage:
    python extract_best_see.py
    # → writes best_chain_see_5chains.csv in the current directory

Then copy the CSV back locally and re-run plot_simulated_results.py.
"""
import pickle, csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = './Results_simulated_5chains'
CONDITIONS = ['normal', 'low_variance', 'high_coverage']

rows = []
missing = []

for run in range(1, 21):
    for cond in CONDITIONS:
        p = os.path.join(BASE, str(run), f'best_oo5_{cond}')
        if not os.path.exists(p):
            print(f'  MISSING: {p}')
            missing.append((run, cond))
            continue
        try:
            obj = pickle.load(open(p, 'rb'))
        except Exception as e:
            print(f'  ERROR loading {p}: {e}')
            missing.append((run, cond))
            continue

        rows.append({
            'run':      run,
            'cond':     cond,
            'H_SEE':    obj.H_SEE,
            'phi_SEE':  obj.phi_SEE,
            'n_SEE':    obj.n_SEE,
            'B_SEE':    obj.B_SEE,
            'loglik':   obj.last_loglik,
        })
        print(f'  run {run:2d} / {cond}: H={obj.H_SEE:.4f}  phi={obj.phi_SEE:.4f}  '
              f'n={obj.n_SEE:.4f}  B={obj.B_SEE:.4f}  loglik={obj.last_loglik:.1f}')

out = 'best_chain_see_5chains.csv'
with open(out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['run', 'cond', 'H_SEE', 'phi_SEE', 'n_SEE', 'B_SEE', 'loglik'])
    writer.writeheader()
    writer.writerows(rows)

print(f'\nWrote {len(rows)} rows to {out}')
if missing:
    print(f'Missing / failed: {missing}')
