"""
generate_all_figures.py — generate Figures 3, 4, 5 (+ supplementary) for all
three run-selection approaches.

Approach | Method
---------|---------------------------------------------------------------
  1      | Highest log-likelihood single run
  2      | Consensus: reference that maximises cross-run agreement count
  3      | Consensus anchored to the highest-loglik run

Output structure
----------------
  plots_paper/
    approach_1_highest_loglik/   figure3.png  figure4.png  figure5.png  figure5_supp.png
    approach_2_consensus/        figure3.png  figure4.png  figure5.png  figure5_supp.png
    approach_3_consensus_hl/     figure3.png  figure4.png  figure5.png  figure5_supp.png

Usage
-----
    python generate_all_figures.py [options]

Options
-------
    --result_prefix   Directory prefix for run results  (default: Results_figure5)
    --num_runs        Number of independent runs         (default: 10)
    --start_seed      Starting seed index                (default: 1)
    --section         Section name                       (default: all)
    --threshold       Pearson r agreement threshold      (default: 0.9)
    --n_s_data        Path to spatial coordinate CSV
    --tum_h           Tumoroscope H .npy (figure5 panel a, optional)
    --tum_n           Tumoroscope N .npy (figure5 panel a, optional)
    --approaches      Which approaches to run, e.g. 1 2 3  (default: all)
    --figures         Which figures to run, e.g. 3 4 5     (default: all)
"""

import argparse
import subprocess
import sys
import os

APPROACH_DIRS = {
    1: 'approach_1_highest_loglik',
    2: 'approach_2_consensus',
    3: 'approach_3_consensus_hl',
}


def run(cmd, desc):
    print(f'\n{"="*70}')
    print(f'  {desc}')
    print(f'{"="*70}')
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f'  WARNING: command exited with code {result.returncode}')
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_prefix', default='Results_figure5')
    ap.add_argument('--num_runs',      type=int, default=10)
    ap.add_argument('--start_seed',    type=int, default=1)
    ap.add_argument('--section',       default='all')
    ap.add_argument('--threshold',     type=float, default=0.9)
    ap.add_argument('--n_s_data',
                    default='prostate_data_configs/'
                            'prostate_cell_count_annotation_any_fixed_margin.txt')
    ap.add_argument('--tum_h',         default=None)
    ap.add_argument('--tum_n',         default=None)
    ap.add_argument('--base_outdir',   default='plots_paper')
    ap.add_argument('--approaches',    type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--figures',       type=int, nargs='+', default=[3, 4, 5])
    args = ap.parse_args()

    py = sys.executable   # same Python interpreter

    common = [
        '--result_prefix', args.result_prefix,
        '--num_runs',       str(args.num_runs),
        '--start_seed',     str(args.start_seed),
        '--section',        args.section,
        '--threshold',      str(args.threshold),
    ]

    errors = 0
    for approach in args.approaches:
        outdir = os.path.join(args.base_outdir, APPROACH_DIRS[approach])
        print(f'\n\n{"#"*70}')
        print(f'  APPROACH {approach}: {APPROACH_DIRS[approach]}')
        print(f'  Output → {outdir}')
        print(f'{"#"*70}')

        if 3 in args.figures:
            rc = run(
                [py, 'plot_figure3.py'] + common +
                ['--approach', str(approach), '--outdir', outdir],
                f'Figure 3  (approach {approach})')
            errors += rc != 0

        if 4 in args.figures:
            rc = run(
                [py, 'plot_figure4.py'] + common +
                ['--approach', str(approach), '--outdir', outdir,
                 '--n_s_data', args.n_s_data],
                f'Figure 4  (approach {approach})')
            errors += rc != 0

        if 5 in args.figures:
            fig5_cmd = (
                [py, 'plot_figure5.py'] + common +
                ['--approach', str(approach),
                 '--cross_run_thresh', str(args.threshold),
                 '--outdir', outdir])
            if args.tum_h:
                fig5_cmd += ['--tum_h', args.tum_h]
            if args.tum_n:
                fig5_cmd += ['--tum_n', args.tum_n]
            rc = run(fig5_cmd, f'Figure 5 + S4  (approach {approach})')
            errors += rc != 0

    print(f'\n\n{"="*70}')
    if errors:
        print(f'  Done with {errors} error(s). Check output above.')
    else:
        print(f'  All figures generated successfully.')
    print(f'  Output root: {args.base_outdir}/')
    for a in args.approaches:
        print(f'    {APPROACH_DIRS[a]}/')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
