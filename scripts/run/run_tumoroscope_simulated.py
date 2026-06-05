"""
Run Tumoroscope on simulated data for comparison with ClonalGE.

Mirrors the interface of main_diff_config.py so results land in the same
directory structure and format as ClonalGE results.

Usage:
    python run_tumoroscope_simulated.py {run_num} {output_dir} {config_dir} {noise_level}

Example (single run):
    python run_tumoroscope_simulated.py 1 /path/to/Results_simulated_tumoroscope/ configs_tumoroscope 0

Environment variables:
    TUMOROSCOPE_SRC   Path to Tumoroscope/src/  (default: ../Tumoroscope/src relative to this script)
    SIM_DIR           If set, load existing ClonalGE simulation objects from
                      {SIM_DIR}/{run_num}/sample_{condition} instead of generating new ones.
                      This gives a proper paired comparison (same data for both models).
                      Example: SIM_DIR=/path/to/test_sim

Outputs per run directory:
    results_normal.txt
    results_low_variance.txt
    results_high_coverage.txt
    sample_{condition}         (simulation object, only if generating new)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")
os.environ.setdefault("BLAS_NUM_THREADS",      "1")

import sys, json, glob, re, random, time, pickle
import numpy as np

# ── Tumoroscope source path — override with TUMOROSCOPE_SRC env var ──────────
_default_tum_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Tumoroscope', 'src')
TUMOROSCOPE_SRC = os.environ.get('TUMOROSCOPE_SRC', _default_tum_src)
sys.path.insert(0, os.path.abspath(TUMOROSCOPE_SRC))

from tumoroscope.simulation import simulation as tum_sim
from tumoroscope.tumoroscope import tumoroscope as tum_model

# ── CLI args ──────────────────────────────────────────────────────────────────
if len(sys.argv) < 5:
    print("Usage: run_tumoroscope_simulated.py <run_num> <output_dir> <config_dir> <noise>")
    sys.exit(1)

run_number  = str(sys.argv[1]).strip()
if not run_number:
    print("ERROR: run_number (argv[1]) is empty.")
    print("  If using SLURM, submit with: sbatch run_tumoroscope_simulated_slurm.sh")
    print("  If running manually:         python run_tumoroscope_simulated.py 1 <output_dir> <config_dir> <noise>")
    sys.exit(1)
results_dir = sys.argv[2].rstrip('/') + run_number   # e.g. .../Results_simulated_tumoroscope1
config_dir  = sys.argv[3]
noise       = str(sys.argv[4])

# ── Optional: reuse existing ClonalGE simulation objects ─────────────────────
# If SIM_DIR is set, load sample_{condition} from {SIM_DIR}/{run_number}/
# This gives a proper paired comparison (exact same A, D, C, H, phi as ClonalGE).
sim_dir = os.environ.get('SIM_DIR', '').rstrip('/')
USE_EXISTING = bool(sim_dir)

# ── MCMC / inference constants (match ClonalGE main_diff_config.py) ───────────
pi_2D          = True
th             = 0.8        # threshold for Z
every_n_sample = 5
changes_batch  = 500
optimal_rate   = 0.40

os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {results_dir}")

# ── Loop over config files ────────────────────────────────────────────────────
for cfg_path in sorted(glob.glob(os.path.join(config_dir, "*.json"))):
    # Derive condition name from filename, e.g. "normal", "low_variance"
    file_name = re.sub(r'\.json$', '', os.path.basename(cfg_path))
    result_txt = os.path.join(results_dir, f"results_{file_name}.txt")

    print(f"\n{'='*60}")
    print(f"Condition: {file_name}")
    print(f"{'='*60}")

    with open(cfg_path) as f:
        data = json.load(f)

    # ── Dimensions ────────────────────────────────────────────────────────────
    K     = data['structure']['K']
    S     = data['structure']['S']
    I     = data['structure']['I']
    theta = data['structure']['theta']

    # ── C matrix ──────────────────────────────────────────────────────────────
    C_temp      = data['C_variation']['C']
    repeat_temp = data['C_variation']['repeat']
    p_c_binom   = data['C_variation']['p_c_binom']
    if C_temp is None:
        C = None
    else:
        C_parts = [np.tile(C_temp[i], (repeat_temp[i], 1))
                   for i in range(len(C_temp))]
        C = np.concatenate(C_parts)

    # ── Priors ────────────────────────────────────────────────────────────────
    n_sampling          = data['n_variation']['n_sampling']
    n_fixed             = np.array(data['n_variation']['n']) if data['n_variation']['n'] else None
    Z_fixed             = data['Z_variation']['Z']
    avarage_clone_in_spot = data['Z_variation']['avarage_clone_in_spot']

    max_iter  = int(data['sampling']['max_iter'])
    min_iter  = int(data['sampling']['min_iter'])
    burn_in   = int(data['sampling']['burn_in'])
    batch     = int(data['sampling']['batch'])
    var_calculation = int(data['sampling']['min_iter'] * 0.9)

    phi_gamma          = np.array(data['Gamma']['phi_gamma'])
    phi_gamma_selected = np.array(data['Gamma']['phi_gamma_selected'])
    F_epsilon          = np.tile(data['Gamma']['F_epsilon'],          (K, 1))
    F_epsilon_selected = np.tile(data['Gamma']['F_epsilon_selected'], (K, 1))
    F                  = np.tile(data['Gamma']['F'],          (K, 1))
    F_selected         = np.tile(data['Gamma']['F_selected'], (K, 1))
    F_fraction         = data['Gamma']['F_fraction']
    mean_read          = int(data['Gamma']['mean_read'])

    gamma          = data['theta']['gamma']
    gamma_sampling = data['theta']['gamma_sampling']
    theta_variable = data['theta']['theta_variable']

    n_lambda = np.tile(data['n_variation']['n_lambda'], S)

    # ── Load or generate simulation ───────────────────────────────────────────
    # Try two path layouts:
    #   1. {SIM_DIR}/{run_number}/sample_{cond}  (generate_simulations.sh layout)
    #   2. {SIM_DIR}/sample_{cond}               (flat layout, no run subdirs)
    sim_path = None
    if USE_EXISTING:
        candidate_1 = os.path.join(sim_dir, run_number, f'sample_{file_name}')
        candidate_2 = os.path.join(sim_dir, f'sample_{file_name}')
        if os.path.exists(candidate_1):
            sim_path = candidate_1
        elif os.path.exists(candidate_2):
            sim_path = candidate_2
        else:
            print(f"  WARNING: simulation not found at either:")
            print(f"    {candidate_1}")
            print(f"    {candidate_2}")
            print(f"  Generating fresh simulation.")

    if sim_path is not None:
        # Reuse the exact ClonalGE simulation object → proper paired comparison
        print(f"Loading existing simulation from {sim_path}")
        # Add ClonalGE to path so the ClonalGE simulation class unpickles correctly
        clonalge_dir = os.path.dirname(os.path.abspath(__file__))
        if clonalge_dir not in sys.path:
            sys.path.insert(0, clonalge_dir)
        sample_1 = pickle.load(open(sim_path, 'rb'))
        print(f"  Loaded: K={sample_1.K}, S={sample_1.S}, I={sample_1.I}")
    else:
        print(f"Generating simulated data (target mean_read ~{mean_read}) ...")
        attempts = 0
        while True:
            attempts += 1
            sample_1 = tum_sim(
                K=K, S=S, r=phi_gamma[0], p=phi_gamma[1], I=I,
                F=F, D=None, A=None, C=C,
                avarage_clone_in_spot=avarage_clone_in_spot,
                random_seed=random.randint(1, 10000),
                F_epsilon=F_epsilon, n=n_fixed,
                p_c_binom=p_c_binom, theta=theta, Z=Z_fixed,
                n_lambda=n_lambda, F_fraction=F_fraction,
                theta_variable=theta_variable, gamma=gamma,
                pi_2D=pi_2D,
            )
            mean_d = np.mean(np.sum(sample_1.D, axis=0))
            if mean_d > mean_read * 0.9 and mean_d < mean_read * 1.1:
                print(f"  OK: mean reads = {mean_d:.1f} (attempt {attempts})")
                break
            if attempts > 200:
                print(f"  WARNING: giving up after 200 attempts (mean reads = {mean_d:.1f})")
                break
        # Save newly generated simulation object
        pickle.dump(sample_1, open(os.path.join(results_dir, f"sample_{file_name}"), 'wb'))

    # ── n_lambda for inference (add noise if requested) ───────────────────────
    if noise == '0':
        n_lambda_tum = sample_1.n
    else:
        b            = np.random.binomial(n=1, p=0.5, size=S)
        noise_pois   = np.random.poisson(lam=int(noise), size=S)
        n_lambda_tum = sample_1.n + noise_pois * b + noise_pois * (b - 1)
        n_lambda_tum[n_lambda_tum < 1] = 1

    # ── Tumoroscope inference (1 chain, matching ClonalGE settings) ───────────
    print("Running Tumoroscope inference ...")
    t_start = time.time()

    tum = tum_model(
        name              = os.path.join(results_dir, f"{file_name}_chain"),
        K                 = sample_1.K,
        S                 = sample_1.S,
        r                 = phi_gamma_selected[0],
        p                 = phi_gamma_selected[1],
        I                 = sample_1.I,
        avarage_clone_in_spot = sample_1.avarage_clone_in_spot,
        F                 = F_selected,
        C                 = sample_1.C,
        A                 = sample_1.A,
        D                 = sample_1.D,
        F_epsilon         = F_epsilon_selected,
        optimal_rate      = optimal_rate,
        n_lambda          = n_lambda_tum,
        gamma             = gamma_sampling,
        pi_2D             = pi_2D,
        result_txt        = result_txt,
    )

    tum.gibbs_sampling(
        seed             = random.randint(1, 10000),
        min_iter         = min_iter,
        max_iter         = max_iter,
        burn_in          = burn_in,
        batch            = batch,
        simulated_data   = sample_1,
        n_sampling       = True,
        F_fraction       = F_fraction,
        theta_variable   = theta_variable,
        pi_2D            = pi_2D,
        th               = th,
        every_n_sample   = every_n_sample,
        changes_batch    = changes_batch,
        var_calculation  = var_calculation,
    )

    tum.time = time.time() - t_start
    print(f"  Done in {tum.time/60:.1f} min  |  H_SEE={tum.H_SEE:.5f}  phi_SEE={tum.phi_SEE:.6f}")

    # ── Save results in ClonalGE-compatible format ─────────────────────────────
    tum.save_in_txt(sample_1, result_txt)
    print(f"  Results → {result_txt}")

print("\nAll conditions complete.")
