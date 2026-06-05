# Estimating B from Tumoroscope output: LR and NB methods

## Background

Tumoroscope infers clone composition **H** (S×K) and cell counts **n** (S,) but has
no gene expression component.  ClonalGE adds latent variable **B** (K×G, average
expression per clone per gene) and observed **Y** (S×G, gene expression per spot).

To provide a fair baseline we estimate B **post-hoc** from Tumoroscope's inferred H
and n, using two methods:

| Method | Noise model | Implementation |
|--------|-------------|----------------|
| Tumoroscope+LR | Gaussian (implicit) | non-negative least squares |
| Tumoroscope+NB | Negative Binomial (correct generative model) | MAP via L-BFGS-B |

---

## Generative model for Y (ClonalGE / ground truth)

The mean expression at spot *s* for gene *g* is:

```
μ_sg = n_s · Σ_k H_sk · B_kg
```

Observed counts are drawn from a Negative Binomial:

```
Y_sg ~ NB( r_sg, p_g )

where  r_sg = μ_sg · p_g / (1 − p_g)
             = n_s · (H B)_sg · p_g / (1 − p_g)
```

Using scipy's parameterisation `nbinom(n, p)`:
- Mean    = n · (1−p) / p  →  mean(Y_sg) = r_sg · (1−p_g)/p_g = μ_sg  ✓
- Variance = n · (1−p) / p²  →  var(Y_sg) = μ_sg / p_g

So **p_g controls the overdispersion**: small p_g → large variance relative to mean.

Prior on B:

```
B_kg ~ Gamma( b_alpha_g, b_beta )
```

where `b_alpha_g` is per-gene (drawn from Gamma(b_alpha_shape, b_alpha_scale) during
simulation) and `b_beta` is a scalar hyperparameter (3.5 in the configs).

---

## Method 1 — Tumoroscope+LR (non-negative least squares)

### Model

Ignores the NB noise structure and solves the linear system:

```
Y ≈ diag(n) · H · B
```

Per gene *g*, solve:

```
min_{B_g ≥ 0}  ‖ diag(n) · H · B_g − Y_g ‖²
```

This is non-negative least squares (NNLS), implemented via
`scipy.optimize.lsq_linear(..., bounds=(0, ∞))`.

### What it ignores

- The NB noise structure (overdispersion, discrete counts).
- The Gamma prior on B.
- Genes with all-zero expression are skipped (B_g = 0).

### Code (`compute_tumoroscope_lr.py`)

```python
def calc_B(Y, H, N):
    K = H.shape[1]
    G = Y.shape[1]
    X = np.diag(N) @ H          # (S, K)
    B = np.zeros((K, G))
    for g in range(G):
        if Y[:, g].sum() > 0:
            B[:, g] = lsq_linear(X, Y[:, g], bounds=(0, np.inf)).x
    return B
```

---

## Method 2 — Tumoroscope+NB (MAP under correct generative model)

### Objective

For each gene *g* independently, maximise the **log posterior**:

```
log P(B_g | Y_g, H, n, p_g)
  ∝  Σ_s log NB( Y_sg | r_sg, p_g )  +  Σ_k log Gamma( B_kg | b_alpha_g, b_beta )
```

where:

```
r_sg = n_s · Σ_k H_sk · B_kg · p_g / (1 − p_g)
```

This is equivalent to:

```
max_{B_g ≥ 0}  Σ_s [ log Γ(Y_sg + r_sg) − log Γ(r_sg)
                      + r_sg · log(p_g)  +  Y_sg · log(1−p_g) ]
             + Σ_k [ (b_alpha_g − 1) · log(B_kg) − B_kg / b_beta ]
```

### Parameters

| Symbol | Source | Notes |
|--------|--------|-------|
| H, n | Tumoroscope inferred | fixed during B optimisation |
| p_g | `sim.p_y[g]` | per-gene; in real data estimated as mean(Y_g)/var(Y_g) |
| b_alpha_g | `sim.b_alpha[g]` | per-gene Gamma shape for B prior |
| b_beta | `sim.b_beta` | scalar Gamma scale for B prior (= 3.5) |

### Initialisation

The LR solution is used as initial guess (warm start), ensuring the NB optimiser
starts from a feasible, reasonable point rather than an arbitrary one.

### Code (`compute_tumoroscope_nb.py`)

```python
def calc_B_nb(Y, H, n, p_y, b_alpha, b_beta):
    p_y_ratio = p_y / (1 - p_y)          # p / (1−p),  shape (G,)
    X = np.diag(n) @ H                   # (S, K)  — for LR warm start

    B = np.zeros((K, G))
    for g in range(G):
        if Y[:, g].sum() == 0:
            continue

        # NB log-posterior (negated for minimisation)
        def neg_log_posterior(b_kg):
            r_sg = n * (H @ b_kg) * p_y_ratio[g]
            r_sg = np.maximum(r_sg, 1e-10)
            ll  = np.sum(scipy.stats.nbinom.logpmf(Y[:, g], r_sg, p_y[g]))
            lp  = np.sum(scipy.stats.gamma.logpdf(b_kg,
                         a=b_alpha[g], scale=b_beta))
            return -(ll + lp)

        b0  = lsq_linear(X, Y[:, g], bounds=(1e-6, np.inf)).x   # warm start
        res = minimize(neg_log_posterior, b0,
                       method='L-BFGS-B',
                       bounds=[(1e-10, None)] * K,
                       options={'maxiter': 500, 'ftol': 1e-9})
        B[:, g] = res.x
    return B
```

---

## Why NB should be better than LR

1. **Correct noise model.** Data is generated from NB, so the NB likelihood
   is the true score function.  LR minimises squared error, which corresponds
   to Gaussian noise — misspecified here.

2. **Informative prior.** The Gamma prior on B regularises towards positive
   values and penalises very large B, whereas NNLS has only non-negativity.

3. **Discrete counts.** NB treats Y as integer-valued; LR treats it as
   continuous, so LR gives equal weight to e.g. errors at Y=0 and Y=1000.

---

## Why the empirical improvement is modest (~10–15%)

Despite using the correct model, NB only improves on LR by ~10–15% in B_SEE
(19 of 20 runs in every condition).  The main reasons:

1. **H is the bottleneck.**  Both methods inherit the same Tumoroscope H and n.
   If H has ~30% rMAE (Normal condition), the estimate of B is fundamentally
   limited regardless of the regression method.

2. **Large S dampens noise model choice.**  With S=300 spots the NB and
   Gaussian likelihoods become increasingly similar via the CLT; LR sees almost
   the same signal.

3. **Non-negativity already regularises LR.**  The NNLS constraint partially
   compensates for the missing prior.

4. **L-BFGS-B local optima.**  The NB posterior is not globally convex in B
   (because B enters via H@B inside the NB log-Gamma terms).  The optimiser
   may find a local optimum, especially when H has near-collinear columns.

---

## Comparison with ClonalGE

ClonalGE is substantially better than both Tumoroscope+LR and Tumoroscope+NB
because it **jointly infers H, n, and B** via Gibbs sampling:

- The posterior over B is explored stochastically, not point-estimated.
- H and n are updated to be consistent with Y (gene expression provides
  additional signal for clone composition).
- B is not optimised for a fixed H; it co-evolves with H during sampling.

This joint inference is the fundamental modelling advantage of ClonalGE.
