# Bayesian Differential Expression: HDI + ROPE

---

## Goal
Determine if gene *g* is **differentially expressed** between clone *k₁* and clone *k₂*.

---

## Step 1: Get posterior samples of the difference

From MCMC, we have *S* posterior samples of B_{k,g} for each clone.  
Compute the difference sample-by-sample:

$$\delta_g^{(i)} = B_{k_1,g}^{(i)} - B_{k_2,g}^{(i)}, \quad i = 1, \ldots, S$$

→ This gives us the **posterior distribution of the expression difference**.

---

## Step 2: Compute the 95% HDI

Find the **shortest interval** containing 95% of the δ_g samples:

$$\text{HDI} = [\text{low}, \text{high}] \quad \text{s.t. } P(\text{low} \leq \delta_g \leq \text{high} \mid \text{Data}) = 0.95$$

→ This tells us **where the true difference credibly lies**.

---

## Step 3: Define the ROPE

The **Region of Practical Equivalence** = differences too small to matter:

$$\text{ROPE} = [-\varepsilon, +\varepsilon]$$

We set ε = 10th percentile of |B_{k₁,g} − B_{k₂,g}| across all genes and clone pairs.  
→ "Ignore the smallest 10% of differences — they're biologically negligible."

---

## Step 4: Decision rule

| Condition | Conclusion |
|-----------|-----------|
| HDI entirely **above** +ε | ✅ DE: gene higher in clone *k₁* |
| HDI entirely **below** −ε | ✅ DE: gene higher in clone *k₂* |
| HDI **overlaps** the ROPE | ❌ Not DE (or inconclusive) |

$$\text{Gene is DE} \iff \text{HDI}_{\text{low}} > \varepsilon \;\;\text{ OR }\;\; \text{HDI}_{\text{high}} < -\varepsilon$$

---

## Why this approach?

- **Purely Bayesian** — no p-values, no frequentist tests on posterior samples
- **Two requirements at once:**
  - The difference is **credibly non-zero** (HDI excludes zero)
  - The difference is **large enough to matter** (HDI excludes the ROPE)
- Avoids calling tiny but "statistically certain" differences as DE

---

## Visual intuition

```
        ROPE
     |------|------|
    -ε      0     +ε

Case 1: DE (higher in k₁)         Case 2: NOT DE
                                   
         [====HDI====]                [===HDI===]
              ↑                           ↑
     entirely right of +ε          overlaps the ROPE
```

---

*Reference: Kruschke (2018), "Rejecting or Accepting Parameter Values in Bayesian Estimation"*
