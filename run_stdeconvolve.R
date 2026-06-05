# Run STdeconvolve on all exported simulated Y matrices.
# Reads from:  /Volumes/LenovoPS8/ClonalGE/stdeconvolve_inputs/{run}/{config}/Y.csv
# Writes to:   /Volumes/LenovoPS8/ClonalGE/stdeconvolve_results/{run}/{config}/
#   - theta.csv : spots x K proportions
#   - beta.csv  : K x genes expression profiles (row sums to 1)

if (!requireNamespace("STdeconvolve", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("JEFworks-Lab/STdeconvolve")
}

library(STdeconvolve)

IN_DIR  <- "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_inputs"
OUT_DIR <- "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_results"
K_TRUE  <- 5
RUNS    <- 1:20
CONFIGS <- c("normal", "low_variance", "high_coverage")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

for (run in RUNS) {
  for (config in CONFIGS) {
    in_path  <- file.path(IN_DIR,  run, config)
    out_path <- file.path(OUT_DIR, run, config)
    y_file   <- file.path(in_path, "Y.csv")

    if (!file.exists(y_file)) {
      cat("MISSING:", y_file, "\n")
      next
    }

    dir.create(out_path, showWarnings = FALSE, recursive = TRUE)

    cat("Processing run", run, "/", config, "\n")

    # Load Y as spots x genes count matrix
    Y_mat <- as.matrix(read.csv(y_file, check.names = FALSE))

    # STdeconvolve expects genes x spots
    counts <- t(Y_mat)

    # Use all genes (no filtering) — most favorable for STdeconvolve
    # Remove spots with zero total counts (if any)
    spot_sums <- colSums(counts)
    if (any(spot_sums == 0)) {
      cat("  Removing", sum(spot_sums == 0), "zero-count spots\n")
      counts <- counts[, spot_sums > 0, drop = FALSE]
    }

    # fitLDA expects pixels x genes
    corpus <- t(counts)

    # Fit LDA with K = true number of clones
    ldaOut <- fitLDA(corpus, Ks = K_TRUE, ncores = 1, verbose = FALSE)

    # Extract results for K = K_TRUE
    optLDA <- optimalModel(models = ldaOut, opt = K_TRUE)

    # theta: spots x K (pixel topic proportions)
    theta <- getBetaTheta(optLDA, perc.filt = 0.0, betaScale = 1)$theta
    # beta:  K x genes (topic gene profiles, rows sum to 1)
    beta  <- getBetaTheta(optLDA, perc.filt = 0.0, betaScale = 1)$beta

    write.csv(theta, file.path(out_path, "theta.csv"), row.names = FALSE)
    write.csv(beta,  file.path(out_path, "beta.csv"),  row.names = FALSE)

    cat("  Done: theta", nrow(theta), "x", ncol(theta),
        "| beta", nrow(beta), "x", ncol(beta), "\n")
  }
}

cat("All done.\n")
