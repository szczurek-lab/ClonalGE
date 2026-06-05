#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_zenodo_deposit.sh
# Stages all ClonalGE benchmarking data + intermediate results into a single
# folder ready for Zenodo upload.
#
# Idempotent: rsync only copies changed files; safe to re-run.
# Run on the LenovoPS8 drive itself (no cross-volume copy).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SRC_RESULTS="/Volumes/LenovoPS8/ClonalGE"
SRC_REPO="/Users/darvis01/Documents/ClonalGE"
DEST="/Volumes/LenovoPS8/zenodo_clonalge"

# rsync flags:
#   -a archive (preserve everything)
#   -h human-readable sizes
#   --info=progress2 single overall progress bar
#   --exclude '._*' drop AppleDouble metadata
#   --exclude '.DS_Store' drop Finder metadata
RSYNC="rsync -ah --progress --exclude '._*' --exclude '.DS_Store'"

echo "==> Building Zenodo deposit at: $DEST"
mkdir -p "$DEST"

# ── 00. README + LICENSE ─────────────────────────────────────────────────────
echo
echo "==> [00/05] README + LICENSE"
cp "$SRC_REPO/zenodo_README.md" "$DEST/README.md"
cat > "$DEST/LICENSE" <<'EOF'
Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/

You are free to share and adapt the material for any purpose, even commercially,
under the following terms:
  * Attribution — You must give appropriate credit, provide a link to the
    license, and indicate if changes were made.

The accompanying source code (https://github.com/szczurek-lab/ClonalGE) is
released separately under the MIT license.
EOF

# ── 01. Input data ───────────────────────────────────────────────────────────
echo
echo "==> [01/05] Input data"
mkdir -p "$DEST/01_input_data"

echo "  → prostate"
$RSYNC "$SRC_REPO/prostate_data_configs/" "$DEST/01_input_data/prostate/"

echo "  → simulated (basic / low_var / high_cov)"
$RSYNC "$SRC_RESULTS/test_sim/" "$DEST/01_input_data/simulated/"

echo "  → simulated_violated (normal / zinb / batch)"
$RSYNC "$SRC_RESULTS/test_sim_violated/" "$DEST/01_input_data/simulated_violated/"

# ── 02. ClonalGE results ─────────────────────────────────────────────────────
echo
echo "==> [02/05] ClonalGE results (this is the bulk — may take hours)"
mkdir -p "$DEST/02_clonalge_results"

echo "  → prostate_real (20 runs × 10 chains)"
mkdir -p "$DEST/02_clonalge_results/prostate_real"
for i in $(seq 1 20); do
    src="$SRC_RESULTS/Results_figure5_$i"
    if [ -d "$src" ]; then
        $RSYNC "$src/" "$DEST/02_clonalge_results/prostate_real/Results_figure5_$i/"
    fi
done
if [ -d "$SRC_RESULTS/Results_figure5_selected" ]; then
    $RSYNC "$SRC_RESULTS/Results_figure5_selected/" \
           "$DEST/02_clonalge_results/prostate_real/Results_figure5_selected/"
fi

echo "  → simulated_clonalge_1chain"
$RSYNC "$SRC_RESULTS/Results_simulated_data/" \
       "$DEST/02_clonalge_results/simulated_clonalge_1chain/"

echo "  → simulated_clonalge_5chains"
$RSYNC "$SRC_RESULTS/Results_simulated_5chains/" \
       "$DEST/02_clonalge_results/simulated_clonalge_5chains/"

echo "  → simulated_tumoroscope (20 runs)"
mkdir -p "$DEST/02_clonalge_results/simulated_tumoroscope"
for i in $(seq 1 20); do
    src="$SRC_RESULTS/Results_simulated_tumoroscope$i"
    if [ -d "$src" ]; then
        $RSYNC "$src/" \
               "$DEST/02_clonalge_results/simulated_tumoroscope/Results_simulated_tumoroscope$i/"
    fi
done

echo "  → simulated_violated (normal/zinb/batch × 20)"
mkdir -p "$DEST/02_clonalge_results/simulated_violated"
for kind in normal zinb batch; do
    mkdir -p "$DEST/02_clonalge_results/simulated_violated/$kind"
    for i in $(seq 1 20); do
        src="$SRC_RESULTS/Results_violated_${kind}$i"
        if [ -d "$src" ]; then
            $RSYNC "$src/" \
                   "$DEST/02_clonalge_results/simulated_violated/$kind/Results_violated_${kind}$i/"
        fi
    done
done

# ── 03. Baselines (STdeconvolve) ─────────────────────────────────────────────
echo
echo "==> [03/05] STdeconvolve baseline"
mkdir -p "$DEST/03_baselines"
$RSYNC "$SRC_RESULTS/stdeconvolve_inputs/"  "$DEST/03_baselines/stdeconvolve_inputs/"
$RSYNC "$SRC_RESULTS/stdeconvolve_results/" "$DEST/03_baselines/stdeconvolve_results/"
cp "$SRC_RESULTS/stdeconvolve_comparison.csv" "$DEST/03_baselines/stdeconvolve_comparison.csv"

# ── 04. Downstream analyses ──────────────────────────────────────────────────
echo
echo "==> [04/05] Downstream analyses"
mkdir -p "$DEST/04_downstream_analyses"
$RSYNC "$SRC_RESULTS/sensitivity_results/" "$DEST/04_downstream_analyses/sensitivity_results/"
$RSYNC "$SRC_RESULTS/gsea_results/"        "$DEST/04_downstream_analyses/gsea_results/"
cp "$SRC_RESULTS/best_chain_see_5chains.csv" "$DEST/04_downstream_analyses/best_chain_see_5chains.csv"

# ── 05. Figures ──────────────────────────────────────────────────────────────
echo
echo "==> [05/05] Figures"
mkdir -p "$DEST/05_figures"
cp "$SRC_RESULTS/stdeconvolve_comparison_figure.png" "$DEST/05_figures/"
cp "$SRC_RESULTS/stdeconvolve_comparison_figure.pdf" "$DEST/05_figures/"

# ── Final cleanup of any AppleDouble files that snuck through ────────────────
echo
echo "==> Cleaning AppleDouble (._*) and .DS_Store files in deposit"
find "$DEST" -name '._*' -delete
find "$DEST" -name '.DS_Store' -delete

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "==> Done."
echo
du -sh "$DEST"
echo
du -sh "$DEST"/*/
