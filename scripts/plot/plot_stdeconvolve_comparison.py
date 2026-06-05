"""
Generate comparison figure: ClonalGE vs STdeconvolve across 3 simulation configs.
Two panels: H MAE (clone proportions) and B MAE (clone-specific expression).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("/Volumes/LenovoPS8/ClonalGE/stdeconvolve_comparison.csv")

CONFIG_LABELS = {
    "normal":        "Normal",
    "low_variance":  "Low Variance",
    "high_coverage": "High Coverage",
}
CONFIGS = ["normal", "low_variance", "high_coverage"]

COLOR_CL  = "#2166ac"   # blue  — ClonalGE
COLOR_STD = "#d6604d"   # red   — STdeconvolve

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.subplots_adjust(wspace=0.38)

def boxplot_pair(ax, metric_cl, metric_std, ylabel, title):
    data, positions, colors = [], [], []
    xticks, xlabels = [], []

    gap = 0.4
    width = 0.55
    spacing = 2.8

    for i, config in enumerate(CONFIGS):
        sub = df[df["config"] == config]
        x_cl  = i * spacing
        x_std = i * spacing + gap + width

        data.extend([sub[metric_cl].values, sub[metric_std].values])
        positions.extend([x_cl, x_std])
        colors.extend([COLOR_CL, COLOR_STD])

        mid = (x_cl + x_std) / 2
        xticks.append(mid)
        xlabels.append(CONFIG_LABELS[config])

    bp = ax.boxplot(data, positions=positions, widths=width,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3.5,
                                    markerfacecolor="gray", alpha=0.6))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate fold-improvement above each config
    for i, config in enumerate(CONFIGS):
        sub = df[df["config"] == config]
        fold = sub[metric_std].mean() / sub[metric_cl].mean()
        x_mid = i * spacing + gap / 2 + width / 2
        y_top = max(sub[metric_cl].max(), sub[metric_std].max())
        ax.text(x_mid, y_top * 1.05, f"{fold:.1f}×",
                ha="center", va="bottom", fontsize=9.5,
                color="#444444", fontstyle="italic")


boxplot_pair(axes[0],
             metric_cl="ClonalGE_H_MAE",
             metric_std="STdeconvolve_H_MAE",
             ylabel="Mean Absolute Error (H)",
             title="Clone Proportion Recovery ($H$)")

boxplot_pair(axes[1],
             metric_cl="ClonalGE_B_MAE_norm",
             metric_std="STdeconvolve_B_MAE",
             ylabel="Mean Absolute Error (B, normalised)",
             title="Clone Expression Recovery ($B$)")

# shared legend
patch_cl  = mpatches.Patch(color=COLOR_CL,  alpha=0.85, label="ClonalGE")
patch_std = mpatches.Patch(color=COLOR_STD, alpha=0.85, label="STdeconvolve")
fig.legend(handles=[patch_cl, patch_std],
           loc="upper center", ncol=2, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 1.02))

# italic annotation note
fig.text(0.5, -0.03,
         "Fold-improvement of ClonalGE over STdeconvolve shown above each group.",
         ha="center", fontsize=9, color="#555555", style="italic")

out = "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_comparison_figure.pdf"
out_png = "/Volumes/LenovoPS8/ClonalGE/stdeconvolve_comparison_figure.png"
plt.savefig(out,     bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved: {out}")
print(f"Saved: {out_png}")
