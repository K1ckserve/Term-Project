"""

Reads results/suite_aggregate.csv and writes three separate PNGs:
  results/panel1_time_breakdown.png
  results/panel2_em_stability.png
  results/panel3_inertia_delta.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "demo_aggregate.csv")
OUT_DIR  = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------
STRATEGIES = ["kmeans++", "kmeans||_fixed_R2", "kmeans||_adaptive"]
LABELS     = ["k-means++", "k-means||\nFixed R=2", "k-means||\nAdaptive"]

COLORS = {
    "kmeans++":          "#4C7BE8",
    "kmeans||_fixed_R2": "#E8873A",
    "kmeans||_adaptive": "#2BAE99",
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "axes.titlepad":     10,
    "axes.labelpad":      6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================================
# Panel 1 — Stacked Bar: Time Breakdown at 1M Rows
# ===========================================================================
fig1, ax1 = plt.subplots(figsize=(7, 6.5))
fig1.suptitle(
    "k-means|| Seeding Strategy Benchmark  —  All Dataset Sizes",
    fontsize=15, fontweight="bold", y=1.02,
)

time_agg = (
    df.groupby("strategy")[["seed_time_s_mean", "em_time_s_mean"]]
    .mean()
    .reindex(STRATEGIES)
)

x         = np.arange(len(STRATEGIES))
bar_width = 0.52

totals   = (time_agg["seed_time_s_mean"] + time_agg["em_time_s_mean"]).reindex(STRATEGIES)
y_margin = totals.max() * 0.04

for i, strat in enumerate(STRATEGIES):
    color    = COLORS[strat]
    seed_val = time_agg.loc[strat, "seed_time_s_mean"]
    em_val   = time_agg.loc[strat, "em_time_s_mean"]
    total    = seed_val + em_val

    ax1.bar(i, seed_val, bar_width, color=color, zorder=3)
    ax1.bar(i, em_val, bar_width, bottom=seed_val,
            color=color, alpha=0.38, hatch="///", zorder=3)

    ax1.text(
        i, total + y_margin,
        f"{total:.2f}s",
        ha="center", va="bottom", fontsize=12, fontweight="bold", color="#333333",
    )

ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=13)
ax1.set_ylabel("Time (seconds)", fontsize=14)
ax1.set_title("Time Breakdown\n(Mean across all sizes & topologies)", fontsize=14, fontweight="bold")
ax1.set_xlabel("")

seed_patch = mpatches.Patch(facecolor="#888888", label="Seed Phase")
em_patch   = mpatches.Patch(facecolor="#888888", alpha=0.38, hatch="///", label="EM Phase")
ax1.legend(handles=[seed_patch, em_patch], fontsize=12, loc="upper right")

plt.tight_layout(pad=2.0)
out1 = os.path.join(OUT_DIR, "panel1_time_breakdown.png")
fig1.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved -> {out1}")

# ===========================================================================
# Panel 2 — Grouped Bar: EM Iterations per Topology at 1M Rows
# ===========================================================================
fig2, ax2 = plt.subplots(figsize=(9, 6.5))
fig2.suptitle(
    "k-means|| Seeding Strategy Benchmark  —  All Dataset Sizes",
    fontsize=15, fontweight="bold", y=1.02,
)

P2_STRATEGIES = ["kmeans||_fixed_R2", "kmeans||_adaptive"]
P2_LABELS     = ["k-means|| Fixed R=2", "k-means|| Adaptive"]
TOPOLOGIES    = ["standard", "anisotropic", "high_dim", "heavy_tail"]

plot_data = (
    df[df["strategy"].isin(P2_STRATEGIES)]
    .groupby(["topology", "strategy"])["em_iterations_mean"]
    .mean()
    .reset_index()
)

n_topo    = len(TOPOLOGIES)
n_strat   = len(P2_STRATEGIES)
bar_width = 0.35
x         = np.arange(n_topo)

for i, (strat, label) in enumerate(zip(P2_STRATEGIES, P2_LABELS)):
    subset  = plot_data[plot_data["strategy"] == strat].set_index("topology").reindex(TOPOLOGIES)
    means   = subset["em_iterations_mean"].values
    offsets = x + (i - (n_strat - 1) / 2) * bar_width

    ax2.bar(
        offsets, means, bar_width,
        color=COLORS[strat], label=label,
        zorder=3, edgecolor="white", linewidth=0.6,
    )

ax2.set_xticks(x)
ax2.set_xticklabels(
    ["Standard", "Anisotropic", "High-Dim", "Heavy Tail"],
    fontsize=12,
)
ax2.set_ylabel("EM Iterations to Convergence (mean)", fontsize=13)
ax2.set_xlabel("Dataset Topology", fontsize=13)
ax2.set_title(
    "EM Iterations per Topology\n(Mean across all dataset sizes)",
    fontsize=14, fontweight="bold",
)
ax2.legend(fontsize=12, loc="upper right")

plt.tight_layout(pad=2.0)
out2 = os.path.join(OUT_DIR, "panel2_em_stability.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved -> {out2}")

# ===========================================================================
# Panel 3 — Diverging Horizontal Bar: Inertia Δ vs Fixed R=2 Baseline
# ===========================================================================
fig3, ax3 = plt.subplots(figsize=(7, 6.5))
fig3.suptitle(
    "k-means|| Seeding Strategy Benchmark  —  All Dataset Sizes",
    fontsize=15, fontweight="bold", y=1.02,
)

pivot    = df.pivot_table(index="topology", columns="strategy", values="final_inertia_mean")
pct_diff = (
    (pivot["kmeans||_adaptive"] - pivot["kmeans||_fixed_R2"])
    / pivot["kmeans||_fixed_R2"]
) * 100
pct_diff = pct_diff.sort_values()

bar_colors = ["#2BAE99" if v < 0 else "#E05555" for v in pct_diff.values]
bars = ax3.barh(
    pct_diff.index, pct_diff.values,
    color=bar_colors, height=0.42, zorder=3, edgecolor="white", linewidth=0.8,
)

ax3.axvline(0, color="#333333", linewidth=1.4, linestyle="--", zorder=4)
ax3.set_xlabel("% Difference in Inertia\n(Adaptive − Fixed R=2) / Fixed R=2 × 100", fontsize=13)
ax3.set_ylabel("Dataset Topology", fontsize=14)
ax3.set_title(
    "Inertia vs Fixed R=2 Baseline\n(All Sizes  |  Adaptive strategy)",
    fontsize=14, fontweight="bold",
)
ax3.tick_params(axis="y", labelsize=13)
ax3.tick_params(axis="x", labelsize=11)

x_span  = pct_diff.abs().max()
x_pad   = x_span * 0.06
for bar, val in zip(bars, pct_diff.values):
    offset = x_pad if val >= 0 else -x_pad
    ha     = "left" if val >= 0 else "right"
    ax3.text(
        val + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}%",
        va="center", ha=ha, fontsize=12, fontweight="bold", color="#222222",
    )

x_lim_pad = x_span * 0.55
ax3.set_xlim(pct_diff.min() - x_lim_pad, max(pct_diff.max(), 0) + x_lim_pad)

green_patch = mpatches.Patch(color="#2BAE99", label="Adaptive lower inertia (better)")
red_patch   = mpatches.Patch(color="#E05555", label="Adaptive higher inertia (worse)")
ax3.legend(handles=[green_patch, red_patch], fontsize=11, loc="lower right")

plt.tight_layout(pad=2.0)
out3 = os.path.join(OUT_DIR, "panel3_inertia_delta.png")
fig3.savefig(out3, dpi=300, bbox_inches="tight")
plt.close(fig3)
print(f"Saved -> {out3}")
