"""
visualize.py — Presentation-ready 3-panel benchmark infographic.

Reads results/local_benchmark.csv and writes results/benchmark_summary.png.

Panels:
  1. Stacked bar — seed + EM time breakdown at 1M rows
  2. Boxplot + swarmplot — EM iteration stability at 1M rows
  3. Diverging horizontal bar — inertia delta vs Fixed R=2 baseline at 1M rows
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
CSV_PATH = os.path.join(BASE_DIR, "results", "local_benchmark.csv")
OUT_PATH = os.path.join(BASE_DIR, "results", "benchmark_summary.png")

# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------
STRATEGIES = ["kmeans++", "kmeans||_fixed_R2", "kmeans||_adaptive"]
LABELS     = ["k-means++", "k-means||\nFixed R=2", "k-means||\nAdaptive"]

COLORS = {
    "kmeans++":          "#4C7BE8",   # blue
    "kmeans||_fixed_R2": "#E8873A",   # amber
    "kmeans||_adaptive": "#2BAE99",   # teal
}

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
df   = pd.read_csv(CSV_PATH)
df1m = df[df["dataset_size"] == 1_000_000].copy()

# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "axes.titlepad":    10,
    "axes.labelpad":     6,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
fig.suptitle(
    "k-means|| Seeding Strategy Benchmark  —  1 Million Rows",
    fontsize=17, fontweight="bold", y=1.02,
)

# ===========================================================================
# Panel 1 — Stacked Bar: Time Breakdown at 1M Rows
# ===========================================================================
ax1 = axes[0]

time_agg = (
    df1m.groupby("strategy")[["seed_time_s", "em_time_s"]]
    .mean()
    .reindex(STRATEGIES)
)

x          = np.arange(len(STRATEGIES))
bar_width  = 0.52

totals = (time_agg["seed_time_s"] + time_agg["em_time_s"]).reindex(STRATEGIES)
y_margin = totals.max() * 0.04   # 4% of max bar height — computed before drawing

for i, strat in enumerate(STRATEGIES):
    color    = COLORS[strat]
    seed_val = time_agg.loc[strat, "seed_time_s"]
    em_val   = time_agg.loc[strat, "em_time_s"]
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
ax1.set_title("Time Breakdown at 1M Rows\n(Mean across topologies)", fontsize=14, fontweight="bold")
ax1.set_xlabel("")

# legend patches
seed_patch = mpatches.Patch(facecolor="#888888", label="Seed Phase")
em_patch   = mpatches.Patch(facecolor="#888888", alpha=0.38, hatch="///", label="EM Phase")
ax1.legend(handles=[seed_patch, em_patch], fontsize=12, loc="upper right")

# ===========================================================================
# Panel 2 — Boxplot + Swarmplot: EM Iteration Stability at 1M Rows
# ===========================================================================
ax2 = axes[1]

plot_data = df1m[["strategy", "em_iterations", "dataset"]].copy()

sns.boxplot(
    data=plot_data, x="strategy", y="em_iterations",
    order=STRATEGIES, hue="strategy", hue_order=STRATEGIES,
    palette=COLORS, legend=False, ax=ax2,
    width=0.48, linewidth=1.8, fliersize=0, zorder=3,
)
sns.swarmplot(
    data=plot_data, x="strategy", y="em_iterations",
    order=STRATEGIES, ax=ax2,
    alpha=0.65, color=".20", size=9, zorder=4,
)

ax2.set_xticks(range(len(STRATEGIES)))
ax2.set_xticklabels(LABELS, fontsize=13)
ax2.set_ylabel("EM Iterations to Convergence", fontsize=14)
ax2.set_xlabel("")
ax2.set_title(
    "EM Iteration Stability at 1M Rows\n(Distribution across topologies)",
    fontsize=14, fontweight="bold",
)

# annotate each point with its topology name
for _, row in plot_data.iterrows():
    idx  = STRATEGIES.index(row["strategy"])
    ax2.text(
        idx + 0.18, row["em_iterations"],
        row["dataset"],
        va="center", ha="left", fontsize=9.5, color="#555555", style="italic",
    )

# ===========================================================================
# Panel 3 — Diverging Horizontal Bar: Inertia Δ vs Fixed R=2 Baseline
# ===========================================================================
ax3 = axes[2]

pivot    = df1m.pivot_table(index="dataset", columns="strategy", values="inertia")
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
    "Inertia vs Fixed R=2 Baseline\n(1M Rows  |  Adaptive strategy)",
    fontsize=14, fontweight="bold",
)
ax3.tick_params(axis="y", labelsize=13)
ax3.tick_params(axis="x", labelsize=11)

x_span  = pct_diff.abs().max()
x_pad   = x_span * 0.06   # 6% of data range for label clearance
for bar, val in zip(bars, pct_diff.values):
    offset = x_pad if val >= 0 else -x_pad
    ha     = "left" if val >= 0 else "right"
    ax3.text(
        val + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}%",
        va="center", ha=ha, fontsize=12, fontweight="bold", color="#222222",
    )

# ensure enough room for text beyond the bars
x_lim_pad = x_span * 0.55
ax3.set_xlim(pct_diff.min() - x_lim_pad, max(pct_diff.max(), 0) + x_lim_pad)

green_patch = mpatches.Patch(color="#2BAE99", label="Adaptive lower inertia (better)")
red_patch   = mpatches.Patch(color="#E05555", label="Adaptive higher inertia (worse)")
ax3.legend(handles=[green_patch, red_patch], fontsize=11, loc="lower right")

# ===========================================================================
# Finalize
# ===========================================================================
plt.tight_layout(pad=2.0)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUT_PATH}")
