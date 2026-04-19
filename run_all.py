"""
Orchestrator: run local + Spark benchmarks, print comparison table, save summary chart.
"""
import os
import csv
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def print_table(rows: list[dict], title: str):
    if not rows:
        print(f"  (no data for {title})")
        return

    cols = ["strategy", "dataset_size", "seed_rounds_used", "em_iterations",
            "seed_time_s", "em_time_s", "total_time_s", "peak_mem_mb", "inertia"]
    # Filter to columns that exist in these rows
    present = [c for c in cols if any(c in r for r in rows)]

    col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
                  for c in present}

    sep = "+-" + "-+-".join("-" * col_widths[c] for c in present) + "-+"
    header = "| " + " | ".join(c.ljust(col_widths[c]) for c in present) + " |"

    print(f"\n{'=' * len(sep)}")
    print(f" {title}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        line = "| " + " | ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in present) + " |"
        print(line)
    print(sep)


def save_chart(local_rows: list[dict], spark_rows: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def plot_panel(ax, rows, title):
        if not rows:
            ax.set_title(f"{title}\n(no data)")
            return

        strategies = sorted(set(r["strategy"] for r in rows))
        sizes = sorted(set(int(r["dataset_size"]) for r in rows))
        x = np.arange(len(sizes))
        width = 0.8 / max(len(strategies), 1)
        colors = plt.cm.tab10(np.linspace(0, 0.6, len(strategies)))

        for i, strat in enumerate(strategies):
            times = []
            for sz in sizes:
                match = [r for r in rows
                         if r["strategy"] == strat and int(r["dataset_size"]) == sz]
                t = float(match[0]["total_time_s"]) if match else 0.0
                times.append(t)
            offset = (i - len(strategies) / 2 + 0.5) * width
            bars = ax.bar(x + offset, times, width * 0.9, label=strat, color=colors[i])

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=15)
        ax.set_xlabel("Dataset size (n)")
        ax.set_ylabel("Total time (s)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plot_panel(axes[0], local_rows, "Local multicore benchmark")
    plot_panel(axes[1], spark_rows, "Spark benchmark")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nChart saved to {out}")


def main():
    print("=" * 60)
    print("  k-means|| Adaptive Early-Stopping Benchmark")
    print("=" * 60)

    # --- Part 1: local benchmark ---
    print("\n[1/2] Running local multicore benchmark ...")
    from benchmark_local import run_local_benchmark
    local_rows = run_local_benchmark()

    # --- Part 2: Spark benchmark ---
    print("\n[2/2] Running Spark benchmark ...")
    try:
        from benchmark_spark import run_spark_benchmark
        spark_rows = run_spark_benchmark()
    except Exception as e:
        print(f"  Spark benchmark failed: {e}")
        spark_rows = load_csv(os.path.join(RESULTS_DIR, "spark_benchmark.csv"))

    # --- Print tables ---
    print_table(local_rows, "Local Multicore Results")
    print_table(spark_rows, "Spark Results")

    # --- Chart ---
    save_chart(local_rows, spark_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
