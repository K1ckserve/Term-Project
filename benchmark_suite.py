"""
benchmark_suite.py — Research-level standardized testing suite for k-means seeding.

Evaluates three strategies across five data topologies and three dataset sizes,
running N_TRIALS=5 independent trials per combination for statistical rigor.
Each combination yields μ and σ over total_time_s, inertia, em_iterations,
and seed_alignment_error — the geometric quality diagnostic absent from the
original benchmark.

CLI:
    python benchmark_suite.py           # full run → results/suite_aggregate.csv
                                        #           + results/suite_raw_trials.json
    python benchmark_suite.py quick     # 10k + 100k, 2 trials (fast dev loop)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import tracemalloc

import numpy as np
from sklearn.datasets import make_blobs

from benchmark_local import (
    K as DEFAULT_K,
    EPSILON,
    MAX_ROUNDS,
    OVERSAMPLE_FACTOR,
    RANDOM_SEED,
    adaptive_kmeans_parallel,
    fixed_kmeans_parallel,
    kmeans_plus_plus,
    run_em,
)

# ---------------------------------------------------------------------------
# Suite-level constants
# ---------------------------------------------------------------------------
N_TRIALS       = 5
TRIAL_SEEDS    = [42, 137, 271, 512, 999]   # one per trial
DATASET_SIZES  = [10_000, 100_000, 1_000_000]
TOPOLOGIES     = ["standard", "imbalanced", "anisotropic", "high_dim", "heavy_tail"]
K              = 20           # clusters for all topologies
HIGH_DIM       = 100          # projection target dimensionality for high_dim
NOISE_FRACTION = 0.10         # fraction of points replaced by uniform noise in heavy_tail
DATASET_SEED   = 42           # fixed: geometry is held constant across trials
RESULTS_DIR    = "results"


# ---------------------------------------------------------------------------
# DatasetSuite
# ---------------------------------------------------------------------------

class DatasetSuite:
    """
    Generates labelled datasets of varying geometries.
    Every method returns (X, true_centers) where true_centers are the
    ground-truth generating positions used to compute seed_alignment_error.
    """

    @staticmethod
    def standard(n: int, seed: int = DATASET_SEED):
        """Symmetric, well-separated Gaussian blobs in 10 dimensions."""
        rng = np.random.default_rng(seed)
        stds = rng.uniform(0.5, 3.0, size=K).tolist()
        X, _, centers = make_blobs(
            n_samples=n, n_features=10, centers=K,
            cluster_std=stds, random_state=seed, return_centers=True,
        )
        return X.astype(np.float32), centers.astype(np.float32)

    @staticmethod
    def imbalanced(n: int, seed: int = DATASET_SEED):
        """3 dominant clusters (90 % of points) + 17 sparse clusters (10 %)."""
        rng = np.random.default_rng(seed)
        K_large, K_small = 3, 17          # K_large + K_small == K
        n_large = int(n * 0.90)
        n_small = n - n_large

        X_large, _, c_large = make_blobs(
            n_samples=n_large, n_features=10, centers=K_large,
            cluster_std=rng.uniform(0.5, 2.0, size=K_large).tolist(),
            random_state=seed, return_centers=True,
        )
        X_small, _, c_small = make_blobs(
            n_samples=n_small, n_features=10, centers=K_small,
            cluster_std=rng.uniform(0.1, 0.5, size=K_small).tolist(),
            random_state=seed + 1, return_centers=True,
        )
        X = np.vstack([X_large, X_small]).astype(np.float32)
        rng.shuffle(X)
        centers = np.vstack([c_large, c_small]).astype(np.float32)
        return X, centers

    @staticmethod
    def anisotropic(n: int, seed: int = DATASET_SEED):
        """Elongated blobs: a random per-axis stretch is applied to standard blobs."""
        rng = np.random.default_rng(seed)
        X_raw, _, centers_raw = make_blobs(
            n_samples=n, n_features=10, centers=K,
            cluster_std=1.0, random_state=seed, return_centers=True,
        )
        # Diagonal stretch matrix — different scale per axis to break sphericity
        stretch = np.diag(rng.uniform(0.1, 5.0, size=10)).astype(np.float32)
        X = (X_raw @ stretch).astype(np.float32)
        centers = (centers_raw @ stretch).astype(np.float32)
        return X, centers

    @staticmethod
    def high_dim(n: int, seed: int = DATASET_SEED):
        """Standard blobs projected from 10-d into HIGH_DIM dimensions."""
        rng = np.random.default_rng(seed)
        X_raw, _, centers_raw = make_blobs(
            n_samples=n, n_features=10, centers=K,
            cluster_std=1.0, random_state=seed, return_centers=True,
        )
        # Johnson-Lindenstrauss-style projection: scale by 1/sqrt(d_out) to
        # approximately preserve pairwise distances in expectation.
        proj = (rng.standard_normal((10, HIGH_DIM)) / np.sqrt(HIGH_DIM)).astype(np.float32)
        X = (X_raw @ proj).astype(np.float32)
        centers = (centers_raw @ proj).astype(np.float32)
        return X, centers

    @staticmethod
    def heavy_tail(n: int, seed: int = DATASET_SEED):
        """Gaussian blobs with NOISE_FRACTION of points replaced by uniform outliers."""
        rng = np.random.default_rng(seed)
        X, _, centers = make_blobs(
            n_samples=n, n_features=10, centers=K,
            cluster_std=1.0, random_state=seed, return_centers=True,
        )
        X = X.astype(np.float32)
        n_noise = int(n * NOISE_FRACTION)
        noise_idx = rng.choice(n, size=n_noise, replace=False)
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        X[noise_idx] = rng.uniform(lo, hi, size=(n_noise, 10)).astype(np.float32)
        return X, centers.astype(np.float32)


TOPOLOGY_FNS: dict[str, callable] = {
    "standard":    DatasetSuite.standard,
    "imbalanced":  DatasetSuite.imbalanced,
    "anisotropic": DatasetSuite.anisotropic,
    "high_dim":    DatasetSuite.high_dim,
    "heavy_tail":  DatasetSuite.heavy_tail,
}


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def seed_alignment_error(seeds: np.ndarray, true_centers: np.ndarray) -> float:
    """
    Mean minimum Euclidean distance from each chosen seed to its nearest true center.

    Lower is better. Proves whether the seeding phase recovers the underlying
    cluster structure prior to EM — independent of convergence quality.

    Uses the same x·c BLAS identity as _sq_dist_matrix so it stays consistent
    with the distance metric used throughout the algorithms.
    """
    s = seeds.astype(np.float64)
    t = true_centers.astype(np.float64)
    s_sq = np.einsum("ij,ij->i", s, s)
    t_sq = np.einsum("ij,ij->i", t, t)
    D = s @ t.T          # (K, K_true)
    D *= -2
    D += s_sq[:, None]
    D += t_sq[None, :]
    np.maximum(D, 0.0, out=D)
    np.sqrt(D, out=D)
    return float(D.min(axis=1).mean())


# ---------------------------------------------------------------------------
# Strategy factories — return (name, callable) pairs wired to a trial seed
# ---------------------------------------------------------------------------

def make_strategies(trial_seed: int) -> list[tuple[str, callable]]:
    """
    Build seed-function callables for a single trial.

    Each callable matches the signature  f(X, k) → (centers, rounds_or_NA).
    The dataset geometry is identical across trials; only the algorithm's
    internal RNG varies so that σ reflects pure seeding stochasticity.
    """
    return [
        (
            "kmeans++",
            lambda X, k, s=trial_seed: (
                kmeans_plus_plus(X, k, np.random.default_rng(s)),
                "N/A",
            ),
        ),
        (
            "kmeans||_fixed_R2",
            lambda X, k, s=trial_seed: (
                fixed_kmeans_parallel(X, k, R=2, seed=s),
                2,
            ),
        ),
        (
            "kmeans||_adaptive",
            lambda X, k, s=trial_seed: adaptive_kmeans_parallel(X, k, seed=s),
        ),
    ]


# ---------------------------------------------------------------------------
# Trial harness
# ---------------------------------------------------------------------------

def run_trial(
    strategy_name: str,
    seed_fn,
    X: np.ndarray,
    true_centers: np.ndarray,
    k: int,
    topology: str,
    n_samples: int,
    trial_idx: int,
) -> dict:
    """
    Execute one trial: seed phase → EM phase → full diagnostic dict.

    tracemalloc wraps only the seed phase to avoid inflating em_time_s.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = seed_fn(X, k)
    seed_time = time.perf_counter() - t0
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    peak_mem_mb = sum(s.size for s in snap.statistics("lineno")) / 1e6

    if isinstance(result, tuple):
        seeds, rounds_used = result[0], result[1]
    else:
        seeds, rounds_used = result, "N/A"

    align_err = seed_alignment_error(seeds, true_centers)
    inertia, n_iter, em_time = run_em(X, seeds, k)

    return {
        "topology":            topology,
        "n_samples":           n_samples,
        "strategy":            strategy_name,
        "trial_idx":           trial_idx,
        "seed_time_s":         round(seed_time, 5),
        "em_time_s":           round(em_time, 5),
        "total_time_s":        round(seed_time + em_time, 5),
        "em_iterations":       int(n_iter),
        "final_inertia":       round(float(inertia), 2),
        "seed_alignment_error": round(align_err, 6),
        "seed_rounds_used":    rounds_used if rounds_used != "N/A" else None,
        "peak_mem_mb":         round(peak_mem_mb, 2),
    }


# ---------------------------------------------------------------------------
# Statistical aggregation
# ---------------------------------------------------------------------------

_AGG_FIELDS = [
    "seed_time_s", "em_time_s", "total_time_s",
    "em_iterations", "final_inertia", "seed_alignment_error",
]


def aggregate_trials(trials: list[dict]) -> dict:
    """
    Compute μ and σ (ddof=1) across N trials for one (topology, n_samples, strategy) group.
    Returns a flat dict ready to be written as a CSV row.
    """
    base = {k: trials[0][k] for k in ("topology", "n_samples", "strategy")}
    for field in _AGG_FIELDS:
        vals = np.array([t[field] for t in trials], dtype=float)
        base[f"{field}_mean"] = round(float(vals.mean()), 6)
        base[f"{field}_std"]  = round(float(vals.std(ddof=1)), 6)

    rounds_vals = [t["seed_rounds_used"] for t in trials
                   if t["seed_rounds_used"] is not None]
    base["rounds_mean"] = (round(float(np.mean(rounds_vals)), 2)
                           if rounds_vals else None)
    return base


# ---------------------------------------------------------------------------
# Aggregate CSV column order
# ---------------------------------------------------------------------------

_STRATEGY_NAMES = ["kmeans++", "kmeans||_fixed_R2", "kmeans||_adaptive"]

_AGG_CSV_FIELDS = (
    ["topology", "n_samples", "strategy"]
    + [f"{f}_{s}" for f in _AGG_FIELDS for s in ("mean", "std")]
    + ["rounds_mean"]
)


# ---------------------------------------------------------------------------
# Main suite runner
# ---------------------------------------------------------------------------

def run_suite(sizes: list[int] = DATASET_SIZES, n_trials: int = N_TRIALS) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_raw: list[dict] = []
    all_agg: list[dict] = []
    seeds = TRIAL_SEEDS[:n_trials]

    total_combos = len(TOPOLOGIES) * len(sizes)
    combo_idx = 0

    for topology in TOPOLOGIES:
        gen_fn = TOPOLOGY_FNS[topology]

        for n in sizes:
            combo_idx += 1
            print(f"\n[{combo_idx}/{total_combos}]  topology={topology:<12}  n={n:>10,}")

            # Generate once; hold geometry constant across all trials so that
            # trial-to-trial variance reflects algorithm randomness, not data shuffle.
            X, true_centers = gen_fn(n, seed=DATASET_SEED)

            combo_raw: dict[str, list[dict]] = {name: [] for name in _STRATEGY_NAMES}

            for trial_idx, trial_seed in enumerate(seeds):
                strategies = make_strategies(trial_seed)
                for name, seed_fn in strategies:
                    print(
                        f"  trial {trial_idx + 1}/{n_trials}  "
                        f"{name:<22} ...",
                        end=" ",
                        flush=True,
                    )
                    row = run_trial(
                        strategy_name=name,
                        seed_fn=seed_fn,
                        X=X,
                        true_centers=true_centers,
                        k=K,
                        topology=topology,
                        n_samples=n,
                        trial_idx=trial_idx,
                    )
                    all_raw.append(row)
                    combo_raw[name].append(row)
                    rounds_str = (
                        f"  rounds={row['seed_rounds_used']}"
                        if row["seed_rounds_used"] is not None
                        else ""
                    )
                    print(
                        f"inertia={row['final_inertia']:>14.0f}  "
                        f"align_err={row['seed_alignment_error']:.4f}  "
                        f"em_iter={row['em_iterations']:>3}  "
                        f"total={row['total_time_s']:.3f}s"
                        f"{rounds_str}"
                    )

            for name in _STRATEGY_NAMES:
                all_agg.append(aggregate_trials(combo_raw[name]))

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    raw_path = os.path.join(RESULTS_DIR, "suite_raw_trials.json")
    with open(raw_path, "w") as f:
        json.dump(all_raw, f, indent=2)
    print(f"\nRaw trials  -> {raw_path}  ({len(all_raw)} rows)")

    agg_path = os.path.join(RESULTS_DIR, "suite_aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AGG_CSV_FIELDS)
        w.writeheader()
        w.writerows(all_agg)
    print(f"Aggregate   -> {agg_path}  ({len(all_agg)} rows)")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "quick":
        # Fast smoke-test: skip 1M, only 2 trials
        run_suite(sizes=[10_000, 100_000], n_trials=2)
    else:
        run_suite()
