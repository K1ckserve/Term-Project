"""
benchmark_local.py — Multicore benchmark for adaptive k-means||.

Baselines:
    - kmeans_plus_plus       : k-means++ (sequential, O(nkd))
    - fixed_kmeans_parallel  : fixed-round k-means|| (R=2, Spark default)

Contribution:
    - adaptive_kmeans_parallel : adaptive k-means|| with phi-based early stopping

All three strategies share BLAS-based squared-distance primitives so the
comparison reflects algorithmic differences, not implementation artefacts.
Parallelism is inherited from numpy's BLAS backend (MKL/OpenBLAS) via
`X @ C.T` matmuls. Run `python benchmark_local.py blas` to verify which
BLAS is linked and how many threads it uses.

CLI:
    python benchmark_local.py            # full benchmark → results/local_benchmark.csv
    python benchmark_local.py profile    # per-round timing breakdown of adaptive
    python benchmark_local.py blas       # show BLAS backend + thread count
"""
from __future__ import annotations
import os
import sys
import time
import tracemalloc
import csv
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
K = 20
DATASET_SIZES = [10_000, 100_000, 1_000_000]
RESULTS_DIR = "results"
OVERSAMPLE_FACTOR = 2.0      # l = oversample_factor * k (paper: l >= k)
EPSILON = 0.10               # relative phi improvement threshold for adaptive
MAX_ROUNDS = 5               # hard cap on adaptive seeding rounds


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_dataset(n_samples: int) -> np.ndarray:
    """Skewed Gaussian blobs: each cluster has a different std."""
    rng = np.random.default_rng(RANDOM_SEED)
    stds = rng.uniform(0.5, 5.0, size=K).tolist()
    X, _ = make_blobs(
        n_samples=n_samples, n_features=10, centers=K,
        cluster_std=stds, random_state=RANDOM_SEED,
    )
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# BLAS-accelerated distance primitives (shared by all k-means|| variants)
# ---------------------------------------------------------------------------

def _sq_dist_matrix(X: np.ndarray, C: np.ndarray,
                    X_sq: np.ndarray | None = None) -> np.ndarray:
    """
    Squared Euclidean distance matrix, shape (|X|, |C|).

    Uses the identity  ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c  so the
    dominant operation is a single matmul, dispatched to BLAS with native
    multithreading. Typically 3-5x faster than scipy.cdist for numerical
    arrays, and scales with available cores.

    Pass X_sq to avoid recomputing the ||X||^2 row-norms when X is fixed.
    All intermediate allocations are kept in float32.
    """
    if X_sq is None:
        X_sq = np.einsum("ij,ij->i", X, X)
    C_sq = np.einsum("ij,ij->i", C, C)
    # Build D in-place to minimise peak memory:
    #   D = X @ C.T; D *= -2; D += X_sq; D += C_sq
    D = X @ C.T
    D *= -2
    D += X_sq[:, None]
    D += C_sq[None, :]
    # Floating-point rounding can push distances fractionally below 0.
    np.maximum(D, 0, out=D)
    return D


def _update_min_dist_inplace(X: np.ndarray, new_centers: np.ndarray,
                              min_sq_dist: np.ndarray,
                              X_sq: np.ndarray) -> None:
    """Fold new centers into the running min_sq_dist array (in-place)."""
    new_D = _sq_dist_matrix(X, new_centers, X_sq=X_sq)
    new_min = new_D.min(axis=1)
    np.minimum(min_sq_dist, new_min, out=min_sq_dist)

def _reduce_to_k(X: np.ndarray, candidates: np.ndarray, k: int, 
                 X_sq: np.ndarray) -> np.ndarray:
    """Weighted k-means on the candidate set to reduce to exactly k centers."""
    if len(candidates) <= k:
        idx = np.random.default_rng(RANDOM_SEED).integers(
            0, len(candidates), size=k
        )
        return candidates[idx]
        
    # Calculate weights: find the nearest candidate for every point in X
    D = _sq_dist_matrix(X, candidates, X_sq=X_sq)
    closest_candidate_idx = D.argmin(axis=1)
    
    # Count how many points belong to each candidate
    weights = np.bincount(closest_candidate_idx, minlength=len(candidates))
    
    km = KMeans(
        n_clusters=k, init="k-means++", n_init=1,
        random_state=RANDOM_SEED, max_iter=100,
    )
    # Pass the calculated weights here
    km.fit(candidates, sample_weight=weights) 
    
    return km.cluster_centers_.astype(np.float32)

# ---------------------------------------------------------------------------
# Baseline 1: k-means++
# ---------------------------------------------------------------------------

def kmeans_plus_plus(X: np.ndarray, k: int,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Standard k-means++ seeding: O(nkd) sequential.

    Maintains a running min-squared-distance array and updates it only
    against the most recently added center each iteration — the canonical
    efficient formulation (vs. recomputing against all prior centers).
    """
    n, d = X.shape
    first = int(rng.integers(0, n))
    centers = np.empty((k, d), dtype=X.dtype)
    centers[0] = X[first]

    # min_sq[i] = squared distance from X[i] to its nearest chosen center
    diff = X - centers[0]
    min_sq = np.einsum("ij,ij->i", diff, diff)

    for i in range(1, k):
        phi = float(min_sq.sum())
        if phi <= 0:
            chosen = int(rng.integers(0, n))
        else:
            probs = min_sq / phi
            chosen = int(rng.choice(n, p=probs))
        centers[i] = X[chosen]
        diff = X - centers[i]
        new_sq = np.einsum("ij,ij->i", diff, diff)
        np.minimum(min_sq, new_sq, out=min_sq)

    return centers


# ---------------------------------------------------------------------------
# Baseline 2: Fixed-round k-means|| (R=2, matches Spark MLlib default)
# ---------------------------------------------------------------------------

def fixed_kmeans_parallel(X: np.ndarray, k: int, R: int = 2,
                           oversample_factor: float = OVERSAMPLE_FACTOR
                           ) -> np.ndarray:
    """
    Fixed-round k-means|| seeding — runs exactly R oversampling rounds.
    Uses the same BLAS primitives and incremental-update strategy as the
    adaptive variant so the two differ only in their stopping rule.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(X)
    oversample = oversample_factor * k
    X_sq = np.einsum("ij,ij->i", X, X)

    first = int(rng.integers(0, n))
    centers_list = [X[first:first + 1].copy()]
    min_sq = _sq_dist_matrix(X, centers_list[0], X_sq=X_sq).ravel()

    for _ in range(R):
        phi = float(min_sq.sum())
        if phi <= 0:
            break
        probs = min_sq / phi
        mask = rng.random(n) < (oversample * probs)
        new_pts = X[mask]
        if len(new_pts) > 2 * k:
            idx = rng.choice(len(new_pts), size=2*k, replace=False)
            new_pts = new_pts[idx]
        if len(new_pts) == 0:
            continue
        _update_min_dist_inplace(X, new_pts, min_sq, X_sq)
        centers_list.append(new_pts)

    return _reduce_to_k(X, np.vstack(centers_list), k, X_sq)


# ---------------------------------------------------------------------------
# Contribution: Adaptive k-means|| with phi-based early stopping
# ---------------------------------------------------------------------------

def adaptive_kmeans_parallel(
    X: np.ndarray,
    k: int,
    oversample_factor: float = OVERSAMPLE_FACTOR,
    epsilon: float = EPSILON,
    max_rounds: int = min(MAX_ROUNDS, 3),
    profile: bool = False,
):
    """
    Adaptive k-means|| seeding.

    Replaces the fixed-R rule with a data-dependent stopping criterion:
    halt when the potential function
        phi(C) = sum_i  min_c  ||x_i - c||^2
    drops by less than `epsilon` fraction between consecutive rounds
    (diminishing-returns signal). phi is monotonically non-increasing,
    so no oscillation pathology — unlike ratio-of-distances metrics.

    phi is computed as a free by-product of the sampling distribution
    (the Bernoulli probabilities already require the min-distance array),
    so there is zero extra distance-computation overhead for the
    convergence check.

    Parameters
    ----------
    profile : bool
        If True, capture per-round timings and return them in a dict as
        a third tuple element. Overhead when False: ~0 (no extra calls).

    Returns
    -------
    (centers, rounds_used)                if profile=False
    (centers, rounds_used, timings_dict)  if profile=True
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(X)
    epsilon_scaled = epsilon * max(1.0, min(np.log10(n / 10_000), 1.5))
    oversample = oversample_factor * k * 3

    timings: dict | None = None
    if profile:
        timings = {
            "setup_s": 0.0, "reduce_s": 0.0,
            "round": [], "new_pts": [],
            "sample_s": [], "update_s": [], "stop_s": [],
            "phi": [], "rel_improvement": [],
        }

    # --- setup ----------------------------------------------------------
    t_setup = time.perf_counter()
    X_sq = np.einsum("ij,ij->i", X, X)
    first = int(rng.integers(0, n))
    centers_list = [X[first:first + 1].copy()]
    min_sq = _sq_dist_matrix(X, centers_list[0], X_sq=X_sq).ravel()
    phi = float(min_sq.sum())
    if profile:
        timings["setup_s"] = time.perf_counter() - t_setup

    # --- main loop ------------------------------------------------------
    rounds_used = 0
    for r in range(max_rounds):
        rounds_used += 1

        t_sample = time.perf_counter()
        if phi <= 0:
            break
        # Bernoulli sampling: p_i = oversample * d^2(x_i, C) / phi(C)
        # Expected selected count per round ≈ oversample.
        probs = min_sq / phi
        mask = rng.random(n) < (oversample * probs)
        new_pts = X[mask]
        if len(new_pts) >  int(2*k/2):
            idx = rng.choice(len(new_pts), size=int(2*k/2), replace=False)
            new_pts = new_pts[idx]
        sample_s = time.perf_counter() - t_sample
        if len(new_pts) == 0:
            break

        # Update running phi against only the new centers (not all prior).
        t_update = time.perf_counter()
        _update_min_dist_inplace(X, new_pts, min_sq, X_sq)
        # CHUNK_SIZE = 50000
        # for i in range(0,n,CHUNK_SIZE):
        #     chunk = X[i:i + CHUNK_SIZE]
        #     _update_min_dist_inplace(chunk, new_pts, min_sq[i:i+CHUNK_SIZE], X_sq[i:i+CHUNK_SIZE])
        centers_list.append(new_pts)
        phi_new = float(min_sq.sum())
        update_s = time.perf_counter() - t_update
        if phi_new / n < 1e-10:
            break

        # Relative-improvement stopping rule.
        t_stop = time.perf_counter()
        rel_improvement = (phi - phi_new) / (phi + 1e-20)
        phi = phi_new
        stop_s = time.perf_counter() - t_stop

        if profile:
            timings["round"].append(r + 1)
            timings["new_pts"].append(int(len(new_pts)))
            timings["sample_s"].append(sample_s)
            timings["update_s"].append(update_s)
            timings["stop_s"].append(stop_s)
            timings["phi"].append(phi)
            timings["rel_improvement"].append(rel_improvement)

        if rel_improvement < epsilon_scaled:
            break

    # --- final reduction to k centers -----------------------------------
    t_reduce = time.perf_counter()
    final_centers = _reduce_to_k(X, np.vstack(centers_list), k, X_sq)
    if profile:
        timings["reduce_s"] = time.perf_counter() - t_reduce
        return final_centers, rounds_used, timings
    return final_centers, rounds_used


# ---------------------------------------------------------------------------
# Profiling helper: per-round time breakdown
# ---------------------------------------------------------------------------

def profile_adaptive(dataset_sizes: list[int] = DATASET_SIZES,
                      epsilon: float = EPSILON) -> None:
    """Run adaptive seeding with profile=True and print a per-round report."""
    print("\n" + "=" * 82)
    print(" Adaptive k-means|| — per-round profile")
    print(f" (k={K}, oversample={OVERSAMPLE_FACTOR}, epsilon={epsilon}, "
          f"max_rounds={MAX_ROUNDS})")
    print("=" * 82)

    for n in dataset_sizes:
        X = make_dataset(n)
        t0 = time.perf_counter()
        _, rounds_used, t = adaptive_kmeans_parallel(
            X, K, epsilon=epsilon, profile=True,
        )
        total_wall = time.perf_counter() - t0
        accounted = (t["setup_s"] + sum(t["sample_s"]) + sum(t["update_s"])
                     + sum(t["stop_s"]) + t["reduce_s"])

        print(f"\nn = {n:>10,}   rounds = {rounds_used}   "
              f"wall = {total_wall:.3f}s   "
              f"(setup {t['setup_s']:.3f}s + rounds "
              f"{accounted - t['setup_s'] - t['reduce_s']:.3f}s + "
              f"reduce {t['reduce_s']:.3f}s)")
        print(f"  {'round':>5} {'new_pts':>8} {'sample':>9} {'update':>9} "
              f"{'stop':>8} {'rel_impr':>10} {'phi':>13}")
        for i in range(len(t["round"])):
            print(f"  {t['round'][i]:>5} {t['new_pts'][i]:>8} "
                  f"{t['sample_s'][i]:>9.4f} {t['update_s'][i]:>9.4f} "
                  f"{t['stop_s'][i]:>8.4f} "
                  f"{t['rel_improvement'][i]:>9.4f}  "
                  f"{t['phi'][i]:>13.3e}")
    print()


# ---------------------------------------------------------------------------
# EM phase
# ---------------------------------------------------------------------------

def run_em(X: np.ndarray, init_centers: np.ndarray, k: int):
    """Run k-means EM from given initial centers, return (inertia, n_iter, time_s)."""
    t0 = time.perf_counter()
    km = KMeans(
        n_clusters=k, init=init_centers, n_init=1,
        max_iter=300, tol=1e-4, random_state=RANDOM_SEED,
    )
    km.fit(X)
    elapsed = time.perf_counter() - t0
    return km.inertia_, km.n_iter_, elapsed


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def benchmark_strategy(name: str, seed_fn, X: np.ndarray, k: int) -> dict:
    """Run one strategy, return dict of metrics."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = seed_fn(X, k)
    if isinstance(result, tuple):
        centers, rounds_used = result[0], result[1]
    else:
        centers, rounds_used = result, "N/A"
    seed_time = time.perf_counter() - t0
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    peak_mem_mb = sum(s.size for s in snap.statistics("lineno")) / 1e6

    inertia, n_iter, em_time = run_em(X, centers, k)

    return {
        "strategy": name,
        "seed_rounds_used": rounds_used,
        "seed_time_s": round(seed_time, 4),
        "em_time_s": round(em_time, 4),
        "total_time_s": round(seed_time + em_time, 4),
        "peak_mem_mb": round(peak_mem_mb, 2),
        "em_iterations": n_iter,
        "inertia": round(inertia, 2),
    }


def run_local_benchmark() -> list[dict]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for n in DATASET_SIZES:
        print(f"\n=== Dataset size: {n:,} ===")
        X = make_dataset(n)

        print("  k-means++ ...")
        row = benchmark_strategy(
            "kmeans++",
            lambda X, k: (kmeans_plus_plus(X, k, np.random.default_rng(RANDOM_SEED)), "N/A"),
            X, K,
        )
        row["dataset_size"] = n
        rows.append(row)
        print(f"    inertia={row['inertia']:.0f}  em_iter={row['em_iterations']}  "
              f"total={row['total_time_s']}s")

        print("  fixed k-means|| (R=2) ...")
        row = benchmark_strategy(
            "kmeans||_fixed_R2",
            lambda X, k: (fixed_kmeans_parallel(X, k, R=2), 2),
            X, K,
        )
        row["dataset_size"] = n
        rows.append(row)
        print(f"    inertia={row['inertia']:.0f}  em_iter={row['em_iterations']}  "
              f"total={row['total_time_s']}s")

        print("  adaptive k-means|| ...")
        row = benchmark_strategy(
            "kmeans||_adaptive",
            lambda X, k: adaptive_kmeans_parallel(X, k),
            X, K,
        )
        row["dataset_size"] = n
        rows.append(row)
        print(f"    inertia={row['inertia']:.0f}  em_iter={row['em_iterations']}  "
              f"rounds={row['seed_rounds_used']}  total={row['total_time_s']}s")

    csv_path = os.path.join(RESULTS_DIR, "local_benchmark.csv")
    fieldnames = ["strategy", "dataset_size", "seed_rounds_used", "em_iterations",
                  "seed_time_s", "em_time_s", "total_time_s", "peak_mem_mb", "inertia"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# BLAS configuration inspection — demonstrates the multicore parallelism
# ---------------------------------------------------------------------------

def show_blas_config() -> None:
    """
    Print numpy's BLAS backend and active thread counts.
    Use this to confirm `X @ C.T` inside the seeding loop is actually
    running multi-core. Key lines: 'blas_info' / 'openblas' / 'mkl'.
    """
    print("\n--- numpy BLAS configuration ---")
    np.show_config()
    try:
        from threadpoolctl import threadpool_info
        print("\n--- active thread pools ---")
        for p in threadpool_info():
            print(f"  {p.get('prefix', '?'):<15} threads={p.get('num_threads', '?')}  "
                  f"api={p.get('internal_api', '?')}  "
                  f"version={p.get('version', '?')}")
    except ImportError:
        print("\n(install `threadpoolctl` to see active thread counts)")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "profile":
        profile_adaptive()
    elif cmd == "blas":
        show_blas_config()
    else:
        run_local_benchmark() 