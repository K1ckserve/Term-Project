"""
Local multicore benchmark: k-means++, fixed k-means||, adaptive k-means||.
"""
import os
import time
import tracemalloc
import numpy as np
import csv
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

RANDOM_SEED = 42
K = 20
DATASET_SIZES = [10_000, 100_000, 1_000_000]
RESULTS_DIR = "results"
OVERSAMPLE_FACTOR = 2.0

MAX_METRIC_SAMPLE = 10_000
_metric_rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_dataset(n_samples: int) -> np.ndarray:
    """Skewed Gaussian blobs: each cluster has a different std to stress-test seeding."""
    rng = np.random.default_rng(RANDOM_SEED)
    stds = rng.uniform(0.5, 5.0, size=K).tolist()
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=K,
        cluster_std=stds,
        random_state=RANDOM_SEED,
    )
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# k-means++ seeding (baseline 1)
# ---------------------------------------------------------------------------

def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Standard k-means++ seeding: O(nkd) sequential."""
    n = len(X)
    first = rng.integers(0, n)
    centers = [X[first]]

    for _ in range(k - 1):
        # Minimum squared distance from each point to the nearest existing center
        D = np.full(n, np.inf, dtype=np.float64)
        for c in centers:
            d = np.sum((X.astype(np.float64) - c.astype(np.float64)) ** 2, axis=1)
            D = np.minimum(D, d)
        probs = D / D.sum()
        chosen = rng.choice(n, p=probs)
        centers.append(X[chosen])

    return np.array(centers)


# ---------------------------------------------------------------------------
# Fixed-round k-means|| seeding (baseline 2, R=2 mirrors Spark default)
# ---------------------------------------------------------------------------

def _sample_candidates(X: np.ndarray, centers: np.ndarray, oversample: int,
                        seed: int) -> np.ndarray:
    """One oversampling round: sample each point with prob proportional to D^2."""
    rng = np.random.default_rng(seed)
    D = _min_sq_dist(X, centers)
    total = D.sum()
    if total == 0:
        return np.empty((0, X.shape[1]), dtype=X.dtype)
    probs = D / total
    # Expected number of samples = oversample; use multinomial for exact count
    mask = rng.random(len(X)) < (oversample * probs)
    return X[mask]


def _min_sq_dist(X, centers):
    # cdist computes pairwise distances without a 3D intermediate array
    # shape: (n_points, n_centers) — memory scales linearly, not cubically
    dists = cdist(X, centers, metric='sqeuclidean')
    return dists.min(axis=1)


def _reduce_to_k(candidates: np.ndarray, k: int) -> np.ndarray:
    """Weighted k-means on the candidate set to reduce to k centers."""
    if len(candidates) <= k:
        # Pad with random repeats if fewer candidates than k
        idx = np.random.default_rng(RANDOM_SEED).integers(0, len(candidates), size=k)
        return candidates[idx]
    km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=RANDOM_SEED, max_iter=100)
    km.fit(candidates)
    return km.cluster_centers_.astype(np.float32)


def fixed_kmeans_parallel(X: np.ndarray, k: int, R: int = 2,
                           oversample_factor: float = OVERSAMPLE_FACTOR,
                           n_jobs: int = -1) -> np.ndarray:
    """Fixed-round k-means|| seeding with R rounds."""
    rng = np.random.default_rng(RANDOM_SEED)
    # Start with a single random center
    centers = X[rng.integers(0, len(X))][np.newaxis, :]
    oversample = int(oversample_factor * k)

    for r in range(R):
        new_pts = _sample_candidates(X, centers, oversample, seed=RANDOM_SEED + r)
        if len(new_pts) > 0:
            centers = np.vstack([centers, new_pts])

    return _reduce_to_k(centers, k)


# ---------------------------------------------------------------------------
# Adaptive k-means|| seeding (the contribution)
# ---------------------------------------------------------------------------

def _metric_sample(X):
    if len(X) > MAX_METRIC_SAMPLE:
        idx = _metric_rng.choice(len(X), MAX_METRIC_SAMPLE, replace=False)
        return X[idx]
    return X


def _quality_metric(centers: np.ndarray) -> float:
    """
    Quality ratio: min pairwise inter-center distance / mean intra-cluster variance.

    A higher ratio means centers are well-separated relative to their spread —
    i.e., the seeding has converged to a good configuration.
    """
    n = len(centers)
    if n < 2:
        return 0.0

    # --- min pairwise inter-center distance ---
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((centers[i] - centers[j]) ** 2)
            if d < min_dist:
                min_dist = d
    min_dist = float(np.sqrt(min_dist))

    # --- mean intra-cluster variance ---
    # Assign each center to its nearest neighbor cluster, compute variance
    variances = []
    for i in range(n):
        dists = np.sum((centers - centers[i]) ** 2, axis=1)
        dists[i] = np.inf
        nearest = np.argmin(dists)
        pair = centers[[i, nearest]]
        variances.append(np.var(pair, axis=0).sum())
    mean_var = float(np.mean(variances))

    if mean_var < 1e-10:
        # Centers already collapsed — ratio is undefined; treat as converged
        return np.inf

    return min_dist / (mean_var + 1e-10)


def _auto_epsilon(n: int) -> float:
    if n < 50_000:
        return 0.05
    elif n < 500_000:
        return 0.02
    else:
        return 0.01


def adaptive_kmeans_parallel(
    X: np.ndarray,
    k: int,
    oversample_factor: float = OVERSAMPLE_FACTOR,
    epsilon: float | None = None,
    max_rounds: int = 5,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Adaptive k-means|| seeding with early stopping.

    After each oversampling round, compute a quality metric over the current
    candidate centers. Stop when |metric(round r) - metric(round r-1)| < epsilon,
    meaning further rounds are adding diminishing returns.
    
    Returns: initial_centers of shape [k, d].
    """
    if epsilon is None:
        epsilon = _auto_epsilon(len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    oversample = int(oversample_factor * k)

    # Step 1: bootstrap with a single random center
    centers = X[rng.integers(0, len(X))][np.newaxis, :]

    prev_metric = None
    rounds_used = 0

    for r in range(max_rounds):
        rounds_used += 1

        # Step 2: parallel oversampling — each job samples a partition of X
        # independently, then we union the results (embarrassingly parallel)
        n_jobs_actual = os.cpu_count() if n_jobs == -1 else n_jobs
        chunk_size = max(1, len(X) // n_jobs_actual)
        chunks = [
            X[i : i + chunk_size] for i in range(0, len(X), chunk_size)
        ]

        new_batches = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_sample_candidates)(chunk, centers, oversample, RANDOM_SEED + r * 1000 + ci)
            for ci, chunk in enumerate(chunks)
        )

        # Step 3: collect newly sampled candidates
        new_pts_list = [b for b in new_batches if len(b) > 0]
        if new_pts_list:
            new_pts = np.vstack(new_pts_list)
            centers = np.vstack([centers, new_pts])

        # Step 4: compute quality metric on the *reduced* candidate set
        # (reduce first so metric is computed on a representative k-sized set)
        candidate_centers = _reduce_to_k(_metric_sample(centers), k)
        metric = _quality_metric(candidate_centers)

        # Step 5: check convergence — stop if metric change is below epsilon
        if prev_metric is not None:
            delta = abs(metric - prev_metric)
            if delta < epsilon:
                # Converged — no need for more rounds
                break

        prev_metric = metric

    # Step 6: final reduction to exactly k centers
    return _reduce_to_k(centers, k), rounds_used


# ---------------------------------------------------------------------------
# EM phase runner (shared)
# ---------------------------------------------------------------------------

def run_em(X: np.ndarray, init_centers: np.ndarray, k: int):
    """Run k-means EM from given initial centers, return (inertia, n_iter, time_s)."""
    t0 = time.perf_counter()
    km = KMeans(
        n_clusters=k,
        init=init_centers,
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_SEED,
    )
    km.fit(X)
    elapsed = time.perf_counter() - t0
    return km.inertia_, km.n_iter_, elapsed


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_strategy(name: str, seed_fn, X: np.ndarray, k: int):
    """
    Run one strategy, return dict of metrics.
    seed_fn(X, k) -> (centers, rounds_used)  OR  centers (rounds_used defaults to N/A)
    """
    tracemalloc.start()
    t_seed_start = time.perf_counter()

    result = seed_fn(X, k)
    if isinstance(result, tuple):
        centers, rounds_used = result
    else:
        centers, rounds_used = result, "N/A"

    seed_time = time.perf_counter() - t_seed_start
    mem_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    peak_mem_mb = sum(s.size for s in mem_snapshot.statistics("lineno")) / 1e6

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

        # Baseline 1: k-means++
        print("  Running k-means++ ...")
        row = benchmark_strategy(
            "kmeans++",
            lambda X, k: (kmeans_plus_plus(X, k, np.random.default_rng(RANDOM_SEED)), "N/A"),
            X, K,
        )
        row["dataset_size"] = n
        rows.append(row)
        print(f"    inertia={row['inertia']}, em_iter={row['em_iterations']}, "
              f"total={row['total_time_s']}s")

        # Baseline 2: fixed k-means|| R=2
        print("  Running fixed k-means|| (R=2) ...")
        row = benchmark_strategy(
            "kmeans||_fixed_R2",
            lambda X, k: (fixed_kmeans_parallel(X, k, R=2), 2),
            X, K,
        )
        row["dataset_size"] = n
        rows.append(row)
        print(f"    inertia={row['inertia']}, em_iter={row['em_iterations']}, "
              f"total={row['total_time_s']}s")

        # Contribution: adaptive k-means||
        print("  Running adaptive k-means|| ...")
        row = benchmark_strategy(
            "kmeans||_adaptive",
            lambda X, k: adaptive_kmeans_parallel(X, k),
            X, K,
        )
        row["dataset_size"] = n
        rows.append(row)
        print(f"    inertia={row['inertia']}, em_iter={row['em_iterations']}, "
              f"rounds={row['seed_rounds_used']}, total={row['total_time_s']}s")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "local_benchmark.csv")
    fieldnames = ["strategy", "dataset_size", "seed_rounds_used", "em_iterations",
                  "seed_time_s", "em_time_s", "total_time_s", "peak_mem_mb", "inertia"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nLocal results saved to {csv_path}")
    return rows


if __name__ == "__main__":
    run_local_benchmark()
