#!/usr/bin/env python3
"""
bench_graphgp_scalar_vs_vector.py

Benchmarks scalar vs vector GraphGP generation performance.

What it measures (per problem size):
- Graph construction time
- First-call time (JIT compile + execution) for:
    * scalar generate()
    * vector generate_vector() with separable covariance (k(r)*I)
    * vector generate_vector() with divergence-free analytic RBF tensor (optional)
- Steady-state execution time (mean ± std over repeats) after compilation
- Throughput (points / second)
- Ratios (vector/scalar)

Outputs:
- CSV: bench_out/results.csv
- Plots: bench_out/*.png
- Console table summary

Usage examples:
  python bench_graphgp_scalar_vs_vector.py --n-list 8 10 12 14 --k 10 --n0-frac 0.15 --repeats 5
  python bench_graphgp_scalar_vs_vector.py --n-list 10 12 14 --vector-modes separable divfree --periodic-vector

Notes:
- For fair comparisons, use the same points and same graph for scalar/vector.
- Scalar GraphGP uses a discretized radial covariance (bins/vals) and interpolation (jnp.interp).
  Vector "separable" mode uses the same scalar bins/vals under the hood, so it isolates vector overhead.
- Vector "divfree" mode uses an analytic tensor (no interp) like your demo-vector.py.

"""

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt

import graphgp as gp


# -----------------------------
# Covariances
# -----------------------------

def make_div_free_rbf_covariance(ell: float, sigma2: float = 1.0, periodic: bool = False):
    """Analytic divergence-free RBF tensor used in demo-vector.py: cov_elem(x,y)->(3,3)."""
    ell2 = ell * ell

    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)  # assumes coords in [0,1)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)

        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))

    return cov_elem


def make_scalar_rbf_bins(points: jnp.ndarray, ell: float, sigma2: float, n_bins: int, jitter: float):
    """Discretized scalar RBF bins/vals for GraphGP scalar code."""
    mins = jnp.min(points, axis=0)
    maxs = jnp.max(points, axis=0)
    r_max = float(jnp.linalg.norm(maxs - mins)) + 1e-6
    r_min = 1e-6
    return gp.extras.rbf_kernel(
        variance=float(sigma2),
        scale=float(ell),
        r_min=float(r_min),
        r_max=float(r_max),
        n_bins=int(n_bins),
        jitter=float(jitter),
    )


def make_separable_vector_covariance_from_scalar_bins(covariance_bins_vals, ncomp: int = 3):
    """Vector covariance: Cov(v_i(x), v_j(y)) = k(|x-y|) * δ_ij."""
    if hasattr(gp, "make_separable_vector_covariance"):
        return gp.make_separable_vector_covariance(covariance_bins_vals, jnp.eye(ncomp))
    # fallback: helper exists in refine-vector.py but may not be re-exported
    from graphgp.refine import make_separable_vector_covariance

    return make_separable_vector_covariance(covariance_bins_vals, jnp.eye(ncomp))


# -----------------------------
# Timing helpers
# -----------------------------

def _block_until_ready(x):
    return jax.tree_util.tree_map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a,
        x,
    )


def time_first_call(fn, *args) -> float:
    t0 = time.perf_counter()
    out = fn(*args)
    _block_until_ready(out)
    return time.perf_counter() - t0


def time_repeated(fn, args: Tuple, repeats: int) -> Tuple[float, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        _block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times, ddof=1) if repeats > 1 else 0.0)


# -----------------------------
# Benchmark
# -----------------------------


@dataclass
class BenchRow:
    n: int
    N: int
    k: int
    n0: int
    device: str

    t_build_graph: float

    t_first_scalar: float
    t_run_scalar_mean: float
    t_run_scalar_std: float

    t_first_vec_sep: float
    t_run_vec_sep_mean: float
    t_run_vec_sep_std: float

    t_first_vec_divfree: float
    t_run_vec_divfree_mean: float
    t_run_vec_divfree_std: float

    speedup_sep_vs_scalar: float
    speedup_divfree_vs_scalar: float
    pts_per_s_scalar: float
    pts_per_s_vec_sep: float
    pts_per_s_vec_divfree: float


def benchmark_one(
    n: int,
    k: int,
    n0: int,
    repeats: int,
    ell: float,
    sigma2: float,
    n_bins: int,
    scalar_jitter: float,
    vector_modes: List[str],
    periodic_vector: bool,
) -> BenchRow:
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))
    N = int(points.shape[0])

    # Build graph (host time)
    t0 = time.perf_counter()
    graph = gp.build_graph(points, n0=n0, k=k)
    t_build = time.perf_counter() - t0

    # Scalar covariance (bins/vals)
    cov_scalar = make_scalar_rbf_bins(points, ell=ell, sigma2=sigma2, n_bins=n_bins, jitter=scalar_jitter)

    # White noise parameters
    key = jr.PRNGKey(0)
    key, k1, k2 = jr.split(key, 3)
    xi_scalar = jr.normal(k1, (N,))
    xi_vec3 = jr.normal(k2, (N, 3))

    # JIT wrappers (first call includes compilation)
    scalar_fn = jax.jit(lambda xi: gp.generate(graph, cov_scalar, xi, cuda=False, fast_jit=True))

    vec_sep_cov = make_separable_vector_covariance_from_scalar_bins(cov_scalar, ncomp=3)
    vec_sep_fn = jax.jit(lambda xi: gp.generate_vector(graph, vec_sep_cov, xi, cuda=False, fast_jit=True))

    if "divfree" in vector_modes:
        vec_div_cov = make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=periodic_vector)
        vec_div_fn = jax.jit(lambda xi: gp.generate_vector(graph, vec_div_cov, xi, cuda=False, fast_jit=True))
    else:
        vec_div_fn = None

    # Timings
    t_first_scalar = time_first_call(scalar_fn, xi_scalar)
    t_run_scalar_mean, t_run_scalar_std = time_repeated(scalar_fn, (xi_scalar,), repeats)

    t_first_vec_sep = time_first_call(vec_sep_fn, xi_vec3)
    t_run_vec_sep_mean, t_run_vec_sep_std = time_repeated(vec_sep_fn, (xi_vec3,), repeats)

    if vec_div_fn is not None:
        t_first_vec_div = time_first_call(vec_div_fn, xi_vec3)
        t_run_vec_div_mean, t_run_vec_div_std = time_repeated(vec_div_fn, (xi_vec3,), repeats)
    else:
        t_first_vec_div = float("nan")
        t_run_vec_div_mean = float("nan")
        t_run_vec_div_std = float("nan")

    # Derived
    speedup_sep = t_run_scalar_mean / t_run_vec_sep_mean if t_run_vec_sep_mean > 0 else float("nan")
    speedup_div = (
        t_run_scalar_mean / t_run_vec_div_mean
        if (vec_div_fn is not None and t_run_vec_div_mean > 0)
        else float("nan")
    )

    pts_s_scalar = N / t_run_scalar_mean
    pts_s_sep = N / t_run_vec_sep_mean
    pts_s_div = (N / t_run_vec_div_mean) if vec_div_fn is not None else float("nan")

    return BenchRow(
        n=n,
        N=N,
        k=k,
        n0=n0,
        device=jax.default_backend(),
        t_build_graph=t_build,
        t_first_scalar=t_first_scalar,
        t_run_scalar_mean=t_run_scalar_mean,
        t_run_scalar_std=t_run_scalar_std,
        t_first_vec_sep=t_first_vec_sep,
        t_run_vec_sep_mean=t_run_vec_sep_mean,
        t_run_vec_sep_std=t_run_vec_sep_std,
        t_first_vec_divfree=t_first_vec_div,
        t_run_vec_divfree_mean=t_run_vec_div_mean,
        t_run_vec_divfree_std=t_run_vec_div_std,
        speedup_sep_vs_scalar=speedup_sep,
        speedup_divfree_vs_scalar=speedup_div,
        pts_per_s_scalar=pts_s_scalar,
        pts_per_s_vec_sep=pts_s_sep,
        pts_per_s_vec_divfree=pts_s_div,
    )


def write_csv(rows: List[BenchRow], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def plot_results(rows: List[BenchRow], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    N = np.array([r.N for r in rows])

    # steady-state runtimes
    tS = np.array([r.t_run_scalar_mean for r in rows])
    tV = np.array([r.t_run_vec_sep_mean for r in rows])
    tD = np.array([r.t_run_vec_divfree_mean for r in rows])

    # first-call
    fS = np.array([r.t_first_scalar for r in rows])
    fV = np.array([r.t_first_vec_sep for r in rows])
    fD = np.array([r.t_first_vec_divfree for r in rows])

    # runtime vs N
    plt.figure(figsize=(7, 4))
    plt.plot(N, tS, marker="o", label="scalar")
    plt.plot(N, tV, marker="o", label="vector-separable")
    if not np.all(np.isnan(tD)):
        plt.plot(N, tD, marker="o", label="vector-divfree")
    plt.xlabel("N points")
    plt.ylabel("steady-state time per call [s]")
    plt.title("GraphGP: scalar vs vector runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "runtime_vs_N.png", dpi=180)
    plt.close()

    # first-call vs N
    plt.figure(figsize=(7, 4))
    plt.plot(N, fS, marker="o", label="scalar first-call")
    plt.plot(N, fV, marker="o", label="vector-separable first-call")
    if not np.all(np.isnan(fD)):
        plt.plot(N, fD, marker="o", label="vector-divfree first-call")
    plt.xlabel("N points")
    plt.ylabel("first-call time (compile+run) [s]")
    plt.title("GraphGP: compile+run overhead")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "firstcall_vs_N.png", dpi=180)
    plt.close()

    # ratio
    plt.figure(figsize=(7, 4))
    plt.plot(N, tV / tS, marker="o", label="vec-sep / scalar")
    if not np.all(np.isnan(tD)):
        plt.plot(N, tD / tS, marker="o", label="vec-divfree / scalar")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("N points")
    plt.ylabel("runtime ratio")
    plt.title("GraphGP: vector slowdown factor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ratio_vs_N.png", dpi=180)
    plt.close()

    # throughput
    ptsS = np.array([r.pts_per_s_scalar for r in rows])
    ptsV = np.array([r.pts_per_s_vec_sep for r in rows])
    ptsD = np.array([r.pts_per_s_vec_divfree for r in rows])
    plt.figure(figsize=(7, 4))
    plt.plot(N, ptsS, marker="o", label="scalar")
    plt.plot(N, ptsV, marker="o", label="vector-separable")
    if not np.all(np.isnan(ptsD)):
        plt.plot(N, ptsD, marker="o", label="vector-divfree")
    plt.xlabel("N points")
    plt.ylabel("throughput [points/s]")
    plt.title("GraphGP: throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "throughput_vs_N.png", dpi=180)
    plt.close()


def print_summary(rows: List[BenchRow]):
    print("n   N       n0  k   build[s]  scalar[s]   vec-sep[s]  vec-div[s]  ratio(sep)  ratio(div)")
    for r in rows:
        ratio_sep = r.t_run_vec_sep_mean / r.t_run_scalar_mean
        ratio_div = (
            r.t_run_vec_divfree_mean / r.t_run_scalar_mean
            if np.isfinite(r.t_run_vec_divfree_mean)
            else float("nan")
        )
        print(
            f"{r.n:<3d} {r.N:<7d} {r.n0:<3d} {r.k:<3d} {r.t_build_graph:>8.3f} "
            f"{r.t_run_scalar_mean:>9.4f} {r.t_run_vec_sep_mean:>10.4f} {r.t_run_vec_divfree_mean:>10.4f} "
            f"{ratio_sep:>10.2f} {ratio_div:>10.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[8, 10, 12, 14], help="Grid sizes; N=n^3")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n0", type=int, default=None)
    ap.add_argument("--n0-frac", type=float, default=0.15)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ell", type=float, default=0.075)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--n-bins", type=int, default=2048)
    ap.add_argument("--scalar-jitter", type=float, default=1e-6)
    ap.add_argument(
        "--vector-modes",
        nargs="+",
        default=["separable", "divfree"],
        choices=["separable", "divfree"],
    )
    ap.add_argument("--periodic-vector", action="store_true")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    try:
        print("JAX devices:", jax.devices())
    except Exception:
        pass

    rows: List[BenchRow] = []
    for n in args.n_list:
        N = n**3
        n0 = args.n0 if args.n0 is not None else max(10, int(args.n0_frac * N))
        rows.append(
            benchmark_one(
                n=n,
                k=args.k,
                n0=n0,
                repeats=args.repeats,
                ell=args.ell,
                sigma2=args.sigma2,
                n_bins=args.n_bins,
                scalar_jitter=args.scalar_jitter,
                vector_modes=args.vector_modes,
                periodic_vector=args.periodic_vector,
            )
        )

    outdir = Path("bench_out")
    write_csv(rows, outdir / "results.csv")
    plot_results(rows, outdir)
    print_summary(rows)

    print("\nWrote:")
    print(" -", outdir / "results.csv")
    for p in sorted(outdir.glob("*.png")):
        print(" -", p)


if __name__ == "__main__":
    main()
