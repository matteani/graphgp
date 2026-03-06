#!/usr/bin/env python3
"""bench_graphgp_breakdown.py

Break down GraphGP runtime into:
- dense stage (Cholesky on first n0 points)
- refinement stage (conditional generation for the remaining N-n0 points)

Compares:
- scalar
- vector-separable (k(r) * I)
- vector-divfree (analytic divergence-free tensor)

Outputs:
- CSV: bench_out/breakdown.csv
- Plot: bench_out/breakdown_stacked.png

Example:
  python bench_graphgp_breakdown.py --n 14 --k 10 --n0 200 --repeats 5

"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt

import graphgp as gp


def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=False):
    ell2 = ell * ell

    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)
        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))

    return cov_elem


def make_scalar_rbf_bins(points, ell, sigma2, n_bins, jitter):
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
    if hasattr(gp, "make_separable_vector_covariance"):
        return gp.make_separable_vector_covariance(covariance_bins_vals, jnp.eye(ncomp))
    from graphgp.refine import make_separable_vector_covariance
    return make_separable_vector_covariance(covariance_bins_vals, jnp.eye(ncomp))


def block_until_ready(x):
    return jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


def time_fn(fn, *args, repeats=5):
    # compile+first run
    t0 = time.perf_counter(); out = fn(*args); block_until_ready(out); t1 = time.perf_counter()
    t_first = t1 - t0

    # steady-state
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(*args); block_until_ready(out); t1 = time.perf_counter()
        ts.append(t1 - t0)
    return t_first, float(np.mean(ts)), float(np.std(ts, ddof=1) if repeats > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n0", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ell", type=float, default=0.075)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--n-bins", type=int, default=2048)
    ap.add_argument("--scalar-jitter", type=float, default=1e-6)
    ap.add_argument("--periodic-vector", action="store_true")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    try:
        print("JAX devices:", jax.devices())
    except Exception:
        pass

    n = args.n
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))

    graph = gp.build_graph(points, n0=args.n0, k=args.k)
    N = len(graph.points)
    n0 = len(graph.points) - len(graph.neighbors)

    cov_scalar = make_scalar_rbf_bins(graph.points, ell=args.ell, sigma2=args.sigma2, n_bins=args.n_bins, jitter=args.scalar_jitter)
    cov_vec_sep = make_separable_vector_covariance_from_scalar_bins(cov_scalar, ncomp=3)
    cov_vec_div = make_div_free_rbf_covariance(args.ell, sigma2=args.sigma2, periodic=args.periodic_vector)

    key = jr.PRNGKey(0)
    key, k1, k2 = jr.split(key, 3)
    xi_s0 = jr.normal(k1, (n0,))
    xi_sR = jr.normal(k1, (N - n0,))

    xi_v0 = jr.normal(k2, (n0, 3))
    xi_vR = jr.normal(k2, (N - n0, 3))

    # Scalar dense + refine
    dense_s = jax.jit(lambda xi0: gp.generate_dense(graph.points[:n0], cov_scalar, xi0))
    refine_s = jax.jit(lambda init, xiR: gp.refine(graph.points, graph.neighbors, graph.offsets, cov_scalar, init, xiR, cuda=False, fast_jit=True))

    # Vector dense + refine
    dense_v_sep = jax.jit(lambda xi0: gp.generate_dense_vector(graph.points[:n0], cov_vec_sep, xi0))
    refine_v_sep = jax.jit(lambda init, xiR: gp.refine_vector(graph.points, graph.neighbors, graph.offsets, cov_vec_sep, init, xiR, cuda=False, fast_jit=True))

    dense_v_div = jax.jit(lambda xi0: gp.generate_dense_vector(graph.points[:n0], cov_vec_div, xi0))
    refine_v_div = jax.jit(lambda init, xiR: gp.refine_vector(graph.points, graph.neighbors, graph.offsets, cov_vec_div, init, xiR, cuda=False, fast_jit=True))

    t_dense_s_first, t_dense_s, t_dense_s_std = time_fn(dense_s, xi_s0, repeats=args.repeats)
    init_s = dense_s(xi_s0); block_until_ready(init_s)
    t_ref_s_first, t_ref_s, t_ref_s_std = time_fn(refine_s, init_s, xi_sR, repeats=args.repeats)

    t_dense_vs_first, t_dense_vs, t_dense_vs_std = time_fn(dense_v_sep, xi_v0, repeats=args.repeats)
    init_vs = dense_v_sep(xi_v0); block_until_ready(init_vs)
    t_ref_vs_first, t_ref_vs, t_ref_vs_std = time_fn(refine_v_sep, init_vs, xi_vR, repeats=args.repeats)

    t_dense_vd_first, t_dense_vd, t_dense_vd_std = time_fn(dense_v_div, xi_v0, repeats=args.repeats)
    init_vd = dense_v_div(xi_v0); block_until_ready(init_vd)
    t_ref_vd_first, t_ref_vd, t_ref_vd_std = time_fn(refine_v_div, init_vd, xi_vR, repeats=args.repeats)

    outdir = Path("bench_out")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"mode": "scalar", "N": N, "n0": n0, "k": args.k, "dense_s": t_dense_s, "refine_s": t_ref_s, "dense_std": t_dense_s_std, "refine_std": t_ref_s_std},
        {"mode": "vec_separable", "N": N, "n0": n0, "k": args.k, "dense_s": t_dense_vs, "refine_s": t_ref_vs, "dense_std": t_dense_vs_std, "refine_std": t_ref_vs_std},
        {"mode": "vec_divfree", "N": N, "n0": n0, "k": args.k, "dense_s": t_dense_vd, "refine_s": t_ref_vd, "dense_std": t_dense_vd_std, "refine_std": t_ref_vd_std},
    ]

    csv_path = outdir / "breakdown.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Plot stacked bars
    labels = [r["mode"] for r in rows]
    dense = [r["dense_s"] for r in rows]
    ref = [r["refine_s"] for r in rows]

    x = np.arange(len(labels))
    plt.figure(figsize=(7, 4))
    plt.bar(x, dense, label="dense")
    plt.bar(x, ref, bottom=dense, label="refine")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("steady-state time [s]")
    plt.title(f"GraphGP breakdown (N={N}, n0={n0}, k={args.k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "breakdown_stacked.png", dpi=180)
    plt.close()

    print("\nBreakdown (steady-state mean ± std):")
    for r in rows:
        print(f"{r['mode']}: dense {r['dense_s']:.4f}±{r['dense_std']:.4f}  refine {r['refine_s']:.4f}±{r['refine_std']:.4f}  total {(r['dense_s']+r['refine_s']):.4f}")

    print("\nWrote:")
    print(" -", csv_path)
    print(" -", outdir / "breakdown_stacked.png")


if __name__ == "__main__":
    main()
