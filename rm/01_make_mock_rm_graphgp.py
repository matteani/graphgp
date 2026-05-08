#!/usr/bin/env python3
"""Make mock pulsar rotation-measure data from a GraphGP magnetic-field prior.

This file uses GraphGP only. It creates a synthetic truth B_true, generates
mock pulsar RMs via line integrals, and writes an .npz file consumed by
02_reconstruct_rm_nifty_graphgp.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from rm_graphgp_nifty_utils import (
    require_graphgp,
    make_grid_points,
    electron_density,
    sample_pulsars,
    build_rm_response_coo,
    response_apply_numpy,
    endpoint_los_component,
    divergence_fd,
    save_npz,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("rm_graphgp_mock.npz"))
    ap.add_argument("--n", type=int, default=30, help="grid side length; n=30 gives 30^3 cells")
    ap.add_argument("--n-pulsars", type=int, default=192)
    ap.add_argument("--n-ray-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ell", type=float, default=0.085)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--graph-n0", type=int, default=512)
    ap.add_argument("--graph-k", type=int, default=18)
    ap.add_argument("--rm-noise-frac", type=float, default=0.08)
    ap.add_argument("--rm-noise-floor", type=float, default=0.025)
    ap.add_argument("--c-rm", type=float, default=1.0)
    args = ap.parse_args()

    gp = require_graphgp()
    rng_np = np.random.default_rng(args.seed)
    key = jr.PRNGKey(args.seed)

    coords, points_np = make_grid_points(args.n)
    points = jnp.asarray(points_np)

    print(f"Building GraphGP graph for n={args.n}, N={points.shape[0]}, n0={args.graph_n0}, k={args.graph_k}")
    graph = gp.build_graph(points, n0=args.graph_n0, k=args.graph_k)
    cov_elem = gp.extras.make_div_free_rbf_covariance(
        ell=args.ell, sigma2=args.sigma2, periodic=args.periodic
    )

    print("Sampling GraphGP divergence-free vector-field truth...")
    xi_true = jr.normal(key, (points.shape[0], 3))
    B = gp.generate_vector(graph, cov_elem, xi_true, cuda=False, fast_jit=True)
    B_true = np.asarray(B).reshape((args.n, args.n, args.n, 3))

    observer = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    pulsar_pos, pulsar_dir, pulsar_dist = sample_pulsars(
        rng_np, args.n_pulsars, observer, r_min=0.16, r_max=0.48
    )

    print("Building RM response matrix...")
    coo = build_rm_response_coo(
        args.n,
        observer,
        pulsar_pos,
        pulsar_dir,
        pulsar_dist,
        n_steps=args.n_ray_steps,
        c_rm=args.c_rm,
    )

    B_flat = B_true.reshape(-1)
    rm_true = response_apply_numpy(coo, B_flat)
    noise_scale = args.rm_noise_floor + args.rm_noise_frac * np.std(rm_true)
    rm_noise = np.full(args.n_pulsars, noise_scale, dtype=np.float64)
    rm_obs = rm_true + rng_np.normal(scale=rm_noise)
    bpar_true = endpoint_los_component(B_true, pulsar_pos, pulsar_dir)

    ne_grid = electron_density(points_np).reshape((args.n, args.n, args.n))
    div = divergence_fd(B_true)
    B_rms = float(np.sqrt(np.mean(np.sum(B_true**2, axis=-1))))
    div_rms = float(np.sqrt(np.mean(div**2)))
    eps_rms = float(args.ell * div_rms / (B_rms + 1e-12))

    save_npz(
        args.out,
        n=np.asarray(args.n),
        coords=coords,
        points=points_np,
        B_true=B_true,
        xi_true=np.asarray(xi_true),
        ne_grid=ne_grid,
        observer=observer,
        pulsar_pos=pulsar_pos,
        pulsar_dir=pulsar_dir,
        pulsar_dist=pulsar_dist,
        rm_true=rm_true,
        rm_obs=rm_obs,
        rm_noise=rm_noise,
        bpar_true=bpar_true,
        response_rows=coo["rows"],
        response_cols=coo["cols"],
        response_data=coo["data"],
        response_shape=coo["shape"],
        ell=np.asarray(args.ell),
        sigma2=np.asarray(args.sigma2),
        periodic=np.asarray(args.periodic),
        graph_n0=np.asarray(args.graph_n0),
        graph_k=np.asarray(args.graph_k),
        n_ray_steps=np.asarray(args.n_ray_steps),
        c_rm=np.asarray(args.c_rm),
        B_rms=np.asarray(B_rms),
        div_rms=np.asarray(div_rms),
        epsilon_rms=np.asarray(eps_rms),
    )

    print(f"Saved mock data: {args.out}")
    print(f"RM std={np.std(rm_true):.4g}, noise sigma={noise_scale:.4g}")
    print(f"B_rms={B_rms:.4g}, div_rms={div_rms:.4g}, ell*div_rms/B_rms={eps_rms:.4g}")


if __name__ == "__main__":
    main()
