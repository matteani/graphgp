#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import jax.random as jr

from rm_common import (
    require_graphgp,
    make_graphgp_covariance,
    make_cartesian_grid_points,
    sample_tomographic_cartesian_pulsars,
    build_cartesian_rm_response,
    build_cartesian_endpoint_response,
    response_apply_numpy,
    divergence_fd_cartesian,
    save_npz,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a regular Cartesian-grid pulsar RM mock from a GraphGP vector field.")
    ap.add_argument("--out", type=Path, default=Path("outputs/mock_regular_cartesian.npz"))
    ap.add_argument("--truth-prior-kind", type=str, default="divfree_rbf", choices=["divfree_rbf", "standard_vector_rbf", "divfree", "standard", "nondivfree"])
    ap.add_argument("--n", type=int, default=48, help="Cartesian grid side length; Nvox=n^3")
    ap.add_argument("--n-directions", type=int, default=384)
    ap.add_argument("--n-shells", type=int, default=5)
    ap.add_argument("--r-min", type=float, default=0.08)
    ap.add_argument("--r-max", type=float, default=0.46)
    ap.add_argument("--n-ray-steps", type=int, default=224)
    ap.add_argument("--ell", type=float, default=0.06)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--graph-n0", type=int, default=1024)
    ap.add_argument("--graph-k", type=int, default=24)
    ap.add_argument("--rm-noise-frac", type=float, default=0.055)
    ap.add_argument("--rm-noise-floor", type=float, default=0.012)
    ap.add_argument("--c-rm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gp = require_graphgp()
    rng = np.random.default_rng(args.seed)
    key = jr.PRNGKey(args.seed)

    coords, points_np = make_cartesian_grid_points(args.n)
    points = jnp.asarray(points_np, dtype=jnp.float64)
    cov_elem, truth_prior_kind = make_graphgp_covariance(
        gp, args.truth_prior_kind, ell=args.ell, sigma2=args.sigma2, periodic=args.periodic
    )

    print(f"[regular] Building GraphGP graph: n={args.n}, Nvox={points.shape[0]}, n0={args.graph_n0}, k={args.graph_k}", flush=True)
    graph = gp.build_graph(points, n0=args.graph_n0, k=args.graph_k)

    print(f"[regular] Sampling truth with prior={truth_prior_kind}", flush=True)
    xi_true = jr.normal(key, (points.shape[0], 3))
    B = gp.generate_vector(graph, cov_elem, xi_true, cuda=False, fast_jit=True)
    B_true = np.asarray(B).reshape((args.n, args.n, args.n, 3))
    B_flat = B_true.reshape(-1)

    observer = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    pulsar_pos, pulsar_dir, pulsar_dist, direction_id, shell_id, shell_distances = sample_tomographic_cartesian_pulsars(
        rng,
        observer=observer,
        n_directions=args.n_directions,
        n_shells=args.n_shells,
        r_min=args.r_min,
        r_max=args.r_max,
    )
    n_pulsars = len(pulsar_dist)
    print(f"[regular] Pulsars: {args.n_directions} directions × {args.n_shells} shells = {n_pulsars}", flush=True)

    print("[regular] Building RM response", flush=True)
    rows, cols, vals = build_cartesian_rm_response(
        args.n,
        observer,
        pulsar_pos,
        pulsar_dir,
        pulsar_dist,
        n_steps=args.n_ray_steps,
        c_rm=args.c_rm,
    )
    endpoint_rows, endpoint_cols, endpoint_vals = build_cartesian_endpoint_response(args.n, pulsar_pos, pulsar_dir)

    rm_true = response_apply_numpy(rows, cols, vals, B_flat, n_pulsars)
    bpar_true = response_apply_numpy(endpoint_rows, endpoint_cols, endpoint_vals, B_flat, n_pulsars)

    noise_sigma = args.rm_noise_floor + args.rm_noise_frac * float(np.std(rm_true))
    rm_noise = np.full(n_pulsars, noise_sigma, dtype=np.float64)
    rm_obs = rm_true + rng.normal(scale=rm_noise)

    B_rms = float(np.sqrt(np.mean(np.sum(B_true**2, axis=-1))))
    div = divergence_fd_cartesian(B_true)
    div_rms = float(np.sqrt(np.mean(div**2)))
    eps_rms = float(args.ell * div_rms / (B_rms + 1e-12))

    print(f"[regular] RM std={np.std(rm_true):.4g}, noise sigma={noise_sigma:.4g}", flush=True)
    print(f"[regular] B_rms={B_rms:.4g}, ell*div_rms/B_rms={eps_rms:.4g}", flush=True)

    save_npz(
        args.out,
        geometry=np.asarray("regular_cartesian"),
        truth_prior_kind=np.asarray(truth_prior_kind),
        reconstruction_prior_kind=np.asarray("unset"),
        n=np.asarray(args.n),
        points=points_np,
        coords=coords,
        observer=observer,
        pulsar_pos=pulsar_pos,
        pulsar_dir=pulsar_dir,
        pulsar_dist=pulsar_dist,
        direction_id=direction_id,
        shell_id=shell_id,
        shell_distances=shell_distances,
        n_directions=np.asarray(args.n_directions),
        n_shells=np.asarray(args.n_shells),
        response_rows=rows,
        response_cols=cols,
        response_vals=vals,
        endpoint_rows=endpoint_rows,
        endpoint_cols=endpoint_cols,
        endpoint_vals=endpoint_vals,
        rm_true=rm_true,
        rm_obs=rm_obs,
        rm_noise=rm_noise,
        bpar_true=bpar_true,
        B_true=B_true,
        B_true_rms=np.asarray(B_rms),
        div_true_rms=np.asarray(div_rms),
        epsilon_true_rms=np.asarray(eps_rms),
        graph_n0=np.asarray(args.graph_n0),
        graph_k=np.asarray(args.graph_k),
        ell=np.asarray(args.ell),
        sigma2=np.asarray(args.sigma2),
        periodic=np.asarray(args.periodic),
        n_ray_steps=np.asarray(args.n_ray_steps),
        rm_noise_frac=np.asarray(args.rm_noise_frac),
        rm_noise_floor=np.asarray(args.rm_noise_floor),
        c_rm=np.asarray(args.c_rm),
        seed=np.asarray(args.seed),
    )
    print(f"[regular] Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
