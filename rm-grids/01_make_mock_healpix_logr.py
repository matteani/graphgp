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
    build_healpix_logr_points,
    make_healpix_pulsars,
    build_healpix_logr_rm_response,
    build_healpix_logr_endpoint_response,
    response_apply_numpy,
    save_npz,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a HEALPix × log-r pulsar RM mock from a GraphGP vector field.")
    ap.add_argument("--out", type=Path, default=Path("outputs/mock_healpix_logr.npz"))
    ap.add_argument("--truth-prior-kind", type=str, default="divfree_rbf", choices=["divfree_rbf", "standard_vector_rbf", "divfree", "standard", "nondivfree"])
    ap.add_argument("--nside", type=int, default=8, help="HEALPix nside; npix=12*nside^2")
    ap.add_argument("--n-radial", type=int, default=32, help="number of logarithmic radial shells")
    ap.add_argument("--r-min", type=float, default=0.035)
    ap.add_argument("--r-max", type=float, default=1.0)
    ap.add_argument("--n-pulsar-directions", type=int, default=0, help="0 means all HEALPix pixels")
    ap.add_argument("--n-pulsar-shells", type=int, default=6)
    ap.add_argument("--n-ray-steps", type=int, default=256)
    ap.add_argument("--ell", type=float, default=0.12)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--graph-n0", type=int, default=1024)
    ap.add_argument("--graph-k", type=int, default=24)
    ap.add_argument("--rm-noise-frac", type=float, default=0.055)
    ap.add_argument("--rm-noise-floor", type=float, default=0.010)
    ap.add_argument("--c-rm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    gp = require_graphgp()
    rng = np.random.default_rng(args.seed)
    key = jr.PRNGKey(args.seed)

    points_np, radii, theta, phi, unit, shell_ids_grid, pixel_ids_grid = build_healpix_logr_points(
        args.nside, args.n_radial, args.r_min, args.r_max
    )
    npix = unit.shape[0]
    N = points_np.shape[0]
    points = jnp.asarray(points_np, dtype=jnp.float64)

    cov_elem, truth_prior_kind = make_graphgp_covariance(
        gp, args.truth_prior_kind, ell=args.ell, sigma2=args.sigma2, periodic=False
    )

    print(
        f"[healpix-logr] Building GraphGP graph: nside={args.nside}, npix={npix}, "
        f"n_radial={args.n_radial}, N={N}, n0={args.graph_n0}, k={args.graph_k}",
        flush=True,
    )
    graph = gp.build_graph(points, n0=args.graph_n0, k=args.graph_k)

    print(f"[healpix-logr] Sampling truth with prior={truth_prior_kind}", flush=True)
    xi_true = jr.normal(key, (N, 3))
    B = gp.generate_vector(graph, cov_elem, xi_true, cuda=False, fast_jit=True)
    B_true = np.asarray(B).reshape((args.n_radial, npix, 3))
    B_flat = B_true.reshape(-1)

    n_dir = args.n_pulsar_directions if args.n_pulsar_directions > 0 else npix
    pulsar_pos, pulsar_dir, pulsar_dist, pulsar_pixel_id, pulsar_shell_id, pulsar_shell_distances = make_healpix_pulsars(
        radii=radii,
        unit=unit,
        n_pulsar_directions=n_dir,
        n_pulsar_shells=args.n_pulsar_shells,
        rng=rng,
    )
    n_pulsars = len(pulsar_dist)
    print(
        f"[healpix-logr] Pulsars: {len(np.unique(pulsar_pixel_id))} angular pixels × "
        f"{len(np.unique(pulsar_shell_id))} radial shells = {n_pulsars}",
        flush=True,
    )

    print("[healpix-logr] Building radial RM response", flush=True)
    rows, cols, vals = build_healpix_logr_rm_response(
        radii=radii,
        unit=unit,
        pixel_ids=pulsar_pixel_id,
        pulsar_dist=pulsar_dist,
        n_steps=args.n_ray_steps,
        c_rm=args.c_rm,
    )
    endpoint_rows, endpoint_cols, endpoint_vals = build_healpix_logr_endpoint_response(
        radii=radii,
        unit=unit,
        pixel_ids=pulsar_pixel_id,
        pulsar_dist=pulsar_dist,
    )

    rm_true = response_apply_numpy(rows, cols, vals, B_flat, n_pulsars)
    bpar_true = response_apply_numpy(endpoint_rows, endpoint_cols, endpoint_vals, B_flat, n_pulsars)

    noise_sigma = args.rm_noise_floor + args.rm_noise_frac * float(np.std(rm_true))
    rm_noise = np.full(n_pulsars, noise_sigma, dtype=np.float64)
    rm_obs = rm_true + rng.normal(scale=rm_noise)
    B_rms = float(np.sqrt(np.mean(np.sum(B_true**2, axis=-1))))

    print(f"[healpix-logr] RM std={np.std(rm_true):.4g}, noise sigma={noise_sigma:.4g}", flush=True)
    print(f"[healpix-logr] B_rms={B_rms:.4g}", flush=True)

    save_npz(
        args.out,
        geometry=np.asarray("healpix_logr"),
        truth_prior_kind=np.asarray(truth_prior_kind),
        reconstruction_prior_kind=np.asarray("unset"),
        n=np.asarray(-1),
        nside=np.asarray(args.nside),
        npix=np.asarray(npix),
        n_radial=np.asarray(args.n_radial),
        points=points_np,
        radii=radii,
        theta_pix=theta,
        phi_pix=phi,
        unit_pix=unit,
        grid_shell_id=shell_ids_grid,
        grid_pixel_id=pixel_ids_grid,
        pulsar_pos=pulsar_pos,
        pulsar_dir=pulsar_dir,
        pulsar_dist=pulsar_dist,
        pulsar_pixel_id=pulsar_pixel_id,
        pulsar_shell_id=pulsar_shell_id,
        pulsar_shell_distances=pulsar_shell_distances,
        n_directions=np.asarray(len(np.unique(pulsar_pixel_id))),
        n_shells=np.asarray(len(np.unique(pulsar_shell_id))),
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
        graph_n0=np.asarray(args.graph_n0),
        graph_k=np.asarray(args.graph_k),
        ell=np.asarray(args.ell),
        sigma2=np.asarray(args.sigma2),
        periodic=np.asarray(False),
        n_ray_steps=np.asarray(args.n_ray_steps),
        rm_noise_frac=np.asarray(args.rm_noise_frac),
        rm_noise_floor=np.asarray(args.rm_noise_floor),
        c_rm=np.asarray(args.c_rm),
        seed=np.asarray(args.seed),
    )
    print(f"[healpix-logr] Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
