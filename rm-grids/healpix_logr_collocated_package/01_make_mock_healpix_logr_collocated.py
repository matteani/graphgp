#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

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
    electron_density,
    response_apply_numpy,
    save_npz,
)


def radial_shell_edges_from_centres(radii: np.ndarray) -> np.ndarray:
    """Return shell boundaries for log-spaced radial shell centres.

    The first boundary is radii[0] and the last boundary is radii[-1], matching
    the previous HEALPix/log-r response convention which starts the RM integral
    at r_min rather than at the singular origin.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("Need at least two radial shell centres.")
    lr = np.log(radii)
    ledges = np.empty(radii.size + 1, dtype=np.float64)
    ledges[0] = lr[0]
    ledges[-1] = lr[-1]
    ledges[1:-1] = 0.5 * (lr[:-1] + lr[1:])
    edges = np.exp(ledges)
    edges[0] = radii[0]
    edges[-1] = radii[-1]
    return edges


def build_healpix_logr_shell_response(
    *,
    radii: np.ndarray,
    unit: np.ndarray,
    pixel_ids: np.ndarray,
    shell_ids: np.ndarray,
    c_rm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build RM response using only existing HEALPix/log-r grid points.

    Each line of sight is exactly a radial HEALPix column.  The field value used
    for radial shell q is B(r_q, pixel), so no spatial/radial interpolation is
    done.  Shell-centred quadrature weights are the overlap of the radial cell
    [edge_q, edge_{q+1}] with [r_min, r_pulsar].
    """
    radii = np.asarray(radii, dtype=np.float64)
    unit = np.asarray(unit, dtype=np.float64)
    pixel_ids = np.asarray(pixel_ids, dtype=np.int64)
    shell_ids = np.asarray(shell_ids, dtype=np.int64)
    npix = int(unit.shape[0])
    edges = radial_shell_edges_from_centres(radii)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for p, (pix, stop_shell) in enumerate(zip(pixel_ids, shell_ids)):
        pix = int(pix)
        stop_shell = int(stop_shell)
        nhat = unit[pix]
        dist = float(radii[stop_shell])
        for shell in range(stop_shell + 1):
            ds = max(0.0, min(float(edges[shell + 1]), dist) - float(edges[shell]))
            if ds <= 0.0:
                continue
            rr = float(radii[shell])
            x = rr * nhat
            ne = float(electron_density(x[None, :])[0])
            pref = float(c_rm) * ne * ds
            v = shell * npix + pix
            base = 3 * v
            for c in range(3):
                rows.append(p)
                cols.append(base + c)
                vals.append(pref * float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def build_healpix_logr_shell_endpoint_response(
    *,
    unit: np.ndarray,
    pixel_ids: np.ndarray,
    shell_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Endpoint B_parallel response with pulsars exactly on grid nodes."""
    unit = np.asarray(unit, dtype=np.float64)
    pixel_ids = np.asarray(pixel_ids, dtype=np.int64)
    shell_ids = np.asarray(shell_ids, dtype=np.int64)
    npix = int(unit.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for p, (pix, shell) in enumerate(zip(pixel_ids, shell_ids)):
        pix = int(pix)
        shell = int(shell)
        nhat = unit[pix]
        v = shell * npix + pix
        base = 3 * v
        for c in range(3):
            rows.append(p)
            cols.append(base + c)
            vals.append(float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make a HEALPix x log-r mock using shell-collocated RM lines of sight, with no interpolation in the RM response."
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/mock_healpix_logr_collocated.npz"))
    ap.add_argument("--truth-prior-kind", type=str, default="divfree_rbf", choices=["divfree_rbf", "standard_vector_rbf", "divfree", "standard", "nondivfree"])
    ap.add_argument("--nside", type=int, default=8, help="HEALPix nside; npix=12*nside^2")
    ap.add_argument("--n-radial", type=int, default=32, help="number of logarithmic radial shell centres")
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--r-max", type=float, default=0.90)
    ap.add_argument("--n-pulsar-directions", type=int, default=384, help="0 means all HEALPix pixels")
    ap.add_argument("--n-pulsar-shells", type=int, default=6)
    ap.add_argument("--ell", type=float, default=0.20)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--graph-n0", type=int, default=256)
    ap.add_argument("--graph-k", type=int, default=14)
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
    npix = int(unit.shape[0])
    N = int(points_np.shape[0])
    points = jnp.asarray(points_np, dtype=jnp.float64)

    cov_elem, truth_prior_kind = make_graphgp_covariance(
        gp, args.truth_prior_kind, ell=args.ell, sigma2=args.sigma2, periodic=False
    )

    print(
        f"[healpix-logr-collocated] Building GraphGP graph: nside={args.nside}, npix={npix}, "
        f"n_radial={args.n_radial}, N={N}, n0={args.graph_n0}, k={args.graph_k}",
        flush=True,
    )
    graph = gp.build_graph(points, n0=args.graph_n0, k=args.graph_k)

    print(f"[healpix-logr-collocated] Sampling truth with prior={truth_prior_kind}", flush=True)
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
    n_pulsars = int(len(pulsar_dist))
    print(
        f"[healpix-logr-collocated] Pulsars: {len(np.unique(pulsar_pixel_id))} angular pixels x "
        f"{len(np.unique(pulsar_shell_id))} radial shells = {n_pulsars}",
        flush=True,
    )

    print("[healpix-logr-collocated] Building shell-collocated RM response", flush=True)
    rows, cols, vals = build_healpix_logr_shell_response(
        radii=radii,
        unit=unit,
        pixel_ids=pulsar_pixel_id,
        shell_ids=pulsar_shell_id,
        c_rm=args.c_rm,
    )
    endpoint_rows, endpoint_cols, endpoint_vals = build_healpix_logr_shell_endpoint_response(
        unit=unit,
        pixel_ids=pulsar_pixel_id,
        shell_ids=pulsar_shell_id,
    )

    rm_true = response_apply_numpy(rows, cols, vals, B_flat, n_pulsars)
    bpar_true = response_apply_numpy(endpoint_rows, endpoint_cols, endpoint_vals, B_flat, n_pulsars)

    noise_sigma = args.rm_noise_floor + args.rm_noise_frac * float(np.std(rm_true))
    rm_noise = np.full(n_pulsars, noise_sigma, dtype=np.float64)
    rm_obs = rm_true + rng.normal(scale=rm_noise)
    B_rms = float(np.sqrt(np.mean(np.sum(B_true**2, axis=-1))))

    print(f"[healpix-logr-collocated] RM std={np.std(rm_true):.4g}, noise sigma={noise_sigma:.4g}", flush=True)
    print(f"[healpix-logr-collocated] B_rms={B_rms:.4g}", flush=True)

    save_npz(
        args.out,
        geometry=np.asarray("healpix_logr"),
        grid_variant=np.asarray("shell_collocated_no_interpolation"),
        truth_prior_kind=np.asarray(truth_prior_kind),
        reconstruction_prior_kind=np.asarray("unset"),
        n=np.asarray(-1),
        nside=np.asarray(args.nside),
        npix=np.asarray(npix),
        n_radial=np.asarray(args.n_radial),
        points=points_np,
        radii=radii,
        radial_edges=radial_shell_edges_from_centres(radii),
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
        n_ray_steps=np.asarray(0),
        rm_response_kind=np.asarray("shell_collocated_no_interpolation"),
        rm_noise_frac=np.asarray(args.rm_noise_frac),
        rm_noise_floor=np.asarray(args.rm_noise_floor),
        c_rm=np.asarray(args.c_rm),
        seed=np.asarray(args.seed),
    )
    print(f"[healpix-logr-collocated] Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
