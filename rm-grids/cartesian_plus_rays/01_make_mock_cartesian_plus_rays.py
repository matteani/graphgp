#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

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
    electron_density,
    response_apply_numpy,
    divergence_fd_cartesian,
    save_npz,
)


def _round_key(x: np.ndarray, decimals: int = 13) -> tuple:
    return tuple(np.round(np.asarray(x, dtype=np.float64), decimals=decimals))


class PointAccumulator:
    """Unique point store with stable integer ids.

    Points are deduplicated by rounded coordinates.  This is important because an exact
    duplicate coordinate would make local covariance blocks singular.
    """

    def __init__(self, *, decimals: int = 13):
        self.decimals = int(decimals)
        self.points: list[np.ndarray] = []
        self.lookup: Dict[tuple, int] = {}
        self.is_base: list[bool] = []
        self.is_ray: list[bool] = []
        self.is_endpoint: list[bool] = []

    def add(self, x: np.ndarray, *, base: bool = False, ray: bool = False, endpoint: bool = False) -> int:
        key = _round_key(x, self.decimals)
        if key in self.lookup:
            i = self.lookup[key]
            self.is_base[i] = self.is_base[i] or bool(base)
            self.is_ray[i] = self.is_ray[i] or bool(ray)
            self.is_endpoint[i] = self.is_endpoint[i] or bool(endpoint)
            return i
        i = len(self.points)
        self.lookup[key] = i
        self.points.append(np.asarray(x, dtype=np.float64))
        self.is_base.append(bool(base))
        self.is_ray.append(bool(ray))
        self.is_endpoint.append(bool(endpoint))
        return i

    def arrays(self):
        return (
            np.asarray(self.points, dtype=np.float64),
            np.asarray(self.is_base, dtype=bool),
            np.asarray(self.is_ray, dtype=bool),
            np.asarray(self.is_endpoint, dtype=bool),
        )


def build_cartesian_plus_rays_pointcloud(
    *,
    n: int,
    observer: np.ndarray,
    pulsar_pos: np.ndarray,
    pulsar_dir: np.ndarray,
    pulsar_dist: np.ndarray,
    n_ray_steps: int,
    ray_start: float,
    decimals: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return regular grid plus exact ray quadrature and endpoint points.

    The RM integral uses a fixed radial quadrature grid for each direction.  This
    keeps the number of added points roughly n_directions*n_ray_steps instead of
    n_pulsars*n_ray_steps.  For a pulsar at distance d, each global radial bin
    contributes only the overlap of that bin with [0,d].
    """
    coords, base_points = make_cartesian_grid_points(int(n))
    acc = PointAccumulator(decimals=decimals)
    base_indices = np.empty(len(base_points), dtype=np.int64)
    for i, x in enumerate(base_points):
        base_indices[i] = acc.add(x, base=True)

    # One shared observer point handles the small inner segment of every ray.
    # This avoids a nearly singular dense sphere of many direction-dependent
    # quadrature points extremely close to the observer.
    observer_index = acc.add(observer, ray=True)

    # Build a stable direction id from repeated directions.  The pulsar catalogue
    # already has repeated directions across shells, but this keeps the function
    # self-contained.
    dir_keys: Dict[tuple, int] = {}
    unique_dirs: list[np.ndarray] = []
    pulsar_direction_index = np.empty(len(pulsar_dir), dtype=np.int64)
    for p, nhat in enumerate(pulsar_dir):
        key = _round_key(nhat, decimals=12)
        if key not in dir_keys:
            dir_keys[key] = len(unique_dirs)
            unique_dirs.append(np.asarray(nhat, dtype=np.float64))
        pulsar_direction_index[p] = dir_keys[key]

    max_dist = float(np.max(pulsar_dist))
    ray_start = float(ray_start)
    if not (0.0 <= ray_start < max_dist):
        raise ValueError(f"Need 0 <= ray_start < max pulsar distance; got ray_start={ray_start}, max_dist={max_dist}")
    edges = np.linspace(ray_start, max_dist, int(n_ray_steps) + 1, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # ray_point_indices[direction, radial-bin] gives the model point id.
    ray_point_indices = np.empty((len(unique_dirs), int(n_ray_steps)), dtype=np.int64)
    for idir, nhat in enumerate(unique_dirs):
        pts = observer[None, :] + centres[:, None] * nhat[None, :]
        for q, x in enumerate(pts):
            ray_point_indices[idir, q] = acc.add(x, ray=True)

    # endpoint_indices[p] gives the exact point at the pulsar location.
    endpoint_indices = np.empty(len(pulsar_pos), dtype=np.int64)
    for p, x in enumerate(pulsar_pos):
        endpoint_indices[p] = acc.add(x, endpoint=True)

    points, is_base, is_ray, is_endpoint = acc.arrays()
    return (
        points,
        coords,
        base_indices,
        is_base,
        is_ray,
        is_endpoint,
        ray_point_indices,
        endpoint_indices,
        observer_index,
    )


def build_direct_cartesian_plus_rays_rm_response(
    *,
    observer: np.ndarray,
    pulsar_dir: np.ndarray,
    pulsar_dist: np.ndarray,
    pulsar_direction_index: np.ndarray,
    ray_point_indices: np.ndarray,
    r_edges: np.ndarray,
    observer_index: int,
    c_rm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build RM response with no spatial interpolation.

    Each quadrature node is an actual GraphGP model point, so the response only
    needs to select that point and project B onto the line of sight.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    ray_start = float(r_edges[0])
    ne_obs = float(electron_density(observer[None, :])[0])
    for p, (nhat, dist, idir) in enumerate(zip(pulsar_dir, pulsar_dist, pulsar_direction_index)):
        # Inner segment [0, ray_start] represented by the shared observer point.
        ds0 = min(ray_start, float(dist))
        if ds0 > 0.0:
            pref0 = float(c_rm) * ne_obs * ds0
            base0 = 3 * int(observer_index)
            for c in range(3):
                rows.append(p)
                cols.append(base0 + c)
                vals.append(pref0 * float(nhat[c]))

        for q in range(len(centres)):
            # Bin overlap length with [0, dist].
            ds = max(0.0, min(float(r_edges[q + 1]), float(dist)) - float(r_edges[q]))
            if ds <= 0.0:
                continue
            x = observer + centres[q] * nhat
            ne = float(electron_density(x[None, :])[0])
            pref = float(c_rm) * ne * ds
            v = int(ray_point_indices[int(idir), q])
            base = 3 * v
            for c in range(3):
                rows.append(p)
                cols.append(base + c)
                vals.append(pref * float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def build_direct_endpoint_response(endpoint_indices: np.ndarray, pulsar_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for p, (idx, nhat) in enumerate(zip(endpoint_indices, pulsar_dir)):
        base = 3 * int(idx)
        for c in range(3):
            rows.append(p)
            cols.append(base + c)
            vals.append(float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make a Cartesian-grid-plus-ray-points pulsar RM mock from a GraphGP vector field."
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/mock_cartesian_plus_rays.npz"))
    ap.add_argument("--truth-prior-kind", type=str, default="divfree_rbf", choices=["divfree_rbf", "standard_vector_rbf", "divfree", "standard", "nondivfree"])
    ap.add_argument("--n", type=int, default=30, help="Cartesian base-grid side length; base grid has n^3 points")
    ap.add_argument("--n-directions", type=int, default=384)
    ap.add_argument("--n-shells", type=int, default=6)
    ap.add_argument("--r-min", type=float, default=0.08)
    ap.add_argument("--r-max", type=float, default=0.46)
    ap.add_argument("--n-ray-steps", type=int, default=160, help="shared radial quadrature nodes per direction")
    ap.add_argument("--ray-start", type=float, default=0.02, help="inner segment [0, ray_start] is represented by one shared observer point")
    ap.add_argument("--ell", type=float, default=0.09)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--graph-n0", type=int, default=256)
    ap.add_argument("--graph-k", type=int, default=14)
    ap.add_argument("--rm-noise-frac", type=float, default=0.055)
    ap.add_argument("--rm-noise-floor", type=float, default=0.012)
    ap.add_argument("--c-rm", type=float, default=1.0)
    ap.add_argument("--dedup-decimals", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gp = require_graphgp()
    rng = np.random.default_rng(args.seed)
    key = jr.PRNGKey(args.seed)

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

    (
        points_np,
        coords,
        base_grid_indices,
        point_is_base_grid,
        point_is_ray_node,
        point_is_endpoint,
        ray_point_indices,
        endpoint_indices,
        observer_index,
    ) = build_cartesian_plus_rays_pointcloud(
        n=args.n,
        observer=observer,
        pulsar_pos=pulsar_pos,
        pulsar_dir=pulsar_dir,
        pulsar_dist=pulsar_dist,
        n_ray_steps=args.n_ray_steps,
        ray_start=args.ray_start,
        decimals=args.dedup_decimals,
    )

    # Reconstruct direction ids consistently with the point-cloud builder.
    dir_keys: Dict[tuple, int] = {}
    pulsar_direction_index = np.empty(len(pulsar_dir), dtype=np.int64)
    for p, nhat in enumerate(pulsar_dir):
        key_tuple = _round_key(nhat, decimals=12)
        if key_tuple not in dir_keys:
            dir_keys[key_tuple] = len(dir_keys)
        pulsar_direction_index[p] = dir_keys[key_tuple]

    r_edges = np.linspace(float(args.ray_start), float(np.max(pulsar_dist)), args.n_ray_steps + 1, dtype=np.float64)

    rows, cols, vals = build_direct_cartesian_plus_rays_rm_response(
        observer=observer,
        pulsar_dir=pulsar_dir,
        pulsar_dist=pulsar_dist,
        pulsar_direction_index=pulsar_direction_index,
        ray_point_indices=ray_point_indices,
        r_edges=r_edges,
        observer_index=observer_index,
        c_rm=args.c_rm,
    )
    endpoint_rows, endpoint_cols, endpoint_vals = build_direct_endpoint_response(endpoint_indices, pulsar_dir)

    points = jnp.asarray(points_np, dtype=jnp.float64)
    cov_elem, truth_prior_kind = make_graphgp_covariance(
        gp, args.truth_prior_kind, ell=args.ell, sigma2=args.sigma2, periodic=args.periodic
    )

    if args.graph_n0 < args.graph_k:
        raise ValueError(f"graph-n0 must be >= graph-k, got graph_n0={args.graph_n0}, graph_k={args.graph_k}")
    if args.graph_n0 >= points_np.shape[0]:
        raise ValueError(f"graph-n0={args.graph_n0} must be smaller than Npoints={points_np.shape[0]}")

    print(
        f"[cartesian-plus-rays] base n={args.n}, base={args.n**3}, "
        f"ray nodes={int(point_is_ray_node.sum())}, endpoints={int(point_is_endpoint.sum())}, "
        f"unique total={points_np.shape[0]}, n0={args.graph_n0}, k={args.graph_k}",
        flush=True,
    )
    print(f"[cartesian-plus-rays] Pulsars: {args.n_directions} directions × {args.n_shells} shells = {n_pulsars}", flush=True)
    print(f"[cartesian-plus-rays] Response nonzeros={len(vals)}; no interpolation weights used", flush=True)

    graph = gp.build_graph(points, n0=args.graph_n0, k=args.graph_k)

    print(f"[cartesian-plus-rays] Sampling truth with prior={truth_prior_kind}", flush=True)
    xi_true = jr.normal(key, (points.shape[0], 3))
    B = gp.generate_vector(graph, cov_elem, xi_true, cuda=False, fast_jit=True)
    B_true = np.asarray(B)  # (Npoints, 3)
    B_flat = B_true.reshape(-1)
    B_true_cartesian_base = B_true[base_grid_indices].reshape((args.n, args.n, args.n, 3))

    rm_true = response_apply_numpy(rows, cols, vals, B_flat, n_pulsars)
    bpar_true = response_apply_numpy(endpoint_rows, endpoint_cols, endpoint_vals, B_flat, n_pulsars)

    noise_sigma = args.rm_noise_floor + args.rm_noise_frac * float(np.std(rm_true))
    rm_noise = np.full(n_pulsars, noise_sigma, dtype=np.float64)
    rm_obs = rm_true + rng.normal(scale=rm_noise)

    B_rms = float(np.sqrt(np.mean(np.sum(B_true**2, axis=-1))))
    B_base_rms = float(np.sqrt(np.mean(np.sum(B_true_cartesian_base**2, axis=-1))))
    div_base = divergence_fd_cartesian(B_true_cartesian_base)
    div_base_rms = float(np.sqrt(np.mean(div_base**2)))
    eps_base_rms = float(args.ell * div_base_rms / (B_base_rms + 1e-12))

    print(f"[cartesian-plus-rays] RM std={np.std(rm_true):.4g}, noise sigma={noise_sigma:.4g}", flush=True)
    print(f"[cartesian-plus-rays] B_rms(all)={B_rms:.4g}, base ell*div_rms/B_rms={eps_base_rms:.4g}", flush=True)

    save_npz(
        args.out,
        geometry=np.asarray("cartesian_plus_rays"),
        truth_prior_kind=np.asarray(truth_prior_kind),
        reconstruction_prior_kind=np.asarray("unset"),
        n=np.asarray(args.n),
        points=points_np,
        coords=coords,
        observer=observer,
        base_grid_indices=base_grid_indices,
        point_is_base_grid=point_is_base_grid,
        point_is_ray_node=point_is_ray_node,
        point_is_endpoint=point_is_endpoint,
        ray_point_indices=ray_point_indices,
        endpoint_indices=endpoint_indices,
        observer_index=np.asarray(observer_index),
        ray_start=np.asarray(args.ray_start),
        ray_r_edges=r_edges,
        ray_r_centres=0.5 * (r_edges[:-1] + r_edges[1:]),
        pulsar_pos=pulsar_pos,
        pulsar_dir=pulsar_dir,
        pulsar_dist=pulsar_dist,
        direction_id=direction_id,
        shell_id=shell_id,
        pulsar_direction_index=pulsar_direction_index,
        shell_distances=shell_distances,
        n_directions=np.asarray(args.n_directions),
        n_shells=np.asarray(args.n_shells),
        n_ray_steps=np.asarray(args.n_ray_steps),
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
        B_true_cartesian_base=B_true_cartesian_base,
        B_true_rms=np.asarray(B_rms),
        B_true_base_rms=np.asarray(B_base_rms),
        div_true_base_rms=np.asarray(div_base_rms),
        epsilon_true_base_rms=np.asarray(eps_base_rms),
        graph_n0=np.asarray(args.graph_n0),
        graph_k=np.asarray(args.graph_k),
        ell=np.asarray(args.ell),
        sigma2=np.asarray(args.sigma2),
        periodic=np.asarray(bool(args.periodic)),
        c_rm=np.asarray(args.c_rm),
        rm_noise_frac=np.asarray(args.rm_noise_frac),
        rm_noise_floor=np.asarray(args.rm_noise_floor),
        seed=np.asarray(args.seed),
        notes=np.asarray("Hybrid point cloud: Cartesian base grid plus exact line-of-sight quadrature and endpoint points; the inner segment is represented by one shared observer point; RM response selects model points directly, with no spatial interpolation."),
    )
    print(f"[cartesian-plus-rays] Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
