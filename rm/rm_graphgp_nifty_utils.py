#!/usr/bin/env python3
"""Utilities for GraphGP-prior pulsar-RM mocks and NIFTy.re inference.

This file deliberately assumes that `graphgp` is installed in your Python
environment. It does not vendor or import any local GraphGP source tree.

Flattening convention
---------------------
B_grid has shape (n, n, n, 3).  The response matrix A multiplies
B_flat = B_grid.reshape(-1), so the column for voxel v and component c is
3*v + c, where v = (ix*n + iy)*n + iz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def require_graphgp():
    """Import installed graphgp and fail loudly if unavailable."""
    try:
        import graphgp as gp  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on user's env
        raise ImportError(
            "Could not import installed `graphgp`. Activate the environment where "
            "your vector GraphGP version is installed, then rerun."
        ) from exc
    return gp


def make_grid_points(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D cell centers and flattened 3D cell-center coordinates."""
    coords = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))
    return coords, points


def electron_density(x: np.ndarray) -> np.ndarray:
    """Toy positive thermal electron density n_e(x) on points (..., 3).

    This is intentionally simple: a smooth overdensity near the box centre plus
    a weak sinusoidal modulation. Replace this by your preferred Galactic n_e
    model later.
    """
    p = np.asarray(x)
    xx, yy, zz = p[..., 0], p[..., 1], p[..., 2]
    r2 = (xx - 0.52) ** 2 + (yy - 0.47) ** 2 + (zz - 0.50) ** 2
    ne = 1.0 + 0.55 * np.exp(-r2 / (2.0 * 0.22**2))
    ne += 0.12 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    return np.clip(ne, 0.15, None)


def sample_pulsars(
    rng: np.random.Generator,
    n_pulsars: int,
    observer: np.ndarray,
    *,
    r_min: float = 0.16,
    r_max: float = 0.48,
    boundary_margin: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample mock pulsars in a unit cube with known distances.

    Returns
    -------
    pulsar_pos : (P, 3)
    pulsar_dir : (P, 3), unit vector from observer to pulsar
    pulsar_dist: (P,)
    """
    positions, dirs, dists = [], [], []
    tries = 0
    while len(positions) < n_pulsars:
        tries += 1
        if tries > 500_000:
            raise RuntimeError("Could not sample enough pulsars inside the cube")
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        dist = rng.uniform(r_min, r_max)
        pos = observer + dist * direction
        if np.all(pos > boundary_margin) and np.all(pos < 1.0 - boundary_margin):
            positions.append(pos)
            dirs.append(direction)
            dists.append(dist)
    return np.asarray(positions), np.asarray(dirs), np.asarray(dists)


def trilinear_weights(point: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return flat voxel indices and weights for trilinear interpolation."""
    scaled = np.asarray(point, dtype=np.float64) * n - 0.5
    i0 = np.floor(scaled).astype(np.int64)
    i0 = np.clip(i0, 0, n - 2)
    frac = np.clip(scaled - i0, 0.0, 1.0)

    idx, wgt = [], []
    for dx in (0, 1):
        wx = 1.0 - frac[0] if dx == 0 else frac[0]
        for dy in (0, 1):
            wy = 1.0 - frac[1] if dy == 0 else frac[1]
            for dz in (0, 1):
                wz = 1.0 - frac[2] if dz == 0 else frac[2]
                ii = i0 + np.array([dx, dy, dz], dtype=np.int64)
                flat = (ii[0] * n + ii[1]) * n + ii[2]
                idx.append(flat)
                wgt.append(wx * wy * wz)
    return np.asarray(idx, dtype=np.int64), np.asarray(wgt, dtype=np.float64)


def sample_vector_field_trilinear(B_grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate B_grid at one point x."""
    n = B_grid.shape[0]
    flat = B_grid.reshape((-1, 3))
    vox, w = trilinear_weights(x, n)
    return np.sum(flat[vox] * w[:, None], axis=0)


def build_rm_response_coo(
    n: int,
    observer: np.ndarray,
    pulsar_pos: np.ndarray,
    pulsar_dir: np.ndarray,
    pulsar_dist: np.ndarray,
    *,
    n_steps: int = 96,
    c_rm: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build sparse COO matrix A for RM = A @ B_flat.

    The midpoint rule approximates
        RM_p = C_RM ∫ n_e(x_obs+s n_hat) B(x).n_hat ds.
    """
    rows, cols, vals = [], [], []
    for p in range(len(pulsar_pos)):
        nhat = pulsar_dir[p]
        dist = float(pulsar_dist[p])
        ss = (np.arange(n_steps, dtype=np.float64) + 0.5) / n_steps * dist
        ds = dist / n_steps
        ray_points = observer[None, :] + ss[:, None] * nhat[None, :]
        ne_ray = electron_density(ray_points)
        for x, ne in zip(ray_points, ne_ray):
            vox, w = trilinear_weights(x, n)
            pref = c_rm * float(ne) * ds
            for vf, ww in zip(vox, w):
                base = 3 * int(vf)
                for comp in range(3):
                    rows.append(p)
                    cols.append(base + comp)
                    vals.append(pref * float(ww) * float(nhat[comp]))
    return {
        "rows": np.asarray(rows, dtype=np.int64),
        "cols": np.asarray(cols, dtype=np.int64),
        "data": np.asarray(vals, dtype=np.float64),
        "shape": np.asarray([len(pulsar_pos), 3 * n**3], dtype=np.int64),
    }


def response_apply_numpy(coo: Dict[str, np.ndarray], b_flat: np.ndarray) -> np.ndarray:
    """Apply stored COO response in NumPy."""
    out = np.zeros(int(coo["shape"][0]), dtype=np.float64)
    np.add.at(out, coo["rows"], coo["data"] * b_flat[coo["cols"]])
    return out


def response_apply_jax(rows, cols, vals, b_flat, n_data: int):
    """Apply stored COO response in JAX, differentiably."""
    out = jnp.zeros((n_data,), dtype=b_flat.dtype)
    return out.at[rows].add(vals * b_flat[cols])


def endpoint_los_component(
    B_grid: np.ndarray, pulsar_pos: np.ndarray, pulsar_dir: np.ndarray
) -> np.ndarray:
    """Compute B_parallel at pulsar endpoints for validation."""
    vals = []
    for pos, nhat in zip(pulsar_pos, pulsar_dir):
        b = sample_vector_field_trilinear(B_grid, pos)
        vals.append(float(np.dot(b, nhat)))
    return np.asarray(vals)


def divergence_fd(B_grid: np.ndarray) -> np.ndarray:
    """Finite-difference divergence diagnostic on the cube."""
    n = B_grid.shape[0]
    dx = 1.0 / n
    return (
        np.gradient(B_grid[..., 0], dx, axis=0, edge_order=2)
        + np.gradient(B_grid[..., 1], dx, axis=1, edge_order=2)
        + np.gradient(B_grid[..., 2], dx, axis=2, edge_order=2)
    )


def save_npz(path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
