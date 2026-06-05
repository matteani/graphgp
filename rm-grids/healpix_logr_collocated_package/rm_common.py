#!/usr/bin/env python3
"""Common utilities for GraphGP + NIFTy pulsar-RM mock reconstructions.

This file is self-contained apart from your installed `graphgp`, `jax`, `numpy`,
`scipy`, and `nifty.re`/`nifty8.re` packages.

Flattening convention
---------------------
For a field B with shape (Npoints, 3), the response acts on B.reshape(-1),
with column 3*i + c for point index i and component c.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp


def require_graphgp():
    try:
        import graphgp as gp  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import installed `graphgp`. Activate the environment where "
            "your vector GraphGP version is installed, or run `pip install -e .` "
            "from the GraphGP repository."
        ) from exc
    return gp


def import_jft():
    try:
        import nifty.re as jft  # type: ignore
        return jft
    except Exception:
        try:
            import nifty8.re as jft  # type: ignore
            return jft
        except Exception as exc:
            raise ImportError("Could not import NIFTy.re as `nifty.re` or `nifty8.re`.") from exc


def canonical_prior_kind(prior_kind: str) -> str:
    aliases = {
        "divfree": "divfree_rbf",
        "div_free": "divfree_rbf",
        "div_free_rbf": "divfree_rbf",
        "divergence_free": "divfree_rbf",
        "divergence_free_rbf": "divfree_rbf",
        "standard": "standard_vector_rbf",
        "standard_rbf": "standard_vector_rbf",
        "generic": "standard_vector_rbf",
        "nondivfree": "standard_vector_rbf",
        "non_divfree": "standard_vector_rbf",
    }
    prior_kind = aliases.get(str(prior_kind), str(prior_kind))
    if prior_kind not in {"divfree_rbf", "standard_vector_rbf"}:
        raise ValueError(f"Unknown prior_kind={prior_kind!r}")
    return prior_kind


def make_standard_vector_rbf_covariance(*, ell: float, sigma2: float = 1.0, periodic: bool = False):
    ell = float(ell)
    sigma2 = float(sigma2)
    periodic = bool(periodic)
    ell2 = ell * ell

    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)
        rr2 = jnp.dot(r, r)
        k = sigma2 * jnp.exp(-0.5 * rr2 / ell2)

        # Stabilise local block Cholesky on irregular HEALPix/log-r point clouds.
        same = jnp.linalg.norm(r) < 1e-10
        jitter = 1e-5

        return (k + jnp.where(same, jitter, 0.0)) * jnp.eye(3, dtype=k.dtype)

    return cov_elem


def make_graphgp_covariance(gp, prior_kind: str, *, ell: float, sigma2: float, periodic: bool):
    prior_kind = canonical_prior_kind(prior_kind)
    if prior_kind == "divfree_rbf":
        if not hasattr(gp, "extras") or not hasattr(gp.extras, "make_div_free_rbf_covariance"):
            raise AttributeError("Your installed graphgp does not expose gp.extras.make_div_free_rbf_covariance")
        base_cov = gp.extras.make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=periodic)

        # A tiny same-point diagonal stabilizer helps the local Cholesky factors on
        # irregular HEALPix x log-r point clouds, especially on float32 local GPUs.
        def cov_elem(x, y):
            K = base_cov(x, y)
            r = y - x
            if periodic:
                r = r - jnp.round(r)
            same = jnp.linalg.norm(r) < 1e-10
            return K + jnp.where(same, 1e-5, 0.0) * jnp.eye(3, dtype=K.dtype)

        return cov_elem, prior_kind
    return make_standard_vector_rbf_covariance(ell=ell, sigma2=sigma2, periodic=periodic), prior_kind


def make_cartesian_grid_points(n: int) -> Tuple[np.ndarray, np.ndarray]:
    coords = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))
    return coords, points


def fibonacci_sphere_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    dirs = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # random global rotation
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    dirs = dirs @ Q.T
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def sample_tomographic_cartesian_pulsars(
    rng: np.random.Generator,
    *,
    observer: np.ndarray,
    n_directions: int,
    n_shells: int,
    r_min: float,
    r_max: float,
    boundary_margin: float = 0.025,
):
    max_radius = 0.5 - boundary_margin
    if not (0.0 < r_min < r_max < max_radius):
        raise ValueError(
            f"Need 0 < r_min < r_max < {max_radius:.3f} so all Cartesian rays stay inside the unit cube."
        )
    base_dirs = fibonacci_sphere_directions(n_directions, rng)
    shell_distances = np.linspace(r_min, r_max, n_shells, dtype=np.float64)

    pos, dirs, dists, direction_id, shell_id = [], [], [], [], []
    for i, nhat in enumerate(base_dirs):
        for j, dist in enumerate(shell_distances):
            p = observer + dist * nhat
            if np.all(p > boundary_margin) and np.all(p < 1.0 - boundary_margin):
                pos.append(p); dirs.append(nhat); dists.append(dist); direction_id.append(i); shell_id.append(j)
    return (
        np.asarray(pos, dtype=np.float64),
        np.asarray(dirs, dtype=np.float64),
        np.asarray(dists, dtype=np.float64),
        np.asarray(direction_id, dtype=np.int64),
        np.asarray(shell_id, dtype=np.int64),
        shell_distances,
    )


def electron_density(x: np.ndarray) -> np.ndarray:
    p = np.asarray(x, dtype=np.float64)
    xx, yy, zz = p[..., 0], p[..., 1], p[..., 2]
    # Works for both unit-cube coordinates and centred radial coordinates.
    if np.nanmin(p) >= -1e-12 and np.nanmax(p) <= 1.0 + 1e-12:
        cx, cy, cz = 0.52, 0.47, 0.50
    else:
        cx, cy, cz = 0.0, 0.0, 0.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ne = 1.0 + 0.45 * np.exp(-r2 / (2.0 * 0.32**2))
    ne += 0.08 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    return np.clip(ne, 0.15, None)


def trilinear_weights(point: np.ndarray, n: int):
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
                idx.append(flat); wgt.append(wx * wy * wz)
    return np.asarray(idx, dtype=np.int64), np.asarray(wgt, dtype=np.float64)


def build_cartesian_rm_response(n, observer, pulsar_pos, pulsar_dir, pulsar_dist, *, n_steps=128, c_rm=1.0):
    rows, cols, vals = [], [], []
    for p, (nhat, dist) in enumerate(zip(pulsar_dir, pulsar_dist)):
        ss = (np.arange(n_steps, dtype=np.float64) + 0.5) / n_steps * float(dist)
        ds = float(dist) / n_steps
        ray_points = observer[None, :] + ss[:, None] * nhat[None, :]
        ne_ray = electron_density(ray_points)
        for x, ne in zip(ray_points, ne_ray):
            vox, w = trilinear_weights(x, n)
            pref = c_rm * float(ne) * ds
            for vf, ww in zip(vox, w):
                base = 3 * int(vf)
                for c in range(3):
                    rows.append(p); cols.append(base + c); vals.append(pref * float(ww) * float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def build_cartesian_endpoint_response(n, pulsar_pos, pulsar_dir):
    rows, cols, vals = [], [], []
    for p, (pos, nhat) in enumerate(zip(pulsar_pos, pulsar_dir)):
        vox, w = trilinear_weights(pos, n)
        for vf, ww in zip(vox, w):
            base = 3 * int(vf)
            for c in range(3):
                rows.append(p); cols.append(base + c); vals.append(float(ww) * float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def healpix_ring_angles(nside: int):
    # Minimal HEALPix RING pixel centres, no dependency on healpy.
    if nside < 1:
        raise ValueError("nside must be >= 1")
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    nl2 = 2 * nside
    nl4 = 4 * nside
    fact1 = 2.0 / (3.0 * nside)
    fact2 = 1.0 / (3.0 * nside * nside)
    theta = np.empty(npix, dtype=np.float64)
    phi = np.empty(npix, dtype=np.float64)
    for ipix in range(npix):
        ipix1 = ipix + 1
        if ipix1 <= ncap:
            iring = int(0.5 * (1.0 + np.sqrt(2.0 * ipix1 - 1.0)))
            iphi = ipix1 - 2 * iring * (iring - 1)
            z = 1.0 - (iring * iring) * fact2
            ang = (iphi - 0.5) * np.pi / (2.0 * iring)
        elif ipix1 <= npix - ncap:
            ip = ipix1 - ncap - 1
            iring = ip // nl4 + nside
            iphi = ip % nl4 + 1
            fodd = 0.5 * (1 + ((iring + nside) & 1))
            z = (nl2 - iring) * fact1
            ang = (iphi - fodd) * np.pi / (2.0 * nside)
        else:
            ip = npix - ipix1 + 1
            iring = int(0.5 * (1.0 + np.sqrt(2.0 * ip - 1.0)))
            iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
            z = -1.0 + (iring * iring) * fact2
            ang = (iphi - 0.5) * np.pi / (2.0 * iring)
        theta[ipix] = np.arccos(np.clip(z, -1.0, 1.0))
        phi[ipix] = np.mod(ang, 2.0 * np.pi)
    return theta, phi


def unit_vectors_from_angles(theta, phi):
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi); sp = np.sin(phi)
    return np.stack([st * cp, st * sp, ct], axis=-1)


def build_healpix_logr_points(nside: int, n_radial: int, r_min: float, r_max: float):
    theta, phi = healpix_ring_angles(nside)
    unit = unit_vectors_from_angles(theta, phi)
    radii = np.geomspace(r_min, r_max, n_radial).astype(np.float64)
    points = []
    shell_ids, pixel_ids = [], []
    for ishell, r in enumerate(radii):
        points.append(r * unit)
        shell_ids.append(np.full(unit.shape[0], ishell, dtype=np.int64))
        pixel_ids.append(np.arange(unit.shape[0], dtype=np.int64))
    return (
        np.concatenate(points, axis=0),
        radii,
        theta,
        phi,
        unit,
        np.concatenate(shell_ids),
        np.concatenate(pixel_ids),
    )


def make_healpix_pulsars(*, radii, unit, n_pulsar_directions, n_pulsar_shells, rng):
    npix = unit.shape[0]
    n_dir = min(int(n_pulsar_directions), npix)
    if n_dir <= 0 or n_dir >= npix:
        pix = np.arange(npix, dtype=np.int64)
    else:
        pix = np.sort(rng.choice(npix, size=n_dir, replace=False).astype(np.int64))
    # Avoid the innermost shell; use shells across the line of sight.
    shell_idx = np.unique(np.linspace(1, len(radii) - 1, int(n_pulsar_shells)).round().astype(np.int64))
    shell_distances = radii[shell_idx]
    pos, dirs, dists, pixel_id, shell_id = [], [], [], [], []
    for p in pix:
        nhat = unit[p]
        for sid, rr in zip(shell_idx, shell_distances):
            pos.append(rr * nhat); dirs.append(nhat); dists.append(rr); pixel_id.append(p); shell_id.append(sid)
    return (
        np.asarray(pos, dtype=np.float64),
        np.asarray(dirs, dtype=np.float64),
        np.asarray(dists, dtype=np.float64),
        np.asarray(pixel_id, dtype=np.int64),
        np.asarray(shell_id, dtype=np.int64),
        shell_distances,
    )


def _radial_interp_weights(radii: np.ndarray, r: float):
    lr = np.log(radii)
    x = np.log(float(np.clip(r, radii[0], radii[-1])))
    hi = int(np.searchsorted(lr, x, side="right"))
    hi = min(max(hi, 1), len(radii) - 1)
    lo = hi - 1
    t = (x - lr[lo]) / (lr[hi] - lr[lo] + 1e-30)
    return (lo, hi), (1.0 - t, t)


def build_healpix_logr_rm_response(*, radii, unit, pixel_ids, pulsar_dist, n_steps=192, c_rm=1.0):
    npix = unit.shape[0]
    rows, cols, vals = [], [], []
    for p, (pix, dist) in enumerate(zip(pixel_ids, pulsar_dist)):
        nhat = unit[int(pix)]
        # start just outside r_min so we stay within interpolation range
        s0 = radii[0]
        if float(dist) <= s0:
            continue
        ss = s0 + (np.arange(n_steps, dtype=np.float64) + 0.5) / n_steps * (float(dist) - s0)
        ds = (float(dist) - s0) / n_steps
        ray_points = ss[:, None] * nhat[None, :]
        ne_ray = electron_density(ray_points)
        for rr, ne in zip(ss, ne_ray):
            (lo, hi), (wlo, whi) = _radial_interp_weights(radii, rr)
            pref = c_rm * float(ne) * ds
            for shell, ww in ((lo, wlo), (hi, whi)):
                v = int(shell) * npix + int(pix)
                base = 3 * v
                for c in range(3):
                    rows.append(p); cols.append(base + c); vals.append(pref * float(ww) * float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def build_healpix_logr_endpoint_response(*, radii, unit, pixel_ids, pulsar_dist):
    npix = unit.shape[0]
    rows, cols, vals = [], [], []
    for p, (pix, dist) in enumerate(zip(pixel_ids, pulsar_dist)):
        nhat = unit[int(pix)]
        (lo, hi), (wlo, whi) = _radial_interp_weights(radii, float(dist))
        for shell, ww in ((lo, wlo), (hi, whi)):
            v = int(shell) * npix + int(pix)
            base = 3 * v
            for c in range(3):
                rows.append(p); cols.append(base + c); vals.append(float(ww) * float(nhat[c]))
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)


def response_apply_numpy(rows, cols, vals, b_flat, nout):
    out = np.zeros(int(nout), dtype=np.float64)
    np.add.at(out, np.asarray(rows, np.int64), np.asarray(vals, np.float64) * np.asarray(b_flat)[np.asarray(cols, np.int64)])
    return out


def response_apply_jax(rows, cols, vals, b_flat, nout: int):
    out = jnp.zeros((int(nout),), dtype=b_flat.dtype)
    return out.at[rows].add(vals * b_flat[cols])


def divergence_fd_cartesian(B_grid):
    n = B_grid.shape[0]
    dx = 1.0 / n
    return (
        np.gradient(B_grid[..., 0], dx, axis=0, edge_order=2)
        + np.gradient(B_grid[..., 1], dx, axis=1, edge_order=2)
        + np.gradient(B_grid[..., 2], dx, axis=2, edge_order=2)
    )


def sign_accuracy_masked(pred, true, q: float = 0.05):
    pred = np.asarray(pred)
    true = np.asarray(true)
    good = np.sign(pred) == np.sign(true)
    acc_all = float(np.mean(good)) if good.size else np.nan
    thresh = float(np.quantile(np.abs(true), q)) if true.size else np.nan
    mask = np.abs(true) > thresh
    acc_masked = float(np.mean(good[mask])) if np.any(mask) else acc_all
    return acc_all, acc_masked


def shape_with_dtype(jft, shape, dtype):
    if hasattr(jft, "ShapeWithDtype"):
        try:
            return jft.ShapeWithDtype(shape, dtype)
        except TypeError:
            return jft.ShapeWithDtype(shape, dtype=dtype)
    return shape


def make_initial_position(jft, domain, key):
    if not hasattr(jft, "random_like"):
        raise AttributeError("This NIFTy.re install has no `random_like`; please adapt initialization.")
    raw = jft.random_like(key, domain)
    return jft.Vector(raw) if hasattr(jft, "Vector") else raw


def compose_likelihood(jft, data, noise_cov_inv, model):
    try:
        gaussian = jft.Gaussian(data, noise_cov_inv)
    except TypeError:
        gaussian = jft.Gaussian(data=data, noise_cov_inv=noise_cov_inv)
    if hasattr(gaussian, "amend"):
        return gaussian.amend(model)
    return gaussian @ model


def call_optimize_kl(jft, likelihood, pos_init, key, *, n_vi_iterations, n_vi_samples, verbosity=1):
    sig = inspect.signature(jft.optimize_kl)
    params = sig.parameters
    kwargs = {}
    if "key" in params:
        kwargs["key"] = key
    if "n_total_iterations" in params:
        kwargs["n_total_iterations"] = n_vi_iterations
    elif "n_iterations" in params:
        kwargs["n_iterations"] = n_vi_iterations
    elif "n_iter" in params:
        kwargs["n_iter"] = n_vi_iterations
    if "n_samples" in params:
        kwargs["n_samples"] = n_vi_samples
    if "point_estimates" in params:
        kwargs["point_estimates"] = ()
    if "verbosity" in params:
        kwargs["verbosity"] = verbosity
    print(f"Calling jft.optimize_kl with kwargs: {list(kwargs.keys())}", flush=True)

    attempts = [
        lambda: jft.optimize_kl(likelihood, pos_init, **kwargs),
        lambda: jft.optimize_kl(likelihood, pos_init, n_vi_iterations, n_vi_samples, key, point_estimates=()),
        lambda: jft.optimize_kl(likelihood, pos_init, n_total_iterations=n_vi_iterations, n_samples=n_vi_samples, key=key, point_estimates=()),
        lambda: jft.optimize_kl(likelihood, pos_init, n_iterations=n_vi_iterations, n_samples=n_vi_samples, key=key, point_estimates=()),
    ]
    errors = []
    for attempt in attempts:
        try:
            out = attempt()
            if isinstance(out, tuple):
                return out[0], out[1:]
            return out, ()
        except TypeError as exc:
            errors.append(str(exc))
    raise TypeError("All optimize_kl call signatures failed:\n" + "\n---\n".join(errors))


def extract_xi_from_samples(samples: Any) -> np.ndarray:
    for attr in ("pos", "position", "mean", "center"):
        if hasattr(samples, attr):
            obj = getattr(samples, attr)
            try:
                return np.asarray(obj["xi"])
            except Exception:
                pass
    if hasattr(samples, "sample_stat"):
        for reducer in ("mean", np.mean):
            try:
                stat = samples.sample_stat(reducer)
                return np.asarray(stat["xi"])
            except Exception:
                pass
    try:
        xis = [np.asarray(s["xi"]) for s in samples]
        if xis:
            return np.mean(np.stack(xis, axis=0), axis=0)
    except Exception:
        pass
    raise RuntimeError("Could not extract xi from NIFTy.re output. Adapt extract_xi_from_samples for this NIFTy version.")


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        if hasattr(obj, "tolist"):
            return obj.tolist()
    except Exception:
        pass
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_npz(path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
