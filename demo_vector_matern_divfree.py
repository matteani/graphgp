#!/usr/bin/env python3
"""
demo_vector_matern_divfree.py

Experimental divergence-free Matérn vector-field kernel for GraphGP.

This script builds a homogeneous, isotropic, non-helical magnetic covariance

    M_ij(r) = M_N(r) delta_ij + (M_L(r) - M_N(r)) rhat_i rhat_j

with

    M_L(r) = half-integer Matérn_{nu=p+1/2}(r; ell)
    M_N(r) = M_L(r) + (r / 2) dM_L/dr

so that partial_i M_ij = 0 for the isotropic tensor.  M_N is therefore not an
independent free choice: it is the Matérn-derived transverse correlation required
by the divergence-free constraint.

Outputs:
    temp-figures/matern_slice_los_projection.png
    temp-figures/matern_realspace_correlations.png
    temp-figures/gaussian_vs_matern_slices.png
    temp-figures/matern_kernel_summary.txt

Run from the repository root:
    python demo_vector_matern_divfree.py
"""

from __future__ import annotations

import gc
import importlib.util
import math
import sys
from pathlib import Path
from typing import Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

OUTDIR = Path("temp-figures")
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "legend.fontsize": 9.0,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "figure.dpi": 150,
    "savefig.dpi": 240,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def import_local_graphgp():
    spec = importlib.util.find_spec("graphgp")
    if spec is not None:
        import graphgp as gp  # type: ignore
        return gp

    here = Path(__file__).resolve().parent
    candidates = [here / "graphgp" / "__init__.py", here / "__init__.py"]
    for init_path in candidates:
        if init_path.exists():
            spec = importlib.util.spec_from_file_location(
                "graphgp",
                init_path,
                submodule_search_locations=[str(init_path.parent)],
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["graphgp"] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError("Could not find/import local graphgp package.")


gp = import_local_graphgp()


# ------------------------------
# Graph construction with periodic predecessor search
# ------------------------------


def make_points_3d(n: int):
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    return jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3)), np.asarray(xs)


def predecessor_neighbors(points_tree, n0: int, k: int, periodic: bool = True, boxsize: float = 1.0):
    pts = np.asarray(points_tree)
    neigh = np.empty((len(pts) - n0, k), dtype=int)
    for i in range(n0, len(pts)):
        prev = pts[:i]
        if periodic:
            d = prev - pts[i]
            d = d - boxsize * np.round(d / boxsize)
            dist = np.linalg.norm(d, axis=1)
        else:
            dist = np.linalg.norm(prev - pts[i], axis=1)
        idx = np.argpartition(dist, k - 1)[:k]
        idx = idx[np.argsort(dist[idx], kind="stable")]
        neigh[i - n0] = idx
    return jnp.asarray(neigh)


def build_graph_local(points, *, n0: int, k: int, periodic: bool = True, boxsize: float = 1.0):
    points_tree, split_dims, indices = gp.build_tree(points)
    neighbors = predecessor_neighbors(points_tree, n0=n0, k=k, periodic=periodic, boxsize=boxsize)
    depths = gp.compute_depths(neighbors, n0=n0)
    pts_ord, idx_ord, neigh_ord, depths = gp.order_by_depth(points_tree, indices, neighbors, depths)
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
    return gp.Graph(pts_ord, neigh_ord, offsets, idx_ord)


# ------------------------------
# Divergence-free Gaussian and Matérn covariance tensors
# ------------------------------


def _matern_half_integer_coefficients(p: int):
    """Coefficients for P_p(u) in Matérn_{p+1/2}: exp(-x) P_p(2x)."""
    if p < 0:
        raise ValueError("p must be >= 0")
    coeffs = []
    for i in range(p + 1):
        coeffs.append(
            math.factorial(p)
            * math.factorial(p + i)
            / (math.factorial(i) * math.factorial(p - i) * math.factorial(2 * p))
        )
    return coeffs


def _polyval_and_derivative(coeffs, u):
    """Evaluate polynomial and its analytic derivative using Horner recursion."""
    coeffs = jnp.asarray(coeffs, dtype=jnp.result_type(u, float))
    val = jnp.zeros_like(u, dtype=jnp.result_type(u, float))
    der = jnp.zeros_like(u, dtype=jnp.result_type(u, float))
    for c in coeffs:
        der = der * u + val
        val = val * u + c
    return val, der


def half_integer_matern_and_derivative(r, *, p: int, ell: float, sigma2: float = 1.0):
    """
    Analytic half-integer Matérn and radial derivative.

    For nu = p + 1/2, write

        M_L(r) = sigma2 * exp(-x) * P_p(2x),
        x = sqrt(2p+1) r / ell.

    The derivative is evaluated analytically as

        dM_L/dr = sigma2 * exp(-x) * alpha * [2 P'_p(2x) - P_p(2x)],
        alpha = sqrt(2p+1) / ell.

    No automatic differentiation or finite differences are used.
    """
    rr = jnp.asarray(r)
    alpha = jnp.sqrt(2.0 * p + 1.0) / ell
    x = alpha * rr
    u = 2.0 * x
    poly, dpoly_du = _polyval_and_derivative(_matern_half_integer_coefficients(p), u)
    expx = jnp.exp(-x)
    ml = sigma2 * expx * poly
    dml_dr = sigma2 * expx * alpha * (2.0 * dpoly_du - poly)
    return ml, dml_dr


def half_integer_matern(r, *, p: int, ell: float, sigma2: float = 1.0):
    """Matérn covariance for nu = p + 1/2 using the analytic polynomial form."""
    return half_integer_matern_and_derivative(r, p=p, ell=ell, sigma2=sigma2)[0]


def matern_parallel_perp_trace_analytic(r, *, ell: float, sigma2: float, p: int):
    """Return analytic M_L, M_N and trace for the divergence-free tensor."""
    rr = jnp.asarray(r)
    ml, dml_dr = half_integer_matern_and_derivative(rr, p=p, ell=ell, sigma2=sigma2)
    mn = ml + 0.5 * rr * dml_dr
    return ml, mn, ml + 2.0 * mn


def make_div_free_matern_covariance(
    *,
    ell: float,
    sigma2: float = 1.0,
    p: int = 2,
    periodic: bool = True,
    jitter: float = 0.0,
):
    """
    Experimental divergence-free 3x3 tensor with Matérn longitudinal kernel.

    M_L is a half-integer Matérn with nu=p+1/2.  M_N is derived from M_L via
    M_N = M_L + (r/2) dM_L/dr.  This is the isotropic divergence-free constraint.
    """
    ell = float(ell)
    sigma2 = float(sigma2)
    p = int(p)
    periodic = bool(periodic)
    jitter = float(jitter)

    def cov_elem(x, y):
        rvec = y - x
        if periodic:
            rvec = rvec - jnp.round(rvec)
        rr = jnp.linalg.norm(rvec)
        ml, mn, _ = matern_parallel_perp_trace_analytic(rr, ell=ell, sigma2=sigma2, p=p)
        safe_rr = jnp.maximum(rr, 1e-12)
        rhat = rvec / safe_rr
        P = jnp.outer(rhat, rhat)
        I = jnp.eye(3, dtype=ml.dtype)
        K = mn * I + (ml - mn) * P
        if jitter != 0.0:
            K = K + jnp.where(rr < 1e-12, jitter, 0.0) * I
        return K

    cov_elem._graphgp_kernel_name = "div_free_matern_vector_3_experimental"
    cov_elem._graphgp_ell = ell
    cov_elem._graphgp_sigma2 = sigma2
    cov_elem._graphgp_matern_p = p
    cov_elem._graphgp_periodic = periodic
    return cov_elem


def make_div_free_rbf_covariance(*, ell: float, sigma2: float = 1.0, periodic: bool = True):
    """Current Gaussian/RBF divergence-free tensor, normalized so K(0)=sigma2 I."""
    ell2 = ell * ell

    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)
        I = jnp.eye(3, dtype=base.dtype)
        ML = sigma2 * base
        MN = sigma2 * (1.0 - 0.5 * rr2 / ell2) * base
        safe_rr2 = jnp.maximum(rr2, 1e-24)
        P = jnp.outer(r, r) / safe_rr2
        return MN * I + (ML - MN) * P

    return cov_elem


def theory_matern_parallel_perp_trace(r, *, ell: float, sigma2: float, p: int):
    """Fully analytic Matérn M_L, M_N and trace, returned as NumPy arrays."""
    cpar, cperp, ctrace = matern_parallel_perp_trace_analytic(
        jnp.asarray(r), ell=ell, sigma2=sigma2, p=p
    )
    return np.asarray(cpar), np.asarray(cperp), np.asarray(ctrace)


def theory_rbf_parallel_perp_trace(r, *, ell: float, sigma2: float):
    q = (np.asarray(r) / ell) ** 2
    cpar = sigma2 * np.exp(-0.5 * q)
    cperp = sigma2 * (1.0 - 0.5 * q) * np.exp(-0.5 * q)
    ctrace = cpar + 2.0 * cperp
    return cpar, cperp, ctrace


# ------------------------------
# Real-space correlation diagnostics
# ------------------------------


def overlapping_shift_pair(Bgrid, axis, h, periodic=True):
    if h == 0:
        return np.asarray(Bgrid), np.asarray(Bgrid)
    if periodic:
        return np.asarray(Bgrid), np.roll(np.asarray(Bgrid), -int(h), axis=axis)
    sl0 = [slice(None)] * 4
    sl1 = [slice(None)] * 4
    sl0[axis] = slice(0, Bgrid.shape[axis] - h)
    sl1[axis] = slice(h, Bgrid.shape[axis])
    return np.asarray(Bgrid[tuple(sl0)]), np.asarray(Bgrid[tuple(sl1)])


def correlation_tensor_axis_lags(Bgrid, max_lag, periodic=True):
    lags = np.arange(0, max_lag + 1)
    Ct = np.zeros((3, 3, 3, len(lags)), dtype=np.float64)
    for a in range(3):
        for ih, h in enumerate(lags):
            A, C = overlapping_shift_pair(Bgrid, a, int(h), periodic=periodic)
            for i in range(3):
                for j in range(3):
                    Ct[a, i, j, ih] = np.mean(A[..., i] * C[..., j])
    return lags, Ct


def summarize_correlations(Bgrid, ell, periodic=True):
    n = Bgrid.shape[0]
    max_lag = min(n // 2, max(6, int(np.ceil(8.0 * ell * n))))
    lags, Ct = correlation_tensor_axis_lags(Bgrid, max_lag, periodic=periodic)
    r = lags / n
    cpar_axes = np.stack([Ct[a, a, a, :] for a in range(3)], axis=0)
    cperp_axes = np.stack([
        0.5 * (Ct[a, (a + 1) % 3, (a + 1) % 3, :] + Ct[a, (a + 2) % 3, (a + 2) % 3, :])
        for a in range(3)
    ], axis=0)

    def sym_pair(i, j):
        arr = np.stack([0.5 * (Ct[a, i, j, :] + Ct[a, j, i, :]) for a in range(3)], axis=0)
        return arr.mean(axis=0), arr.std(axis=0)

    cxy_mean, cxy_std = sym_pair(0, 1)
    cxz_mean, cxz_std = sym_pair(0, 2)
    cyz_mean, cyz_std = sym_pair(1, 2)
    return {
        "r": r,
        "cpar_mean": cpar_axes.mean(axis=0),
        "cpar_std": cpar_axes.std(axis=0),
        "cperp_mean": cperp_axes.mean(axis=0),
        "cperp_std": cperp_axes.std(axis=0),
        "cxy_mean": cxy_mean,
        "cxy_std": cxy_std,
        "cxz_mean": cxz_mean,
        "cxz_std": cxz_std,
        "cyz_mean": cyz_mean,
        "cyz_std": cyz_std,
    }


# ------------------------------
# Field generation and plotting
# ------------------------------


def generate_pair(
    *,
    n: int,
    n0: int,
    k: int,
    ell: float,
    sigma2: float,
    matern_p: int,
    seed: int,
    periodic: bool = True,
):
    points, coords_1d = make_points_3d(n)
    graph = build_graph_local(points, n0=n0, k=k, periodic=periodic, boxsize=1.0)
    xi = jr.normal(jr.key(seed), (points.shape[0], 3))

    cov_gauss = make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=periodic)
    cov_matern = make_div_free_matern_covariance(ell=ell, sigma2=sigma2, p=matern_p, periodic=periodic, jitter=1e-9)

    B_gauss = np.asarray(gp.generate_vector(graph, cov_gauss, xi, fast_jit=True)).reshape((n, n, n, 3))
    gc.collect()
    B_matern = np.asarray(gp.generate_vector(graph, cov_matern, xi, fast_jit=True)).reshape((n, n, n, 3))
    spacing = (1.0 / n, 1.0 / n, 1.0 / n)
    return B_gauss, B_matern, coords_1d, spacing


def field_metrics(Bgrid, ell):
    Bmag = np.linalg.norm(Bgrid, axis=-1)
    Brms = float(np.sqrt(np.mean(Bmag ** 2)))
    return {"B_rms": Brms, "Bmag_rms": Brms, "ell": ell}


def plot_slice_los_projection(Bgrid, coords_1d, *, ell, sigma2, matern_p, outpath: Path):
    n = Bgrid.shape[0]
    zslice = n // 2
    z0 = coords_1d[zslice]
    Bx = Bgrid[:, :, zslice, 0]
    By = Bgrid[:, :, zslice, 1]
    Bz = Bgrid[:, :, zslice, 2]
    Babs = np.linalg.norm(Bgrid[:, :, zslice, :], axis=-1)
    los_bz = np.trapezoid(Bgrid[..., 2], x=coords_1d, axis=2)

    X, Y = np.meshgrid(coords_1d, coords_1d, indexing="xy")
    U = Bx.T
    V = By.T
    mag = np.hypot(U, V)
    Udir = U / np.where(mag > 0.0, mag, 1.0)
    Vdir = V / np.where(mag > 0.0, mag, 1.0)

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    extent = [0.0, 1.0, 0.0, 1.0]

    im0 = axs[0].imshow(Babs.T, origin="lower", extent=extent, aspect="equal")
    axs[0].quiver(X, Y, Udir, Vdir, angles="xy", scale_units="xy", scale=44.0,
                  color="k", width=0.0022, alpha=0.85, pivot="mid")
    axs[0].set_title(rf"$(B_x,B_y)$ slice at $z_0={z0:.3f}$")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.03, label=r"$|B|$")

    vmax_bz = np.nanpercentile(np.abs(Bz), 99.0)
    im1 = axs[1].imshow(Bz.T, origin="lower", extent=extent, aspect="equal", vmin=-vmax_bz, vmax=vmax_bz)
    axs[1].set_title(r"Signed $B_z(x,y,z_0)$")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.03, label=r"$B_z$")

    vmax_los = np.nanpercentile(np.abs(los_bz), 99.0)
    im2 = axs[2].imshow(los_bz.T, origin="lower", extent=extent, aspect="equal", vmin=-vmax_los, vmax=vmax_los)
    axs[2].set_title(r"Line-of-sight projection $\int B_z\,dz$")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.03, label=r"$\int B_z dz$")

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        rf"Experimental divergence-free Matérn vector field: $\nu={matern_p}+1/2$, $\ell={ell:.3f}$, $\sigma^2={sigma2:.1f}$",
        fontsize=14.2,
        fontweight="bold",
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_realspace_matern(Bgrid, *, ell, sigma2, matern_p, periodic, outpath: Path):
    corr = summarize_correlations(Bgrid, ell=ell, periodic=periodic)
    r = corr["r"]
    x = r / ell
    cpar_th, cperp_th, ctrace_th = theory_matern_parallel_perp_trace(r, ell=ell, sigma2=sigma2, p=matern_p)
    ctrace = corr["cpar_mean"] + 2.0 * corr["cperp_mean"]
    ctrace_std = np.sqrt(corr["cpar_std"] ** 2 + 4.0 * corr["cperp_std"] ** 2)

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.3), sharex=True, constrained_layout=True)
    top = [
        (axs[0, 0], corr["cpar_mean"] / sigma2, corr["cpar_std"] / sigma2, cpar_th / sigma2, r"$C_{\parallel}(r)/\sigma^2=M_L/\sigma^2$"),
        (axs[0, 1], corr["cperp_mean"] / sigma2, corr["cperp_std"] / sigma2, cperp_th / sigma2, r"$C_{\perp}(r)/\sigma^2=M_N/\sigma^2$"),
        (axs[0, 2], ctrace / sigma2, ctrace_std / sigma2, ctrace_th / sigma2, r"${\rm tr}\,C(r)/\sigma^2$"),
    ]
    legend_handles = None
    for ax, mean, std, theory, title in top:
        h1 = ax.plot(x, mean, "o", ms=4.0, label="sample mean")
        h2 = ax.fill_between(x, mean - std, mean + std, alpha=0.20, label="axis scatter")
        h3 = ax.plot(x, theory, lw=2.0, label="theory")
        ax.axhline(0.0, color="0.45", ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"$r/\ell$")
        if legend_handles is None:
            legend_handles = [h1[0], h2, h3[0]]

    bottom = [
        (axs[1, 0], corr["cxy_mean"] / sigma2, corr["cxy_std"] / sigma2, r"$C_{xy}(r)/\sigma^2$"),
        (axs[1, 1], corr["cxz_mean"] / sigma2, corr["cxz_std"] / sigma2, r"$C_{xz}(r)/\sigma^2$"),
        (axs[1, 2], corr["cyz_mean"] / sigma2, corr["cyz_std"] / sigma2, r"$C_{yz}(r)/\sigma^2$"),
    ]
    for ax, mean, std, title in bottom:
        ax.plot(x, mean, "o", ms=4.0)
        ax.fill_between(x, mean - std, mean + std, alpha=0.20)
        ax.axhline(0.0, lw=2.0)
        ax.set_title(title)
        ax.set_xlabel(r"$r/\ell$")

    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="lower center", ncol=3, frameon=True)
    fig.suptitle(
        rf"Real-space correlations for divergence-free Matérn kernel, $\nu={matern_p}+1/2$\n"
        rf"$M_N(r)=M_L(r)+(r/2)\partial_r M_L(r)$ enforces $\nabla\cdot B=0$ for the isotropic tensor.",
        y=1.035,
        fontsize=14.5,
        fontweight="bold",
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_gaussian_vs_matern_slices(B_gauss, B_matern, coords_1d, *, ell, sigma2, matern_p, outpath: Path):
    fields = [("Gaussian/RBF", B_gauss), (rf"Matérn $\nu={matern_p}+1/2$", B_matern)]
    n = B_gauss.shape[0]
    zslice = n // 2
    extent = [0.0, 1.0, 0.0, 1.0]
    X, Y = np.meshgrid(coords_1d, coords_1d, indexing="xy")

    all_abs = np.concatenate([np.linalg.norm(B[:, :, zslice, :], axis=-1).ravel() for _, B in fields])
    all_bz = np.concatenate([B[:, :, zslice, 2].ravel() for _, B in fields])
    norm_abs = mcolors.Normalize(vmin=np.nanpercentile(all_abs, 1.0), vmax=np.nanpercentile(all_abs, 99.0))
    vmax_bz = np.nanpercentile(np.abs(all_bz), 99.0)

    fig, axs = plt.subplots(2, 2, figsize=(11.8, 10.0), constrained_layout=True)
    for row, (name, B) in enumerate(fields):
        Bx = B[:, :, zslice, 0]
        By = B[:, :, zslice, 1]
        Bz = B[:, :, zslice, 2]
        Babs = np.linalg.norm(B[:, :, zslice, :], axis=-1)
        U, V = Bx.T, By.T
        mag = np.hypot(U, V)
        Udir = U / np.where(mag > 0.0, mag, 1.0)
        Vdir = V / np.where(mag > 0.0, mag, 1.0)

        im0 = axs[row, 0].imshow(Babs.T, origin="lower", extent=extent, aspect="equal", norm=norm_abs)
        axs[row, 0].quiver(X, Y, Udir, Vdir, angles="xy", scale_units="xy", scale=44.0,
                           color="k", width=0.0021, alpha=0.85, pivot="mid")
        axs[row, 0].set_title(rf"{name}: $(B_x,B_y)$ over $|B|$")

        im1 = axs[row, 1].imshow(Bz.T, origin="lower", extent=extent, aspect="equal", vmin=-vmax_bz, vmax=vmax_bz)
        axs[row, 1].set_title(rf"{name}: $B_z(x,y,z_0)$")

    for ax in axs.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im0, ax=axs[:, 0], fraction=0.046, pad=0.025, label=r"$|B|$")
    fig.colorbar(im1, ax=axs[:, 1], fraction=0.046, pad=0.025, label=r"$B_z$")
    fig.suptitle(
        rf"Same graph and same white-noise draw: Gaussian vs experimental Matérn divergence-free kernel\n"
        rf"$z_0={coords_1d[zslice]:.3f}$, $\ell={ell:.3f}$, $\sigma^2={sigma2:.1f}$",
        fontsize=14.2,
        fontweight="bold",
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def write_summary(path: Path, *, n, n0, k, ell, sigma2, matern_p, B_gauss, B_matern):
    lines = [
        "Experimental divergence-free Matérn vector kernel",
        "=================================================",
        f"n={n}, N={n**3}, n0={n0}, k={k}",
        f"ell={ell}, sigma2={sigma2}, matern p={matern_p}, nu={matern_p}+1/2",
        "",
        "Tensor:",
        "  M_ij(r) = M_N(r) delta_ij + (M_L(r)-M_N(r)) rhat_i rhat_j",
        "  M_L(r) = half-integer Matern_{nu=p+1/2}(r; ell)",
        "  M_N(r) = M_L(r) + (r/2) dM_L/dr  [analytic derivative]",
        "",
        "Field RMS:",
        f"  Gaussian/RBF B_rms = {np.sqrt(np.mean(np.sum(B_gauss**2, axis=-1))):.6g}",
        f"  Matérn       B_rms = {np.sqrt(np.mean(np.sum(B_matern**2, axis=-1))):.6g}",
        "",
        "Figures:",
        "  matern_slice_los_projection.png",
        "  matern_realspace_correlations.png",
        "  gaussian_vs_matern_slices.png",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Saved {path}")


def main():
    # Laptop-friendly defaults. For publication-quality figures try n=40, n0=1000, k=18.
    n = 40
    n0 = 1000
    k = 18
    ell = 0.08
    sigma2 = 1.0
    matern_p = 2  # nu = 5/2; analytic polynomial and derivative are used.
    seed = 0
    periodic = True

    print("Generating Gaussian and Matérn fields...")
    B_gauss, B_matern, coords_1d, spacing = generate_pair(
        n=n,
        n0=n0,
        k=k,
        ell=ell,
        sigma2=sigma2,
        matern_p=matern_p,
        seed=seed,
        periodic=periodic,
    )

    plot_slice_los_projection(
        B_matern,
        coords_1d,
        ell=ell,
        sigma2=sigma2,
        matern_p=matern_p,
        outpath=OUTDIR / "matern_slice_los_projection.png",
    )
    gc.collect()
    plot_realspace_matern(
        B_matern,
        ell=ell,
        sigma2=sigma2,
        matern_p=matern_p,
        periodic=periodic,
        outpath=OUTDIR / "matern_realspace_correlations.png",
    )
    gc.collect()
    plot_gaussian_vs_matern_slices(
        B_gauss,
        B_matern,
        coords_1d,
        ell=ell,
        sigma2=sigma2,
        matern_p=matern_p,
        outpath=OUTDIR / "gaussian_vs_matern_slices.png",
    )
    write_summary(
        OUTDIR / "matern_kernel_summary.txt",
        n=n,
        n0=n0,
        k=k,
        ell=ell,
        sigma2=sigma2,
        matern_p=matern_p,
        B_gauss=B_gauss,
        B_matern=B_matern,
    )
    print("Wrote outputs to", OUTDIR.resolve())


if __name__ == "__main__":
    main()
