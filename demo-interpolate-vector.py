#!/usr/bin/env python3
"""
Demo: conditional GraphGP interpolation of a sampled vector field from a coarse
Cartesian grid to a finer Cartesian grid.

This uses two new helpers:

- gp.build_interpolation_graph(source_points, target_points, k=...)
- gp.interpolate_vector(graph, cov_elem, source_values, xi=...)

`xi=None` gives the GraphGP conditional mean on the target grid.
Non-zero `xi` adds a conditional residual sample on top of that mean.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("temp-figures")
OUTDIR.mkdir(exist_ok=True)


def import_local_graphgp():
    spec = importlib.util.find_spec("graphgp")
    if spec is not None:
        import graphgp as gp  # type: ignore
        return gp

    here = Path(__file__).resolve().parent
    init_path = here / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "graphgp",
        init_path,
        submodule_search_locations=[str(here)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load local graphgp package.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["graphgp"] = module
    spec.loader.exec_module(module)
    return module


gp = import_local_graphgp()


plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "font.size": 11,
})


def make_cell_centered_grid(n: int) -> tuple[jnp.ndarray, np.ndarray]:
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))
    return points, np.asarray(xs)


def plot_slice(ax_amp, ax_bz, values, coords, title):
    n = len(coords)
    zslice = n // 2
    extent = [coords[0], coords[-1], coords[0], coords[-1]]

    grid = np.asarray(values).reshape((n, n, n, 3))
    Bx = grid[:, :, zslice, 0]
    By = grid[:, :, zslice, 1]
    Bz = grid[:, :, zslice, 2]
    Bmag = np.linalg.norm(grid[:, :, zslice, :], axis=-1)

    im0 = ax_amp.imshow(Bmag.T, origin="lower", extent=extent, aspect="equal")
    skip = max(1, n // 10)
    Xq, Yq = np.meshgrid(coords[::skip], coords[::skip], indexing="ij")
    U = Bx[::skip, ::skip].T
    V = By[::skip, ::skip].T
    ax_amp.quiver(Xq, Yq, U, V, color="white", pivot="mid", scale=18)
    ax_amp.set_title(f"{title}: |B| with $(B_x,B_y)$")
    ax_amp.set_xlabel("x")
    ax_amp.set_ylabel("y")
    plt.colorbar(im0, ax=ax_amp, fraction=0.046, pad=0.04)

    vmax = np.nanpercentile(np.abs(Bz), 99)
    im1 = ax_bz.imshow(Bz.T, origin="lower", extent=extent, aspect="equal", vmin=-vmax, vmax=vmax)
    ax_bz.set_title(f"{title}: $B_z$")
    ax_bz.set_xlabel("x")
    ax_bz.set_ylabel("y")
    plt.colorbar(im1, ax=ax_bz, fraction=0.046, pad=0.04)


def main():
    rng = jr.key(7)

    # Coarse source grid and finer target grid.
    # The chosen sizes avoid exact coordinate overlap between source and target.
    n_coarse = 40
    n_fine = 50
    coarse_points, coarse_coords = make_cell_centered_grid(n_coarse)
    fine_points, fine_coords = make_cell_centered_grid(n_fine)

    # Step 1: sample a vector field on the coarse grid with the existing GraphGP API.
    graph_source = gp.build_graph(coarse_points, n0=64, k=8)
    cov_elem = gp.extras.make_div_free_rbf_covariance(ell=0.18, sigma2=1.0, periodic=False)
    rng, key = jr.split(rng)
    xi_source = jr.normal(key, (coarse_points.shape[0], 3))
    source_values = gp.generate_vector(graph_source, cov_elem, xi_source, fast_jit=True)

    # Step 2: build a graph whose initial nodes are exactly the already-sampled
    # coarse-grid points and whose refined nodes are the new fine-grid points.
    interp_graph = gp.build_interpolation_graph(coarse_points, fine_points, k=8)
    gp.check_graph(interp_graph)

    # Step 3a: conditional mean on the fine grid (set xi=0).
    mean_fine = gp.interpolate_vector(
        interp_graph,
        cov_elem,
        source_values,
        xi=None,
        fast_jit=True,
        jitter=1e-10,
    )

    # Step 3b: conditional sample on the fine grid (non-zero xi).
    rng, key = jr.split(rng)
    xi_fine = jr.normal(key, (fine_points.shape[0], 3))
    sample_fine = gp.interpolate_vector(
        interp_graph,
        cov_elem,
        source_values,
        xi=xi_fine,
        fast_jit=True,
        jitter=1e-10,
    )

    # Optional check: the helper can also return the full concatenated field,
    # and the original source values are preserved exactly.
    full_values = gp.interpolate_vector(
        interp_graph,
        cov_elem,
        source_values,
        xi=xi_fine,
        fast_jit=True,
        jitter=1e-10,
        return_full=True,
    )
    source_restore_err = float(jnp.max(jnp.abs(full_values[: len(coarse_points)] - source_values)))

    # Basic run summary.
    print("Conditional GraphGP interpolation demo")
    print("--------------------------------------")
    print(f"coarse points        : {coarse_points.shape[0]}")
    print(f"fine target points   : {fine_points.shape[0]}")
    print(f"graph depth batches  : {len(interp_graph.offsets)}")
    print(f"max source restore err: {source_restore_err:.3e}")
    print(f"mean_fine finite     : {bool(jnp.isfinite(mean_fine).all())}")
    print(f"sample_fine finite   : {bool(jnp.isfinite(sample_fine).all())}")

    # Visualize the source slice, the conditional mean, and one conditional sample.
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    plot_slice(axs[0, 0], axs[1, 0], source_values, coarse_coords, "coarse source field")
    plot_slice(axs[0, 1], axs[1, 1], mean_fine, fine_coords, "fine conditional mean")
    plot_slice(axs[0, 2], axs[1, 2], sample_fine, fine_coords, "fine conditional sample")
    fig.suptitle(
        "Left: sampled coarse field. Middle: conditional mean on fine grid. Right: conditional refinement sample.",
        y=1.02,
    )
    fig.savefig(OUTDIR / "demo_interpolate_vector.png", bbox_inches="tight")
    plt.close(fig)

    summary = OUTDIR / "demo_interpolate_vector_summary.txt"
    summary.write_text(
        "Conditional GraphGP interpolation demo\n"
        f"coarse points = {coarse_points.shape[0]}\n"
        f"fine target points = {fine_points.shape[0]}\n"
        f"graph depth batches = {len(interp_graph.offsets)}\n"
        f"max source restore err = {source_restore_err:.6e}\n"
        f"mean_fine finite = {bool(jnp.isfinite(mean_fine).all())}\n"
        f"sample_fine finite = {bool(jnp.isfinite(sample_fine).all())}\n"
        f"figure = {OUTDIR / 'demo_interpolate_vector.png'}\n"
    )


if __name__ == "__main__":
    main()
