#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rm_common import load_npz


def import_base_plotter():
    path = Path(__file__).resolve().parent / "03_plot_compare_reconstructions.py"
    if not path.exists():
        path = Path("03_plot_compare_reconstructions.py").resolve()
    spec = importlib.util.spec_from_file_location("base_rm_plotter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base plotting script from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scalar(x):
    a = np.asarray(x)
    return a.item() if a.shape == () else a


def scalar_str(d, key, default="unknown"):
    return str(scalar(d.get(key, default)))


def _set_3d_equal(ax, pts):
    pts = np.asarray(pts, dtype=float)
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    cen = 0.5 * (mins + maxs)
    rad = 0.5 * np.nanmax(maxs - mins)
    rad = max(float(rad), 1e-6)
    ax.set_xlim(cen[0] - rad, cen[0] + rad)
    ax.set_ylim(cen[1] - rad, cen[1] + rad)
    ax.set_zlim(cen[2] - rad, cen[2] + rad)
    ax.set_box_aspect((1, 1, 1))


def draw_unit_cube(ax, color="0.35", alpha=0.45, lw=0.9):
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        ax.plot(*zip(corners[a], corners[b]), color=color, alpha=alpha, lw=lw)
    return corners


def plot_pulsar_geometry_plus_rays(outdir: Path, d):
    pos = np.asarray(d["pulsar_pos"], dtype=float)
    obs = np.asarray(d.get("observer", np.array([0.5, 0.5, 0.5])), dtype=float)
    dist = np.asarray(d["pulsar_dist"], dtype=float)
    shell = np.asarray(d.get("shell_id", np.zeros(len(pos))), dtype=int)
    pts = np.asarray(d["points"], dtype=float)
    is_base = np.asarray(d.get("point_is_base_grid", np.zeros(len(pts), dtype=bool)), dtype=bool)
    is_ray = np.asarray(d.get("point_is_ray_node", np.zeros(len(pts), dtype=bool)), dtype=bool)
    is_endpoint = np.asarray(d.get("point_is_endpoint", np.zeros(len(pts), dtype=bool)), dtype=bool)

    rng = np.random.default_rng(0)
    max_lines = min(220, len(pos))
    idx = np.arange(len(pos)) if len(pos) <= max_lines else np.sort(rng.choice(len(pos), size=max_lines, replace=False))

    fig = plt.figure(figsize=(13.0, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    axr = fig.add_subplot(gs[0, 1])

    # Subsample point cloud for visual clarity.
    ray_idx = np.flatnonzero(is_ray)
    base_idx = np.flatnonzero(is_base)
    end_idx = np.flatnonzero(is_endpoint)
    if len(base_idx) > 1800:
        base_idx = rng.choice(base_idx, size=1800, replace=False)
    if len(ray_idx) > 5000:
        ray_idx = rng.choice(ray_idx, size=5000, replace=False)

    ax3.scatter(pts[base_idx, 0], pts[base_idx, 1], pts[base_idx, 2], s=1.5, c="0.75", alpha=0.18, depthshade=False, label="Cartesian base grid")
    ax3.scatter(pts[ray_idx, 0], pts[ray_idx, 1], pts[ray_idx, 2], s=2.5, c="#377eb8", alpha=0.20, depthshade=False, label="direct RM ray nodes")
    if len(end_idx):
        show_end = end_idx if len(end_idx) <= 1200 else rng.choice(end_idx, size=1200, replace=False)
        ax3.scatter(pts[show_end, 0], pts[show_end, 1], pts[show_end, 2], s=7, c="#e41a1c", alpha=0.65, depthshade=False, label="pulsar endpoints")

    for ii in idx:
        ax3.plot([obs[0], pos[ii, 0]], [obs[1], pos[ii, 1]], [obs[2], pos[ii, 2]], color="0.25", alpha=0.10, lw=0.55)
    ax3.scatter([obs[0]], [obs[1]], [obs[2]], marker="*", s=160, c="k", depthshade=False, label="observer")
    corners = draw_unit_cube(ax3)
    _set_3d_equal(ax3, np.vstack([corners, pos, obs[None, :]]))
    ax3.set_title("Hybrid Cartesian + exact ray-node GraphGP point cloud")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=22, azim=-58)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)

    shell_unique = np.unique(shell)
    cmap = plt.get_cmap("viridis")
    dist_norm = (dist - np.nanmin(dist)) / (np.nanmax(dist) - np.nanmin(dist) + 1e-12)
    colors = cmap(dist_norm)
    for s in shell_unique:
        m = shell == s
        jitter = np.linspace(-0.15, 0.15, max(2, int(m.sum())))[: int(m.sum())]
        axr.scatter(dist[m], np.full(int(m.sum()), s, dtype=float) + jitter, c=colors[m], s=16, alpha=0.85, edgecolors="none")
        med = float(np.median(dist[m]))
        axr.axvline(med, color="0.55", ls="--", lw=0.8, alpha=0.4)
    axr.set_xlabel("distance from observer")
    axr.set_ylabel("pulsar shell index")
    axr.set_title("Tomographic pulsar placement")
    axr.set_yticks(shell_unique)
    axr.grid(alpha=0.18)

    fig.savefig(outdir / "00_pulsar_geometry_and_sightlines.png", bbox_inches="tight")
    plt.close(fig)


def plot_base_cartesian_slices(outdir: Path, div, std):
    if scalar_str(div, "geometry") != "cartesian_plus_rays":
        return
    if "B_true_cartesian_base" not in div or "B_est_cartesian_base" not in div:
        return
    Btrue = np.asarray(div["B_true_cartesian_base"], dtype=float)
    Bdiv = np.asarray(div["B_est_cartesian_base"], dtype=float)
    Bstd = np.asarray(std["B_est_cartesian_base"], dtype=float)
    n = Btrue.shape[0]
    z = n // 2
    panels = [
        (Btrue[:, :, z, 2], "truth $B_z$ on base grid"),
        (Bdiv[:, :, z, 2], "div-free prior"),
        (Bstd[:, :, z, 2], "standard prior"),
        (Bdiv[:, :, z, 2] - Bstd[:, :, z, 2], "div-free minus standard"),
    ]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([p[0].ravel() for p in panels[:3]])), 99))
    vmax = max(vmax, 1e-6)
    fig, axs = plt.subplots(1, 4, figsize=(14.0, 3.7), constrained_layout=True)
    for ax, (arr, title) in zip(axs, panels):
        v = float(np.nanpercentile(np.abs(arr), 99)) if "minus" in title else vmax
        v = max(v, 1e-6)
        im = ax.imshow(arr.T, origin="lower", vmin=-v, vmax=v, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Central $z$-slice on the Cartesian base-grid subset", y=1.02)
    fig.savefig(outdir / "04_cartesian_base_Bz_slice.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divfree", type=Path, required=True)
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    base = import_base_plotter()
    div = load_npz(args.divfree)
    std = load_npz(args.standard)

    stats_div = base.sign_stats(div["bpar_est"], div["bpar_true"])
    stats_std = base.sign_stats(std["bpar_est"], std["bpar_true"])
    rows = [base.metric_row("divfree_rbf", div, stats_div), base.metric_row("standard_vector_rbf", std, stats_std)]
    paired = base.paired_counts(div, std)

    csv, txt = base.write_summary(args.outdir, rows, paired)
    plot_pulsar_geometry_plus_rays(args.outdir, div)
    base.plot_summary(args.outdir, rows, div, std, paired)
    base.plot_endpoint_scatter(args.outdir, div, std)
    base.plot_rm_fit(args.outdir, div, std)
    plot_base_cartesian_slices(args.outdir, div, std)

    print(f"Wrote {txt}")
    print(f"Wrote {csv}")
    print(f"sign accuracy divfree={rows[0]['sign_accuracy']:.4f}, standard={rows[1]['sign_accuracy']:.4f}, delta={rows[0]['sign_accuracy'] - rows[1]['sign_accuracy']:+.4f}")
    print(f"paired counts: divfree_only={paired['divfree_only']}, standard_only={paired['standard_only']}, mcnemar_p={paired['mcnemar_p_value']:.4g}")


if __name__ == "__main__":
    main()
