#!/usr/bin/env python3
"""
Ray-node Bx/By/Bz recovery diagnostics for the Cartesian+exact-rays mock.

This script is for the cartesian_plus_rays setup, where the RM quadrature
points along each line of sight are actual GraphGP model points. It compares
truth, div-free reconstruction, and standard-vector reconstruction directly at
those ray nodes, with no interpolation.

Outputs:
  06_ray_xyz_component_profiles.png
  07_ray_xyz_component_summary.png
  ray_component_recovery_summary.txt
  ray_component_recovery_metrics.csv

Example:
  python 06_ray_component_recovery_plus_rays.py \
    --divfree outputs/local_cartesian_plus_rays_recon_divfree_prior.npz \
    --standard outputs/local_cartesian_plus_rays_recon_standard_prior.npz \
    --outdir figures/local_cartesian_plus_rays_compare/ray_components
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rm_common import load_npz
except Exception:
    def load_npz(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}


plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11.0,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 260,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COMPONENTS = ["Bx", "By", "Bz"]
COMP_LATEX = [r"$B_x$", r"$B_y$", r"$B_z$"]
DIVFREE_COLOR = "#1f77b4"
STANDARD_COLOR = "#d95f02"
TRUTH_COLOR = "#222222"
GRID_ALPHA = 0.18


def scalar(x):
    a = np.asarray(x)
    return a.item() if a.shape == () else a


def scalar_str(d: Dict[str, np.ndarray], key: str, default: str = "unknown") -> str:
    return str(scalar(d.get(key, default)))


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def nrmse(x_true: np.ndarray, x_rec: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float).ravel()
    x_rec = np.asarray(x_rec, dtype=float).ravel()
    m = np.isfinite(x_true) & np.isfinite(x_rec)
    x_true = x_true[m]
    x_rec = x_rec[m]
    if x_true.size == 0:
        return np.nan
    denom = np.sqrt(np.mean(x_true**2))
    return float(np.sqrt(np.mean((x_rec - x_true) ** 2)) / max(denom, 1e-30))


def symmetric_limit(*arrays, q: float = 99.0, floor: float = 1e-8) -> float:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(vals), q)), floor)


def ray_arrays(div: Dict[str, np.ndarray], std: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scalar_str(div, "geometry") != "cartesian_plus_rays":
        raise ValueError(f"Expected geometry=cartesian_plus_rays, got {scalar_str(div, 'geometry')}.")
    ray_idx = np.asarray(div["ray_point_indices"], dtype=np.int64)
    r = np.asarray(div["ray_r_centres"], dtype=float)
    Btrue_all = np.asarray(div["B_true"], dtype=float)
    Bdiv_all = np.asarray(div["B_est"], dtype=float)
    Bstd_all = np.asarray(std["B_est"], dtype=float)
    Btrue = Btrue_all[ray_idx]  # (n_directions, n_ray_steps, 3)
    Bdiv = Bdiv_all[ray_idx]
    Bstd = Bstd_all[ray_idx]
    return r, Btrue, Bdiv, Bstd


def metric_rows(Btrue: np.ndarray, Bdiv: np.ndarray, Bstd: np.ndarray):
    rows = []
    for c, name in enumerate(COMPONENTS):
        t = Btrue[..., c]
        d = Bdiv[..., c]
        s = Bstd[..., c]
        rows.append({
            "component": name,
            "r_divfree": corrcoef_safe(t, d),
            "r_standard": corrcoef_safe(t, s),
            "delta_r": corrcoef_safe(t, d) - corrcoef_safe(t, s),
            "nrmse_divfree": nrmse(t, d),
            "nrmse_standard": nrmse(t, s),
            "nrmse_reduction": nrmse(t, s) - nrmse(t, d),
        })
    return rows


def save_metrics(outdir: Path, rows):
    csv_path = outdir / "ray_component_recovery_metrics.csv"
    fields = ["component", "r_divfree", "r_standard", "delta_r", "nrmse_divfree", "nrmse_standard", "nrmse_reduction"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = outdir / "ray_component_recovery_summary.txt"
    with txt_path.open("w") as f:
        f.write("Cartesian+ray node component recovery\n")
        f.write("======================================\n")
        f.write("Metrics are computed only at the explicit GraphGP ray quadrature nodes.\n")
        f.write("No interpolation is used. Positive delta_r and positive nRMSE_reduction favour the div-free prior.\n\n")
        for row in rows:
            f.write(
                f"{row['component']}: "
                f"r_divfree={row['r_divfree']:.6g}, r_standard={row['r_standard']:.6g}, "
                f"delta_r={row['delta_r']:+.6g}, "
                f"nrmSE_divfree={row['nrmse_divfree']:.6g}, nRMSE_standard={row['nrmse_standard']:.6g}, "
                f"nRMSE_reduction={row['nrmse_reduction']:+.6g}\n"
            )
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


def select_example_directions(Btrue: np.ndarray, Bdiv: np.ndarray, Bstd: np.ndarray, n_examples: int, seed: int) -> np.ndarray:
    # Pick a mixture of directions where div-free helps and directions with strong signal.
    n_dir = Btrue.shape[0]
    mean_signal = np.sqrt(np.mean(np.sum(Btrue**2, axis=-1), axis=1))
    div_err = np.sqrt(np.mean(np.sum((Bdiv - Btrue) ** 2, axis=-1), axis=1))
    std_err = np.sqrt(np.mean(np.sum((Bstd - Btrue) ** 2, axis=-1), axis=1))
    gain = std_err - div_err  # positive = div-free lower error
    finite = np.isfinite(gain) & np.isfinite(mean_signal)
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        return np.arange(min(n_examples, n_dir))
    selected = []
    # strongly improved directions
    for i in candidates[np.argsort(gain[candidates])[::-1]]:
        selected.append(int(i))
        if len(selected) >= max(1, n_examples // 2):
            break
    # high signal directions
    for i in candidates[np.argsort(mean_signal[candidates])[::-1]]:
        if int(i) not in selected:
            selected.append(int(i))
        if len(selected) >= n_examples:
            break
    # random fill, if needed
    rng = np.random.default_rng(seed)
    while len(selected) < min(n_examples, n_dir):
        i = int(rng.integers(0, n_dir))
        if i not in selected:
            selected.append(i)
    return np.asarray(selected[:n_examples], dtype=int)


def plot_component_profiles(outdir: Path, r: np.ndarray, Btrue: np.ndarray, Bdiv: np.ndarray, Bstd: np.ndarray, *, n_examples: int, seed: int):
    dirs = select_example_directions(Btrue, Bdiv, Bstd, n_examples=n_examples, seed=seed)
    n_cols = len(dirs)
    fig, axs = plt.subplots(3, n_cols, figsize=(3.15 * n_cols, 8.6), sharex=True, constrained_layout=True)
    if n_cols == 1:
        axs = axs[:, None]

    for j, idir in enumerate(dirs):
        for c, clabel in enumerate(COMP_LATEX):
            ax = axs[c, j]
            t = Btrue[idir, :, c]
            d = Bdiv[idir, :, c]
            s = Bstd[idir, :, c]
            ax.axhline(0.0, color="0.55", lw=0.8)
            ax.plot(r, t, color=TRUTH_COLOR, lw=2.0, label="truth")
            ax.plot(r, d, color=DIVFREE_COLOR, lw=1.6, label="div-free")
            ax.plot(r, s, color=STANDARD_COLOR, lw=1.6, label="standard")
            ax.grid(alpha=GRID_ALPHA)
            ax.set_ylabel(clabel)
            if c == 0:
                rd = corrcoef_safe(t, d)
                rs = corrcoef_safe(t, s)
                ax.set_title(f"ray direction {idir}\n$r_d={rd:.2f}$, $r_s={rs:.2f}$")
            if c == 2:
                ax.set_xlabel("distance from observer")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(r"Cartesian+ray recovery of $B_x,B_y,B_z$ along explicit line-of-sight nodes", y=1.075, fontsize=15)
    path = outdir / "06_ray_xyz_component_profiles.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_summary(outdir: Path, rows):
    comps = [row["component"] for row in rows]
    x = np.arange(len(comps), dtype=float)
    width = 0.34

    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)

    ax = axs[0]
    ax.bar(x - width / 2, [row["r_divfree"] for row in rows], width, label="div-free prior", color=DIVFREE_COLOR, alpha=0.88)
    ax.bar(x + width / 2, [row["r_standard"] for row in rows], width, label="standard vector prior", color=STANDARD_COLOR, alpha=0.88)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("correlation with truth")
    ax.set_title("Ray-node component correlation")
    ax.set_ylim(min(0.0, np.nanmin([row["r_divfree"] for row in rows] + [row["r_standard"] for row in rows]) - 0.05), 1.02)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    ax = axs[1]
    ax.bar(x - width / 2, [row["nrmse_divfree"] for row in rows], width, label="div-free prior", color=DIVFREE_COLOR, alpha=0.88)
    ax.bar(x + width / 2, [row["nrmse_standard"] for row in rows], width, label="standard vector prior", color=STANDARD_COLOR, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("normalized RMSE")
    ax.set_title("Ray-node component error")
    ax.set_ylim(0.0, 1.15 * max([row["nrmse_divfree"] for row in rows] + [row["nrmse_standard"] for row in rows] + [1e-6]))
    ax.grid(axis="y", alpha=GRID_ALPHA)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Component recovery on explicit RM line-of-sight nodes", y=1.16, fontsize=15)
    path = outdir / "07_ray_xyz_component_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_component_scatters(outdir: Path, Btrue: np.ndarray, Bdiv: np.ndarray, Bstd: np.ndarray, *, max_points: int, seed: int):
    rng = np.random.default_rng(seed)
    flat_true = Btrue.reshape(-1, 3)
    flat_div = Bdiv.reshape(-1, 3)
    flat_std = Bstd.reshape(-1, 3)
    n = flat_true.shape[0]
    idx = np.arange(n) if n <= max_points else np.sort(rng.choice(n, size=max_points, replace=False))

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for c, ax in enumerate(axs):
        x = flat_true[idx, c]
        yd = flat_div[idx, c]
        ys = flat_std[idx, c]
        lim = symmetric_limit(x, yd, ys, q=99.2)
        ax.scatter(x, yd, s=8, alpha=0.28, color=DIVFREE_COLOR, linewidths=0, label="div-free prior")
        ax.scatter(x, ys, s=8, alpha=0.28, color=STANDARD_COLOR, linewidths=0, label="standard vector prior")
        ax.plot([-lim, lim], [-lim, lim], "--", color="0.25", lw=1.1)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"true {COMP_LATEX[c]}")
        ax.set_ylabel(f"reconstructed {COMP_LATEX[c]}")
        ax.set_title(f"Ray-node {COMP_LATEX[c]}")
        ax.grid(alpha=GRID_ALPHA)
        ax.text(
            0.03, 0.97,
            f"div-free: r={corrcoef_safe(flat_true[:, c], flat_div[:, c]):.3f}, nRMSE={nrmse(flat_true[:, c], flat_div[:, c]):.3f}\n"
            f"standard: r={corrcoef_safe(flat_true[:, c], flat_std[:, c]):.3f}, nRMSE={nrmse(flat_true[:, c], flat_std[:, c]):.3f}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.28", alpha=0.94),
        )
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Reconstructed vs true Cartesian components at explicit ray nodes", y=1.12, fontsize=15)
    path = outdir / "08_ray_xyz_component_scatters.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divfree", type=Path, required=True)
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--max-scatter-points", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    div = load_npz(args.divfree)
    std = load_npz(args.standard)
    r, Btrue, Bdiv, Bstd = ray_arrays(div, std)

    rows = metric_rows(Btrue, Bdiv, Bstd)
    save_metrics(args.outdir, rows)
    plot_component_profiles(args.outdir, r, Btrue, Bdiv, Bstd, n_examples=args.n_examples, seed=args.seed)
    plot_summary(args.outdir, rows)
    plot_component_scatters(args.outdir, Btrue, Bdiv, Bstd, max_points=args.max_scatter_points, seed=args.seed)

    print("\nHeadline ray-node gains, div-free minus standard:")
    for row in rows:
        print(f"  {row['component']}: delta r={row['delta_r']:+.4f}, nRMSE reduction={row['nrmse_reduction']:+.4f}")


if __name__ == "__main__":
    main()
