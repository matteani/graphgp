#!/usr/bin/env python3
"""
Radial-shell recovery diagnostics for the Cartesian+exact-rays mock.

This script asks whether the reconstruction quality changes with distance from
observer.  It works on the explicit GraphGP ray nodes, so no interpolation is used.
It computes component-wise correlation and nRMSE in radial bins, with bootstrap
uncertainty by resampling line-of-sight directions.

Example:
  python 09_ray_shell_recovery_plus_rays.py \
    --divfree outputs/local_cartesian_plus_rays_recon_divfree_prior.npz \
    --standard outputs/local_cartesian_plus_rays_recon_standard_prior.npz \
    --outdir figures/local_cartesian_plus_rays_compare/ray_shells
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
    "axes.titlesize": 12.0,
    "axes.labelsize": 11.0,
    "legend.fontsize": 9.4,
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
GAIN_COLOR = "#2ca02c"
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


def nrmse_safe(x_true: np.ndarray, x_rec: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float).ravel()
    x_rec = np.asarray(x_rec, dtype=float).ravel()
    m = np.isfinite(x_true) & np.isfinite(x_rec)
    x_true = x_true[m]
    x_rec = x_rec[m]
    if x_true.size == 0:
        return np.nan
    denom = np.sqrt(np.mean(x_true**2))
    return float(np.sqrt(np.mean((x_rec - x_true) ** 2)) / max(denom, 1e-30))


def ray_arrays(div: Dict[str, np.ndarray], std: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return r, ray_idx, Btrue, Bdiv, Bstd


def bin_radial_indices(r: np.ndarray, n_bins: int):
    r = np.asarray(r, dtype=float)
    if n_bins <= 0 or n_bins >= len(r):
        bins = [(i, i + 1, np.array([i], dtype=int)) for i in range(len(r))]
        return bins
    edges = np.linspace(r.min(), r.max(), n_bins + 1)
    bins = []
    for i in range(n_bins):
        if i == n_bins - 1:
            q = np.where((r >= edges[i]) & (r <= edges[i + 1]))[0]
        else:
            q = np.where((r >= edges[i]) & (r < edges[i + 1]))[0]
        if q.size:
            bins.append((float(edges[i]), float(edges[i + 1]), q.astype(int)))
    return bins


def stat_for_bin(Btrue, Brec, dirs, qidx, comp, kind):
    t = Btrue[np.ix_(dirs, qidx, [comp])].reshape(-1)
    r = Brec[np.ix_(dirs, qidx, [comp])].reshape(-1)
    if kind == "corr":
        return corrcoef_safe(t, r)
    if kind == "nrmse":
        return nrmse_safe(t, r)
    raise ValueError(kind)


def bootstrap_stat(Btrue, Brec, qidx, comp, kind, *, nboot, rng):
    ndir = Btrue.shape[0]
    dirs_all = np.arange(ndir)
    point = stat_for_bin(Btrue, Brec, dirs_all, qidx, comp, kind)
    if nboot <= 0:
        return point, np.nan, np.nan
    vals = np.empty(nboot, dtype=float)
    for b in range(nboot):
        dirs = rng.integers(0, ndir, size=ndir)
        vals[b] = stat_for_bin(Btrue, Brec, dirs, qidx, comp, kind)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return point, np.nan, np.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return point, float(lo), float(hi)


def response_use_counts(d: Dict[str, np.ndarray], ray_idx: np.ndarray) -> np.ndarray:
    """Number of RM data rows using each explicit ray node, shape (ndir,nray)."""
    cols = np.asarray(d.get("response_cols", []), dtype=np.int64)
    npoints = int(np.asarray(d["points"]).shape[0])
    if cols.size == 0:
        return np.full(ray_idx.shape, np.nan)
    point_cols = cols // 3
    entry_counts = np.bincount(point_cols, minlength=npoints)
    # Each RM use contributes three component columns, so divide by 3.
    return entry_counts[ray_idx] / 3.0


def point_density_by_bin(d: Dict[str, np.ndarray], observer: np.ndarray, bins) -> np.ndarray:
    points = np.asarray(d["points"], dtype=float)
    rr = np.linalg.norm(points - observer[None, :], axis=1)
    density = []
    for lo, hi, _q in bins:
        count = np.count_nonzero((rr >= lo) & (rr < hi if hi < bins[-1][1] else rr <= hi))
        vol = (4.0 / 3.0) * np.pi * (hi**3 - lo**3)
        density.append(count / max(vol, 1e-30))
    return np.asarray(density, dtype=float)


def compute_metrics(div, std, *, n_bins, nboot, seed):
    r, ray_idx, Btrue, Bdiv, Bstd = ray_arrays(div, std)
    bins = bin_radial_indices(r, n_bins)
    rng = np.random.default_rng(seed)
    use_counts = response_use_counts(div, ray_idx)
    observer = np.asarray(div.get("observer", np.array([0.5, 0.5, 0.5])), dtype=float)
    density = point_density_by_bin(div, observer, bins)

    rows = []
    for ib, (lo, hi, qidx) in enumerate(bins):
        qcenter = float(np.mean(r[qidx]))
        shell_use = float(np.nanmean(use_counts[:, qidx]))
        for c, cname in enumerate(COMPONENTS):
            rd, rd_lo, rd_hi = bootstrap_stat(Btrue, Bdiv, qidx, c, "corr", nboot=nboot, rng=rng)
            rs, rs_lo, rs_hi = bootstrap_stat(Btrue, Bstd, qidx, c, "corr", nboot=nboot, rng=rng)
            nd, nd_lo, nd_hi = bootstrap_stat(Btrue, Bdiv, qidx, c, "nrmse", nboot=nboot, rng=rng)
            ns, ns_lo, ns_hi = bootstrap_stat(Btrue, Bstd, qidx, c, "nrmse", nboot=nboot, rng=rng)
            rows.append({
                "bin": ib,
                "r_lo": float(lo),
                "r_hi": float(hi),
                "r_mid": qcenter,
                "n_ray_nodes_per_direction": int(len(qidx)),
                "mean_rm_uses_per_ray_node": shell_use,
                "point_density_proxy": float(density[ib]),
                "component": cname,
                "r_divfree": rd,
                "r_divfree_lo95": rd_lo,
                "r_divfree_hi95": rd_hi,
                "r_standard": rs,
                "r_standard_lo95": rs_lo,
                "r_standard_hi95": rs_hi,
                "delta_r": rd - rs,
                "nrmse_divfree": nd,
                "nrmse_divfree_lo95": nd_lo,
                "nrmse_divfree_hi95": nd_hi,
                "nrmse_standard": ns,
                "nrmse_standard_lo95": ns_lo,
                "nrmse_standard_hi95": ns_hi,
                "nrmse_reduction": ns - nd,
            })
    return rows


def arr(rows, component, key):
    return np.asarray([row[key] for row in rows if row["component"] == component], dtype=float)


def err_from_ci(y, lo, hi):
    return np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])


def plot_correlation(outdir: Path, rows):
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True, constrained_layout=True)
    for ax, comp, clab in zip(axs, COMPONENTS, COMP_LATEX):
        x = arr(rows, comp, "r_mid")
        yd = arr(rows, comp, "r_divfree")
        ys = arr(rows, comp, "r_standard")
        yd_lo = arr(rows, comp, "r_divfree_lo95")
        yd_hi = arr(rows, comp, "r_divfree_hi95")
        ys_lo = arr(rows, comp, "r_standard_lo95")
        ys_hi = arr(rows, comp, "r_standard_hi95")
        ax.errorbar(x, yd, yerr=err_from_ci(yd, yd_lo, yd_hi), fmt="o-", lw=1.7, capsize=3, color=DIVFREE_COLOR, label="div-free prior")
        ax.errorbar(x, ys, yerr=err_from_ci(ys, ys_lo, ys_hi), fmt="s-", lw=1.7, capsize=3, color=STANDARD_COLOR, label="standard vector prior")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_title(clab)
        ax.set_xlabel("distance from observer")
        ax.grid(alpha=GRID_ALPHA)
    axs[0].set_ylabel("correlation with truth")
    axs[0].legend(loc="lower right", frameon=True)
    fig.suptitle("Component recovery versus radius at explicit ray nodes", fontsize=15)
    path = outdir / "09_ray_shell_component_correlation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_nrmse(outdir: Path, rows):
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True, constrained_layout=True)
    for ax, comp, clab in zip(axs, COMPONENTS, COMP_LATEX):
        x = arr(rows, comp, "r_mid")
        yd = arr(rows, comp, "nrmse_divfree")
        ys = arr(rows, comp, "nrmse_standard")
        yd_lo = arr(rows, comp, "nrmse_divfree_lo95")
        yd_hi = arr(rows, comp, "nrmse_divfree_hi95")
        ys_lo = arr(rows, comp, "nrmse_standard_lo95")
        ys_hi = arr(rows, comp, "nrmse_standard_hi95")
        ax.errorbar(x, yd, yerr=err_from_ci(yd, yd_lo, yd_hi), fmt="o-", lw=1.7, capsize=3, color=DIVFREE_COLOR, label="div-free prior")
        ax.errorbar(x, ys, yerr=err_from_ci(ys, ys_lo, ys_hi), fmt="s-", lw=1.7, capsize=3, color=STANDARD_COLOR, label="standard vector prior")
        ax.set_title(clab)
        ax.set_xlabel("distance from observer")
        ax.grid(alpha=GRID_ALPHA)
    axs[0].set_ylabel("normalized RMSE")
    axs[0].legend(loc="upper left", frameon=True)
    fig.suptitle("Component amplitude error versus radius at explicit ray nodes", fontsize=15)
    path = outdir / "10_ray_shell_component_nrmse.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_improvement_support(outdir: Path, rows):
    # One row per radial bin using averaged component gains.
    bins = sorted(set(row["bin"] for row in rows))
    x = np.asarray([np.mean([row["r_mid"] for row in rows if row["bin"] == b]) for b in bins])
    delta_r_mean = np.asarray([np.nanmean([row["delta_r"] for row in rows if row["bin"] == b]) for b in bins])
    nrmse_red_mean = np.asarray([np.nanmean([row["nrmse_reduction"] for row in rows if row["bin"] == b]) for b in bins])
    uses = np.asarray([np.nanmean([row["mean_rm_uses_per_ray_node"] for row in rows if row["bin"] == b]) for b in bins])
    density = np.asarray([np.nanmean([row["point_density_proxy"] for row in rows if row["bin"] == b]) for b in bins])
    density_norm = density / np.nanmax(density) if np.nanmax(density) > 0 else density
    uses_norm = uses / np.nanmax(uses) if np.nanmax(uses) > 0 else uses

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.4), constrained_layout=True)
    ax = axs[0]
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.plot(x, delta_r_mean, "o-", color=DIVFREE_COLOR, lw=1.8, label=r"mean $\Delta r$")
    ax.plot(x, nrmse_red_mean, "s-", color=GAIN_COLOR, lw=1.8, label="mean nRMSE reduction")
    ax.set_xlabel("distance from observer")
    ax.set_ylabel("div-free improvement")
    ax.set_title("Average gain over $B_x,B_y,B_z$")
    ax.legend(frameon=True)
    ax.grid(alpha=GRID_ALPHA)

    ax = axs[1]
    ax.plot(x, uses, "o-", color="0.25", lw=1.8, label="RM uses per ray node")
    ax.set_xlabel("distance from observer")
    ax.set_ylabel("mean number of RM equations using node")
    ax.grid(alpha=GRID_ALPHA)
    ax2 = ax.twinx()
    ax2.plot(x, density_norm, "s--", color="0.55", lw=1.4, label="relative point density")
    ax2.set_ylabel("relative point density")
    ax.set_title("Why inner shells can be better constrained")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    fig.suptitle("Radial support and reconstruction gain", fontsize=15)
    path = outdir / "11_ray_shell_improvement_and_support.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def save_rows(outdir: Path, rows):
    csv_path = outdir / "ray_shell_recovery_metrics.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = outdir / "ray_shell_recovery_summary.txt"
    with txt_path.open("w") as f:
        f.write("Radial-shell component recovery at explicit ray nodes\n")
        f.write("=====================================================\n")
        f.write("Correlation/nRMSE are computed in radial bins. Bootstrap CIs resample line-of-sight directions, not individual ray nodes.\n")
        f.write("Positive delta_r and positive nRMSE_reduction favour the div-free prior.\n\n")
        for comp in COMPONENTS:
            dr = arr(rows, comp, "delta_r")
            nr = arr(rows, comp, "nrmse_reduction")
            f.write(f"{comp}: mean delta_r={np.nanmean(dr):+.6g}, mean nRMSE_reduction={np.nanmean(nr):+.6g}\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--divfree", type=Path, required=True)
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=8, help="coarse radial bins; use 0 for every ray step separately")
    ap.add_argument("--nboot", type=int, default=800, help="bootstrap resamples over sightline directions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    div = load_npz(args.divfree)
    std = load_npz(args.standard)
    rows = compute_metrics(div, std, n_bins=args.n_bins, nboot=args.nboot, seed=args.seed)
    save_rows(args.outdir, rows)
    plot_correlation(args.outdir, rows)
    plot_nrmse(args.outdir, rows)
    plot_improvement_support(args.outdir, rows)

    print("\nMain figures:")
    print(f"  {args.outdir / '09_ray_shell_component_correlation.png'}")
    print(f"  {args.outdir / '10_ray_shell_component_nrmse.png'}")
    print(f"  {args.outdir / '11_ray_shell_improvement_and_support.png'}")


if __name__ == "__main__":
    main()
