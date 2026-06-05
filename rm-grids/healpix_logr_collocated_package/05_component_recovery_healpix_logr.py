#!/usr/bin/env python3
"""Component-wise HEALPix/log-r vector-field recovery plots.

This is the HEALPix/log-r analogue of the Cartesian component recovery plot.
It compares Bx, By, Bz on the spherical shells and produces:

  05_healpix_component_shell_scatter_maps.png
  06_healpix_component_summary.png
  07_healpix_component_radial_profiles.png
  component_recovery_healpix_metrics.csv
  component_recovery_healpix_summary.txt
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
    "axes.titlesize": 11.2,
    "axes.labelsize": 10.8,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 260,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DIVFREE_COLOR = "#1f77b4"
STANDARD_COLOR = "#d95f02"
TRUTH_COLOR = "#222222"
GRID_ALPHA = 0.18
COMPONENTS = ["Bx", "By", "Bz"]
COMP_LATEX = [r"$B_x$", r"$B_y$", r"$B_z$"]


def scalar(x):
    a = np.asarray(x)
    return a.item() if a.shape == () else a


def scalar_str(d: Dict[str, np.ndarray], key: str, default: str = "unknown") -> str:
    return str(scalar(d.get(key, default)))


def ensure_healpix_logr(B: np.ndarray, d: Dict[str, np.ndarray], *, name: str = "B") -> np.ndarray:
    B = np.asarray(B, dtype=np.float64)
    n_radial = int(np.asarray(d["n_radial"]).item())
    npix = int(np.asarray(d["npix"]).item())
    if B.shape == (n_radial, npix, 3):
        return B
    if B.shape == (n_radial * npix, 3):
        return B.reshape((n_radial, npix, 3))
    raise ValueError(f"{name} must have shape (n_radial,npix,3) or (n_radial*npix,3), got {B.shape}.")


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def nrmse(x_true: np.ndarray, x_rec: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=np.float64).ravel()
    x_rec = np.asarray(x_rec, dtype=np.float64).ravel()
    m = np.isfinite(x_true) & np.isfinite(x_rec)
    x_true = x_true[m]; x_rec = x_rec[m]
    if x_true.size == 0:
        return np.nan
    denom = np.sqrt(np.mean(x_true**2))
    return float(np.sqrt(np.mean((x_rec - x_true) ** 2)) / max(denom, 1e-30))


def slope_bias(x_true: np.ndarray, x_rec: np.ndarray) -> Tuple[float, float]:
    x_true = np.asarray(x_true, dtype=np.float64).ravel()
    x_rec = np.asarray(x_rec, dtype=np.float64).ravel()
    m = np.isfinite(x_true) & np.isfinite(x_rec)
    x = x_true[m]; y = x_rec[m]
    if x.size < 3 or np.var(x) == 0.0:
        return np.nan, np.nan
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    bias = float(np.mean(y - x))
    return slope, bias


def metrics_for_component(Btrue: np.ndarray, Brec: np.ndarray, comp: int, shell_idx: int) -> Dict[str, float]:
    true_full = Btrue[..., comp]
    rec_full = Brec[..., comp]
    true_shell = Btrue[shell_idx, :, comp]
    rec_shell = Brec[shell_idx, :, comp]
    full_slope, full_bias = slope_bias(true_full, rec_full)
    shell_slope, shell_bias = slope_bias(true_shell, rec_shell)
    return {
        "r_full": corrcoef_safe(true_full, rec_full),
        "nrmse_full": nrmse(true_full, rec_full),
        "slope_full": full_slope,
        "bias_full": full_bias,
        "r_shell": corrcoef_safe(true_shell, rec_shell),
        "nrmse_shell": nrmse(true_shell, rec_shell),
        "slope_shell": shell_slope,
        "bias_shell": shell_bias,
    }


def compute_all_metrics(Btrue, Bdiv, Bstd, shell_idx):
    out: Dict[str, Dict[str, Dict[str, float]]] = {"divfree": {}, "standard": {}}
    for c, cname in enumerate(COMPONENTS):
        out["divfree"][cname] = metrics_for_component(Btrue, Bdiv, c, shell_idx)
        out["standard"][cname] = metrics_for_component(Btrue, Bstd, c, shell_idx)
    return out


def save_metrics(outdir: Path, metrics):
    outdir.mkdir(parents=True, exist_ok=True)
    fields = [
        "prior", "component", "r_full", "nrmse_full", "slope_full", "bias_full",
        "r_shell", "nrmse_shell", "slope_shell", "bias_shell",
    ]
    csv_path = outdir / "component_recovery_healpix_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for prior in ["divfree", "standard"]:
            for comp in COMPONENTS:
                row = {"prior": prior, "component": comp}
                row.update(metrics[prior][comp])
                writer.writerow(row)

    txt_path = outdir / "component_recovery_healpix_summary.txt"
    with txt_path.open("w") as f:
        f.write("Component-wise HEALPix/log-r vector-field recovery\n")
        f.write("==================================================\n")
        for comp in COMPONENTS:
            f.write(f"\n{comp}\n")
            for prior in ["divfree", "standard"]:
                m = metrics[prior][comp]
                f.write(
                    f"  {prior:8s}: r_full={m['r_full']:.6g}, nrmse_full={m['nrmse_full']:.6g}, "
                    f"r_shell={m['r_shell']:.6g}, nrmse_shell={m['nrmse_shell']:.6g}\n"
                )
            df = metrics["divfree"][comp]
            st = metrics["standard"][comp]
            f.write(
                f"  gain: delta_r_full={df['r_full'] - st['r_full']:+.6g}, "
                f"nrmse_reduction_full={st['nrmse_full'] - df['nrmse_full']:+.6g}\n"
            )
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


def wrap_lon(phi: np.ndarray) -> np.ndarray:
    lon = np.asarray(phi, dtype=np.float64)
    lon = (lon + np.pi) % (2.0 * np.pi) - np.pi
    return lon


def symmetric_limit(*arrays, q: float = 99.0, floor: float = 1e-12) -> float:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(vals), q)), floor)


def add_scatter_panel(ax, true_shell, div_shell, std_shell, *, comp_label, metrics, max_points, seed):
    x = true_shell.ravel()
    yd = div_shell.ravel()
    ys = std_shell.ravel()
    n = x.size
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        x = x[idx]; yd = yd[idx]; ys = ys[idx]
    lim = symmetric_limit(x, yd, ys, q=99.2)
    ax.scatter(x, yd, s=13, alpha=0.36, color=DIVFREE_COLOR, linewidths=0, label="div-free prior")
    ax.scatter(x, ys, s=13, alpha=0.36, color=STANDARD_COLOR, linewidths=0, label="standard vector prior")
    ax.plot([-lim, lim], [-lim, lim], "--", lw=1.2, color="0.25")
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"true {comp_label}")
    ax.set_ylabel(f"reconstructed {comp_label}")
    ax.grid(alpha=GRID_ALPHA)
    md = metrics["divfree"][comp_label.replace("$", "").replace("B_", "B").replace("{", "").replace("}", "")]
    ms = metrics["standard"][comp_label.replace("$", "").replace("B_", "B").replace("{", "").replace("}", "")]
    # robust fallback for B_x style labels
    key = "Bx" if "x" in comp_label else ("By" if "y" in comp_label else "Bz")
    md = metrics["divfree"][key]
    ms = metrics["standard"][key]
    txt = (
        f"div-free: r={md['r_shell']:.3f}, nRMSE={md['nrmse_shell']:.3f}\n"
        f"standard: r={ms['r_shell']:.3f}, nRMSE={ms['nrmse_shell']:.3f}"
    )
    ax.text(0.03, 0.97, txt, ha="left", va="top", transform=ax.transAxes,
            fontsize=8.8, bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"))


def mollweide_scatter(ax, theta, phi, values, *, title, vlim, cmap="coolwarm", s=8):
    lon = wrap_lon(phi)
    lat = 0.5 * np.pi - theta
    sc = ax.scatter(lon, lat, c=values, s=s, cmap=cmap, vmin=-vlim, vmax=vlim, linewidths=0, rasterized=True)
    ax.set_title(title, pad=6)
    ax.grid(alpha=0.22)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return sc


def plot_shell_scatter_maps(outdir: Path, d, Btrue, Bdiv, Bstd, metrics, shell_idx, max_scatter_points):
    theta = np.asarray(d["theta_pix"], dtype=np.float64)
    phi = np.asarray(d["phi_pix"], dtype=np.float64)
    radii = np.asarray(d["radii"], dtype=np.float64)
    npix = len(theta)
    marker_size = 10 if npix <= 1024 else (5 if npix <= 4096 else 2.5)

    fig = plt.figure(figsize=(18.8, 14.8), constrained_layout=False)
    gs = fig.add_gridspec(4, 5, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.35, wspace=0.18)

    ax_scx = fig.add_subplot(gs[0, 0:2])
    ax_scy = fig.add_subplot(gs[0, 2:4])
    ax_txt = fig.add_subplot(gs[0, 4])
    add_scatter_panel(ax_scx, Btrue[shell_idx, :, 0], Bdiv[shell_idx, :, 0], Bstd[shell_idx, :, 0], comp_label=r"$B_x$", metrics=metrics, max_points=max_scatter_points, seed=0)
    add_scatter_panel(ax_scy, Btrue[shell_idx, :, 1], Bdiv[shell_idx, :, 1], Bstd[shell_idx, :, 1], comp_label=r"$B_y$", metrics=metrics, max_points=max_scatter_points, seed=1)
    handles, labels = ax_scx.get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975))

    ax_txt.axis("off")
    lines = [f"shell index {shell_idx}", f"r = {radii[shell_idx]:.3f}", "", "full-volume gains"]
    for comp in COMPONENTS:
        df = metrics["divfree"][comp]
        st = metrics["standard"][comp]
        lines.append(f"{comp}: Δr={df['r_full'] - st['r_full']:+.3f}")
        lines.append(f"    ΔnRMSE={st['nrmse_full'] - df['nrmse_full']:+.3f}")
    ax_txt.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10.2,
                bbox=dict(facecolor="white", edgecolor="0.82", boxstyle="round,pad=0.35"))

    col_titles = ["truth", "div-free recon", "standard recon", "div-free residual", "standard residual"]
    for row, (comp, lab) in enumerate(zip(range(3), COMP_LATEX), start=1):
        true_vals = Btrue[shell_idx, :, comp]
        div_vals = Bdiv[shell_idx, :, comp]
        std_vals = Bstd[shell_idx, :, comp]
        div_res = div_vals - true_vals
        std_res = std_vals - true_vals
        vlim_field = symmetric_limit(true_vals, div_vals, std_vals, q=98.5)
        vlim_res = symmetric_limit(div_res, std_res, q=98.5)
        vals = [true_vals, div_vals, std_vals, div_res, std_res]
        vlims = [vlim_field, vlim_field, vlim_field, vlim_res, vlim_res]
        for col in range(5):
            ax = fig.add_subplot(gs[row, col], projection="mollweide")
            sc = mollweide_scatter(
                ax, theta, phi, vals[col],
                title=(col_titles[col] if row == 1 else ""),
                vlim=vlims[col], s=marker_size,
            )
            if col == 0:
                ax.text(-0.14, 0.5, lab, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=13)
            cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
            cb.ax.tick_params(labelsize=7.5)

    fig.suptitle("HEALPix/log-r component recovery on a spherical shell", y=0.995, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.963, "Top: reconstructed-vs-true scatter at the displayed shell.  Maps: Cartesian components on the sky.", ha="center", fontsize=10.5)
    fig.savefig(outdir / "05_healpix_component_shell_scatter_maps.png", bbox_inches="tight")
    plt.close(fig)


def plot_summary(outdir: Path, metrics):
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    x = np.arange(3)
    width = 0.34
    div_r = np.array([metrics["divfree"][c]["r_full"] for c in COMPONENTS])
    std_r = np.array([metrics["standard"][c]["r_full"] for c in COMPONENTS])
    div_e = np.array([metrics["divfree"][c]["nrmse_full"] for c in COMPONENTS])
    std_e = np.array([metrics["standard"][c]["nrmse_full"] for c in COMPONENTS])
    dr = div_r - std_r
    de = std_e - div_e

    axs[0].bar(x - width/2, div_r, width, color=DIVFREE_COLOR, label="div-free prior")
    axs[0].bar(x + width/2, std_r, width, color=STANDARD_COLOR, label="standard vector prior")
    axs[0].set_xticks(x); axs[0].set_xticklabels(COMPONENTS)
    axs[0].set_ylabel("full-grid Pearson r")
    axs[0].set_title("Component correlation")
    axs[0].grid(axis="y", alpha=GRID_ALPHA)
    axs[0].set_ylim(0, max(1.0, np.nanmax([div_r, std_r]) * 1.15))

    axs[1].bar(x - width/2, div_e, width, color=DIVFREE_COLOR)
    axs[1].bar(x + width/2, std_e, width, color=STANDARD_COLOR)
    axs[1].set_xticks(x); axs[1].set_xticklabels(COMPONENTS)
    axs[1].set_ylabel("full-grid normalized RMSE")
    axs[1].set_title("Component error")
    axs[1].grid(axis="y", alpha=GRID_ALPHA)
    axs[1].set_ylim(0, np.nanmax([div_e, std_e]) * 1.25)

    axs[2].axhline(0.0, color="0.3", lw=1.0)
    axs[2].plot(x, dr, "o-", color=DIVFREE_COLOR, label=r"$\Delta r$")
    axs[2].plot(x, de, "s--", color=STANDARD_COLOR, label="nRMSE reduction")
    for i in range(3):
        axs[2].text(x[i], dr[i], f" {dr[i]:+.3f}", va="center", fontsize=8.8)
        axs[2].text(x[i], de[i], f" {de[i]:+.3f}", va="center", fontsize=8.8)
    axs[2].set_xticks(x); axs[2].set_xticklabels(COMPONENTS)
    axs[2].set_title("Div-free gain over standard")
    axs[2].grid(axis="y", alpha=GRID_ALPHA)
    axs[2].legend(frameon=False, loc="best")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("HEALPix/log-r full-grid component recovery", y=1.14, fontsize=15, fontweight="bold")
    fig.savefig(outdir / "06_healpix_component_summary.png", bbox_inches="tight")
    plt.close(fig)


def radial_profile(Btrue, Brec, comp: int) -> Tuple[np.ndarray, np.ndarray]:
    n_radial = Btrue.shape[0]
    rvals = np.full(n_radial, np.nan)
    evals = np.full(n_radial, np.nan)
    for s in range(n_radial):
        rvals[s] = corrcoef_safe(Btrue[s, :, comp], Brec[s, :, comp])
        evals[s] = nrmse(Btrue[s, :, comp], Brec[s, :, comp])
    return rvals, evals


def plot_radial_profiles(outdir: Path, d, Btrue, Bdiv, Bstd):
    radii = np.asarray(d["radii"], dtype=np.float64)
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    for comp, lab in enumerate(COMP_LATEX):
        rd, ed = radial_profile(Btrue, Bdiv, comp)
        rs, es = radial_profile(Btrue, Bstd, comp)
        axs[0].plot(radii, rd, "o-", label=f"{lab} div-free")
        axs[0].plot(radii, rs, "s--", label=f"{lab} standard")
        axs[1].plot(radii, ed, "o-")
        axs[1].plot(radii, es, "s--")
    axs[0].set_xscale("log"); axs[1].set_xscale("log")
    axs[0].set_xlabel("radius"); axs[1].set_xlabel("radius")
    axs[0].set_ylabel("per-shell Pearson r")
    axs[1].set_ylabel("per-shell nRMSE")
    axs[0].set_title("Correlation by radial shell")
    axs[1].set_title("Error by radial shell")
    for ax in axs:
        ax.grid(alpha=GRID_ALPHA)
    axs[0].legend(frameon=False, fontsize=8.2, ncol=2)
    fig.suptitle("HEALPix/log-r radial recovery profiles", y=1.05, fontsize=15, fontweight="bold")
    fig.savefig(outdir / "07_healpix_component_radial_profiles.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divfree", type=Path, required=True)
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--shell-index", type=int, default=-1, help="shell to display; -1 means outermost shell")
    ap.add_argument("--max-scatter-points", type=int, default=5000)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    div = load_npz(args.divfree)
    std = load_npz(args.standard)
    if scalar_str(div, "geometry") != "healpix_logr":
        raise ValueError("This plotting script expects geometry=healpix_logr.")

    Btrue = ensure_healpix_logr(div["B_true"], div, name="B_true")
    Bdiv = ensure_healpix_logr(div["B_est"], div, name="B_est divfree")
    Bstd = ensure_healpix_logr(std["B_est"], div, name="B_est standard")
    n_radial = Btrue.shape[0]
    shell_idx = args.shell_index if args.shell_index >= 0 else n_radial - 1
    if not (0 <= shell_idx < n_radial):
        raise ValueError(f"shell-index={shell_idx} out of range [0,{n_radial-1}]")

    metrics = compute_all_metrics(Btrue, Bdiv, Bstd, shell_idx)
    save_metrics(args.outdir, metrics)
    plot_shell_scatter_maps(args.outdir, div, Btrue, Bdiv, Bstd, metrics, shell_idx, args.max_scatter_points)
    plot_summary(args.outdir, metrics)
    plot_radial_profiles(args.outdir, div, Btrue, Bdiv, Bstd)

    print("\nHeadline full-grid gains, div-free minus standard:")
    for comp in COMPONENTS:
        df = metrics["divfree"][comp]
        st = metrics["standard"][comp]
        print(f"  {comp}: delta r={df['r_full'] - st['r_full']:+.4f}, nRMSE reduction={st['nrmse_full'] - df['nrmse_full']:+.4f}")
    print("\nMain figures:")
    print(f"  {args.outdir / '05_healpix_component_shell_scatter_maps.png'}")
    print(f"  {args.outdir / '06_healpix_component_summary.png'}")
    print(f"  {args.outdir / '07_healpix_component_radial_profiles.png'}")


if __name__ == "__main__":
    main()
