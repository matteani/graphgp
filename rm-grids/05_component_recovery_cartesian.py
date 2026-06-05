#!/usr/bin/env python3
"""
Component-wise Cartesian vector-field recovery plots for RM reconstructions.

This script compares the true Cartesian vector field against two reconstructions:
  1. div-free GraphGP prior
  2. standard vector RBF prior

It creates publication-style figures showing:
  - Bx and By reconstructed-vs-true scatter plots on one z-slice
  - Bx, By, Bz slice maps: truth, div-free, standard, and residuals
  - component-wise full-volume correlations and normalized RMSE

Example:

python 05_component_recovery_cartesian.py \
  --divfree outputs/local_cartesian_recon_divfree_prior.npz \
  --standard outputs/local_cartesian_recon_standard_prior.npz \
  --outdir figures/local_cartesian_compare/component_recovery
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


def ensure_cartesian_grid(B: np.ndarray, *, name: str = "B") -> np.ndarray:
    """Return B as (n,n,n,3). Accepts either grid-shaped or flat (N,3)."""
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 4 and B.shape[-1] == 3:
        return B
    if B.ndim == 2 and B.shape[-1] == 3:
        n = round(B.shape[0] ** (1.0 / 3.0))
        if n**3 != B.shape[0]:
            raise ValueError(f"{name} has shape {B.shape}; cannot infer cubic grid side length.")
        return B.reshape((n, n, n, 3))
    raise ValueError(f"{name} must have shape (n,n,n,3) or (n^3,3), got {B.shape}.")


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def nrmse(x_true: np.ndarray, x_rec: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=np.float64).ravel()
    x_rec = np.asarray(x_rec, dtype=np.float64).ravel()
    m = np.isfinite(x_true) & np.isfinite(x_rec)
    x_true = x_true[m]
    x_rec = x_rec[m]
    denom = np.sqrt(np.mean(x_true**2))
    return float(np.sqrt(np.mean((x_rec - x_true) ** 2)) / max(denom, 1e-30))


def slope_bias(x_true: np.ndarray, x_rec: np.ndarray) -> Tuple[float, float]:
    x_true = np.asarray(x_true, dtype=np.float64).ravel()
    x_rec = np.asarray(x_rec, dtype=np.float64).ravel()
    m = np.isfinite(x_true) & np.isfinite(x_rec)
    x = x_true[m]
    y = x_rec[m]
    if x.size < 3 or np.var(x) == 0.0:
        return np.nan, np.nan
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    bias = float(np.mean(y - x))
    return slope, bias


def metrics_for_component(Btrue: np.ndarray, Brec: np.ndarray, comp: int, zidx: int) -> Dict[str, float]:
    true_full = Btrue[..., comp]
    rec_full = Brec[..., comp]
    true_slice = Btrue[:, :, zidx, comp]
    rec_slice = Brec[:, :, zidx, comp]
    full_slope, full_bias = slope_bias(true_full, rec_full)
    slice_slope, slice_bias = slope_bias(true_slice, rec_slice)
    return {
        "r_full": corrcoef_safe(true_full, rec_full),
        "nrmse_full": nrmse(true_full, rec_full),
        "slope_full": full_slope,
        "bias_full": full_bias,
        "r_slice": corrcoef_safe(true_slice, rec_slice),
        "nrmse_slice": nrmse(true_slice, rec_slice),
        "slope_slice": slice_slope,
        "bias_slice": slice_bias,
    }


def compute_all_metrics(Btrue: np.ndarray, Bdiv: np.ndarray, Bstd: np.ndarray, zidx: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {"divfree": {}, "standard": {}}
    for c, cname in enumerate(COMPONENTS):
        out["divfree"][cname] = metrics_for_component(Btrue, Bdiv, c, zidx)
        out["standard"][cname] = metrics_for_component(Btrue, Bstd, c, zidx)
    return out


def save_metrics_csv(outdir: Path, metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    path = outdir / "component_recovery_metrics.csv"
    fields = [
        "prior", "component", "r_full", "nrmse_full", "slope_full", "bias_full",
        "r_slice", "nrmse_slice", "slope_slice", "bias_slice",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for prior in ["divfree", "standard"]:
            for comp in COMPONENTS:
                row = {"prior": prior, "component": comp}
                row.update(metrics[prior][comp])
                writer.writerow(row)

    txt = outdir / "component_recovery_summary.txt"
    with txt.open("w") as f:
        f.write("Component-wise Cartesian vector-field recovery\n")
        f.write("==============================================\n")
        for comp in COMPONENTS:
            f.write(f"\n{comp}\n")
            for prior in ["divfree", "standard"]:
                m = metrics[prior][comp]
                f.write(
                    f"  {prior:8s}: r_full={m['r_full']:.6g}, nrmse_full={m['nrmse_full']:.6g}, "
                    f"r_slice={m['r_slice']:.6g}, nrmse_slice={m['nrmse_slice']:.6g}\n"
                )
            df = metrics["divfree"][comp]
            st = metrics["standard"][comp]
            f.write(
                f"  gain: delta_r_full={df['r_full'] - st['r_full']:+.6g}, "
                f"nrmse_reduction_full={st['nrmse_full'] - df['nrmse_full']:+.6g}\n"
            )
    print(f"Wrote {path}")
    print(f"Wrote {txt}")


def symmetric_limit(*arrays, q: float = 99.0, floor: float = 1e-12) -> float:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(vals), q)), floor)


def sample_indices(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def add_scatter_panel(ax, true_slice: np.ndarray, div_slice: np.ndarray, std_slice: np.ndarray, *, comp_label: str, metrics, seed: int, max_points: int):
    x = true_slice.ravel()
    yd = div_slice.ravel()
    ys = std_slice.ravel()
    idx = sample_indices(x.size, max_points=max_points, seed=seed)
    x = x[idx]
    yd = yd[idx]
    ys = ys[idx]
    lim = symmetric_limit(x, yd, ys, q=99.2)

    ax.scatter(x, yd, s=13, alpha=0.36, color=DIVFREE_COLOR, linewidths=0, label="div-free prior")
    ax.scatter(x, ys, s=13, alpha=0.36, color=STANDARD_COLOR, linewidths=0, label="standard vector prior")
    ax.plot([-lim, lim], [-lim, lim], "--", lw=1.2, color="0.25")
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=GRID_ALPHA)
    ax.set_xlabel(f"true {comp_label} on slice")
    ax.set_ylabel(f"reconstructed {comp_label}")
    ax.set_title(f"Slice scatter: {comp_label}")

    md = metrics["divfree"][comp_label.replace("$", "").replace("B_", "B").replace("{", "").replace("}", "")]
    ms = metrics["standard"][comp_label.replace("$", "").replace("B_", "B").replace("{", "").replace("}", "")]
    # The string conversion above is fragile for LaTeX labels, so remap explicitly below if needed.


def annotate_scatter(ax, md: Dict[str, float], ms: Dict[str, float]) -> None:
    txt = (
        f"div-free: r={md['r_slice']:.3f}, nRMSE={md['nrmse_slice']:.3f}\n"
        f"standard: r={ms['r_slice']:.3f}, nRMSE={ms['nrmse_slice']:.3f}"
    )
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.82", alpha=0.95),
    )


def plot_component_slice_scatter(
    outdir: Path,
    Btrue: np.ndarray,
    Bdiv: np.ndarray,
    Bstd: np.ndarray,
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    *,
    zidx: int,
    max_scatter_points: int,
) -> None:
    # Top row: Bx and By scatter. Lower rows: maps for Bx, By, Bz.
    fig = plt.figure(figsize=(17.2, 14.6), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=5,
        height_ratios=[1.15, 1.0, 1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    ax_sx = fig.add_subplot(gs[0, 0:2])
    ax_sy = fig.add_subplot(gs[0, 2:4])
    ax_note = fig.add_subplot(gs[0, 4])
    ax_note.axis("off")

    for ax, comp, label, seed in [(ax_sx, 0, r"$B_x$", 11), (ax_sy, 1, r"$B_y$", 13)]:
        x = Btrue[:, :, zidx, comp].ravel()
        yd = Bdiv[:, :, zidx, comp].ravel()
        ys = Bstd[:, :, zidx, comp].ravel()
        idx = sample_indices(x.size, max_points=max_scatter_points, seed=seed)
        lim = symmetric_limit(x[idx], yd[idx], ys[idx], q=99.2)
        ax.scatter(x[idx], yd[idx], s=14, alpha=0.38, color=DIVFREE_COLOR, linewidths=0, label="div-free prior")
        ax.scatter(x[idx], ys[idx], s=14, alpha=0.38, color=STANDARD_COLOR, linewidths=0, label="standard vector prior")
        ax.plot([-lim, lim], [-lim, lim], "--", lw=1.2, color="0.25")
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=GRID_ALPHA)
        ax.set_xlabel(f"true {label} on z-slice")
        ax.set_ylabel(f"reconstructed {label}")
        ax.set_title(f"Amplitude recovery on the central z-slice: {label}")
        annotate_scatter(ax, metrics["divfree"][COMPONENTS[comp]], metrics["standard"][COMPONENTS[comp]])

    handles, labels = ax_sx.get_legend_handles_labels()
    ax_note.legend(handles, labels, loc="upper left", frameon=False)
    ax_note.text(
        0.0, 0.55,
        f"Slice index: z={zidx}\n"
        "Top panels compare true and reconstructed\n"
        "component amplitudes on the same slice.\n\n"
        "Lower panels show the spatial structure\n"
        "and residuals component by component.",
        ha="left", va="top", fontsize=9.7,
    )

    axes = np.empty((3, 5), dtype=object)
    for row in range(3):
        for col in range(5):
            axes[row, col] = fig.add_subplot(gs[row + 1, col])

    col_titles = ["truth", "div-free reconstruction", "standard reconstruction", "div-free residual", "standard residual"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)

    for comp in range(3):
        truth = Btrue[:, :, zidx, comp]
        div = Bdiv[:, :, zidx, comp]
        std = Bstd[:, :, zidx, comp]
        res_div = div - truth
        res_std = std - truth
        v_field = symmetric_limit(truth, div, std, q=99.0)
        v_res = symmetric_limit(res_div, res_std, q=99.0)
        fields = [truth, div, std, res_div, res_std]
        vlims = [v_field, v_field, v_field, v_res, v_res]
        cmaps = ["coolwarm", "coolwarm", "coolwarm", "RdBu_r", "RdBu_r"]
        ims = []
        for col, (arr, vv, cmap) in enumerate(zip(fields, vlims, cmaps)):
            ax = axes[comp, col]
            im = ax.imshow(arr.T, origin="lower", vmin=-vv, vmax=vv, cmap=cmap, interpolation="nearest")
            ims.append(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if col == 0:
                ax.set_ylabel(COMP_LATEX[comp], rotation=0, labelpad=18, va="center", fontsize=14)
            if col == 1:
                md = metrics["divfree"][COMPONENTS[comp]]
                ax.text(
                    0.03, 0.97,
                    f"r={md['r_slice']:.3f}\nnRMSE={md['nrmse_slice']:.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.3,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.82", alpha=0.86),
                )
            if col == 2:
                ms = metrics["standard"][COMPONENTS[comp]]
                ax.text(
                    0.03, 0.97,
                    f"r={ms['r_slice']:.3f}\nnRMSE={ms['nrmse_slice']:.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.3,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.82", alpha=0.86),
                )

        # One field colorbar and one residual colorbar per component row.
        cbar1 = fig.colorbar(ims[2], ax=[axes[comp, 0], axes[comp, 1], axes[comp, 2]], shrink=0.74, pad=0.008)
        cbar1.set_label(COMP_LATEX[comp])
        cbar2 = fig.colorbar(ims[4], ax=[axes[comp, 3], axes[comp, 4]], shrink=0.74, pad=0.008)
        cbar2.set_label(r"residual")

    fig.suptitle(
        "Component-wise Cartesian magnetic-field recovery",
        y=1.012,
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(outdir / "05_cartesian_component_slice_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outdir / '05_cartesian_component_slice_scatter.png'}")


def plot_component_summary(outdir: Path, metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    x = np.arange(3)
    labels = COMPONENTS
    r_div = np.array([metrics["divfree"][c]["r_full"] for c in COMPONENTS])
    r_std = np.array([metrics["standard"][c]["r_full"] for c in COMPONENTS])
    e_div = np.array([metrics["divfree"][c]["nrmse_full"] for c in COMPONENTS])
    e_std = np.array([metrics["standard"][c]["nrmse_full"] for c in COMPONENTS])

    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    ax = axs[0]
    ax.plot(x, r_div, "o-", color=DIVFREE_COLOR, lw=1.8, ms=7, label="div-free prior")
    ax.plot(x, r_std, "s-", color=STANDARD_COLOR, lw=1.8, ms=6.5, label="standard vector prior")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("full-volume Pearson correlation")
    ax.set_ylim(max(-0.05, np.nanmin([r_div, r_std]) - 0.08), 1.02)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_title("Component correlation")

    ax = axs[1]
    ax.plot(x, e_div, "o-", color=DIVFREE_COLOR, lw=1.8, ms=7, label="div-free prior")
    ax.plot(x, e_std, "s-", color=STANDARD_COLOR, lw=1.8, ms=6.5, label="standard vector prior")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("full-volume normalized RMSE")
    ax.set_ylim(0.0, max(1.05, np.nanmax([e_div, e_std]) * 1.20))
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_title("Component amplitude error")

    ax = axs[2]
    delta_r = r_div - r_std
    delta_e = e_std - e_div
    width = 0.36
    ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
    ax.bar(x - width / 2, delta_r, width=width, color=DIVFREE_COLOR, alpha=0.85, label=r"$\Delta r$")
    ax.bar(x + width / 2, delta_e, width=width, color="#6a51a3", alpha=0.80, label="nRMSE reduction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("div-free improvement")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_title("Gain over standard prior")
    ax.text(
        0.02, 0.98,
        "positive means div-free is better",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.84", alpha=0.94),
    )

    handles, lab = axs[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    handles2, lab2 = axs[2].get_legend_handles_labels()
    axs[2].legend(handles2, lab2, loc="lower right", frameon=False)
    fig.suptitle("Full-volume component-wise vector-field recovery", y=1.09, fontsize=15, fontweight="bold")
    fig.savefig(outdir / "06_cartesian_component_summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outdir / '06_cartesian_component_summary.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divfree", type=Path, required=True, help="div-free reconstruction .npz")
    ap.add_argument("--standard", type=Path, required=True, help="standard-vector reconstruction .npz")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--slice-index", type=int, default=-1, help="z-slice index; -1 means central slice")
    ap.add_argument("--max-scatter-points", type=int, default=4000, help="subsample points in scatter panels")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    div = load_npz(args.divfree)
    std = load_npz(args.standard)

    geom = scalar_str(div, "geometry")
    if geom != "regular_cartesian":
        raise ValueError(f"This script expects a Cartesian reconstruction, but geometry={geom!r}.")

    Btrue = ensure_cartesian_grid(div["B_true"], name="B_true")
    Bdiv = ensure_cartesian_grid(div["B_est"], name="divfree B_est")
    Bstd = ensure_cartesian_grid(std["B_est"], name="standard B_est")
    if not (Btrue.shape == Bdiv.shape == Bstd.shape):
        raise ValueError(f"Shape mismatch: true={Btrue.shape}, div={Bdiv.shape}, std={Bstd.shape}")

    n = Btrue.shape[0]
    zidx = n // 2 if args.slice_index < 0 else int(args.slice_index)
    if not (0 <= zidx < n):
        raise ValueError(f"slice-index must be in [0,{n-1}], got {zidx}")

    metrics = compute_all_metrics(Btrue, Bdiv, Bstd, zidx)
    save_metrics_csv(args.outdir, metrics)
    plot_component_slice_scatter(args.outdir, Btrue, Bdiv, Bstd, metrics, zidx=zidx, max_scatter_points=args.max_scatter_points)
    plot_component_summary(args.outdir, metrics)

    print("\nHeadline full-volume gains, div-free minus standard:")
    for comp in COMPONENTS:
        md = metrics["divfree"][comp]
        ms = metrics["standard"][comp]
        print(
            f"  {comp}: delta r={md['r_full'] - ms['r_full']:+.4f}, "
            f"nRMSE reduction={ms['nrmse_full'] - md['nrmse_full']:+.4f}"
        )


if __name__ == "__main__":
    main()
