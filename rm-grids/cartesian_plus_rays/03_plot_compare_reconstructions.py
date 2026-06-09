#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rm_common import load_npz


plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "figure.dpi": 150,
    "savefig.dpi": 240,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DIVFREE_COLOR = "#1f77b4"
STANDARD_COLOR = "#d95f02"
WRONG_COLOR = "#e66101"
CORRECT_COLOR = "#3182bd"
GRID_ALPHA = 0.18


def scalar(x):
    a = np.asarray(x)
    return a.item() if a.shape == () else a


def scalar_str(d, key, default="unknown") -> str:
    return str(scalar(d.get(key, default)))


def finite_arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_stats(b_est, b_true, *, q: float = 0.05) -> Dict[str, float]:
    b_est = finite_arr(b_est)
    b_true = finite_arr(b_true)
    finite = np.isfinite(b_est) & np.isfinite(b_true)
    correct = np.sign(b_est[finite]) == np.sign(b_true[finite])
    n = int(correct.size)
    k = int(correct.sum())
    lo, hi = wilson_interval(k, n)

    abs_true = np.abs(b_true[finite])
    thresh = float(np.quantile(abs_true, q)) if abs_true.size else np.nan
    mask = abs_true > thresh
    correct_m = correct[mask]
    n_m = int(correct_m.size)
    k_m = int(correct_m.sum())
    lo_m, hi_m = wilson_interval(k_m, n_m)
    return {
        "k": k,
        "n": n,
        "acc": k / n if n else np.nan,
        "lo": lo,
        "hi": hi,
        "k_masked": k_m,
        "n_masked": n_m,
        "acc_masked": k_m / n_m if n_m else np.nan,
        "lo_masked": lo_m,
        "hi_masked": hi_m,
        "mask_threshold": thresh,
    }


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(m + 1)]
    mx = max(logs)
    tail = math.exp(mx) * sum(math.exp(v - mx) for v in logs)
    return float(min(1.0, 2.0 * tail))


def paired_counts(div, std) -> Dict[str, int | float]:
    btrue = finite_arr(div["bpar_true"])
    bdiv = finite_arr(div["bpar_est"])
    bstd = finite_arr(std["bpar_est"])
    finite = np.isfinite(btrue) & np.isfinite(bdiv) & np.isfinite(bstd)
    c_div = np.sign(bdiv[finite]) == np.sign(btrue[finite])
    c_std = np.sign(bstd[finite]) == np.sign(btrue[finite])
    both = int(np.sum(c_div & c_std))
    div_only = int(np.sum(c_div & ~c_std))
    std_only = int(np.sum(~c_div & c_std))
    neither = int(np.sum(~c_div & ~c_std))
    return {
        "both_correct": both,
        "divfree_only": div_only,
        "standard_only": std_only,
        "both_wrong": neither,
        "mcnemar_p_value": mcnemar_exact_pvalue(div_only, std_only),
    }


def bootstrap_accuracy_difference(b_true, b_div, b_std, mask=None, *, nboot: int = 4000, seed: int = 0):
    b_true = finite_arr(b_true)
    b_div = finite_arr(b_div)
    b_std = finite_arr(b_std)
    finite = np.isfinite(b_true) & np.isfinite(b_div) & np.isfinite(b_std)
    if mask is not None:
        finite &= mask
    idx = np.flatnonzero(finite)
    n = int(idx.size)
    if n == 0:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    div_correct = (np.sign(b_div[idx]) == np.sign(b_true[idx])).astype(np.float64)
    std_correct = (np.sign(b_std[idx]) == np.sign(b_true[idx])).astype(np.float64)
    delta = float(div_correct.mean() - std_correct.mean())
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(nboot, n))
    boot = div_correct[boot_idx].mean(axis=1) - std_correct[boot_idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"delta": delta, "lo": float(lo), "hi": float(hi), "n": n}


def metric_row(label, d, stats):
    return {
        "label": label,
        "geometry": scalar_str(d, "geometry"),
        "truth_prior_kind": scalar_str(d, "truth_prior_kind"),
        "reconstruction_prior_kind": scalar_str(d, "reconstruction_prior_kind", label),
        "n_data": int(len(d["rm_obs"])),
        "n_points": int(len(d["points"])),
        "reduced_chi2": float(scalar(d.get("reduced_chi2", np.nan))),
        "sign_accuracy": float(stats["acc"]),
        "sign_accuracy_low": float(stats["lo"]),
        "sign_accuracy_high": float(stats["hi"]),
        "sign_accuracy_masked": float(stats["acc_masked"]),
        "sign_accuracy_masked_low": float(stats["lo_masked"]),
        "sign_accuracy_masked_high": float(stats["hi_masked"]),
        "sign_count": int(stats["k"]),
        "sign_total": int(stats["n"]),
        "masked_sign_count": int(stats["k_masked"]),
        "masked_sign_total": int(stats["n_masked"]),
        "mask_abs_bparallel_threshold": float(stats["mask_threshold"]),
        "epsilon_est_rms": float(scalar(d.get("epsilon_est_rms", np.nan))),
        "B_est_rms": float(scalar(d.get("B_est_rms", np.nan))),
    }


def write_summary(outdir: Path, rows, paired):
    keys = list(rows[0].keys())
    csv = outdir / "prior_comparison_summary.csv"
    with csv.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    txt = outdir / "prior_comparison_summary.txt"
    with txt.open("w") as f:
        f.write("GraphGP pulsar-RM prior comparison\n")
        f.write("==================================\n")
        for r in rows:
            f.write(f"\n{r['label']}\n")
            for k in keys:
                if k != "label":
                    f.write(f"  {k}: {r[k]}\n")
        f.write("\nDifferences: divfree - standard\n")
        f.write(f"  sign_accuracy_delta: {rows[0]['sign_accuracy'] - rows[1]['sign_accuracy']:.6g}\n")
        f.write(f"  masked_sign_accuracy_delta: {rows[0]['sign_accuracy_masked'] - rows[1]['sign_accuracy_masked']:.6g}\n")
        f.write(f"  reduced_chi2_delta: {rows[0]['reduced_chi2'] - rows[1]['reduced_chi2']:.6g}\n")
        f.write("\nPaired endpoint sign table\n")
        for k, v in paired.items():
            f.write(f"  {k}: {v}\n")
    return csv, txt


def _set_3d_equal(ax, pts):
    pts = np.asarray(pts)
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    cen = 0.5 * (mins + maxs)
    rad = 0.5 * np.nanmax(maxs - mins)
    rad = max(rad, 1e-6)
    ax.set_xlim(cen[0] - rad, cen[0] + rad)
    ax.set_ylim(cen[1] - rad, cen[1] + rad)
    ax.set_zlim(cen[2] - rad, cen[2] + rad)
    ax.set_box_aspect((1, 1, 1))


def _draw_cube(ax, lo: float, hi: float, *, color: str = "0.45", alpha: float = 0.35, lw: float = 0.9):
    corners = np.array([
        [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
        [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi],
    ], dtype=float)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color=color, lw=lw, alpha=alpha)
    return corners


def plot_pulsar_geometry(outdir: Path, div):
    geometry = scalar_str(div, "geometry")
    pos = finite_arr(div["pulsar_pos"])
    pdir = finite_arr(div["pulsar_dir"])
    dist = finite_arr(div["pulsar_dist"])
    if geometry == "regular_cartesian":
        observer = finite_arr(div.get("observer", np.array([0.5, 0.5, 0.5])))
        shell = np.asarray(div.get("shell_id", np.zeros(len(pos))), dtype=int)
        box_lo, box_hi = 0.0, 1.0
    else:
        observer = np.zeros(3, dtype=np.float64)
        shell = np.asarray(div.get("pulsar_shell_id", np.zeros(len(pos))), dtype=int)
        half = float(np.nanmax(np.abs(pos)))
        box_lo, box_hi = -half, half

    n = len(pos)
    rng = np.random.default_rng(0)
    max_lines = min(220, n)
    if n > max_lines:
        idx = rng.choice(n, size=max_lines, replace=False)
        idx = idx[np.argsort(dist[idx])]
    else:
        idx = np.arange(n)

    fig = plt.figure(figsize=(12.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    axr = fig.add_subplot(gs[0, 1])

    dist_norm = (dist - np.nanmin(dist)) / (np.nanmax(dist) - np.nanmin(dist) + 1e-12)
    dist_colors = plt.get_cmap("viridis")(dist_norm)

    for ii in idx:
        ax3.plot(
            [observer[0], pos[ii, 0]], [observer[1], pos[ii, 1]], [observer[2], pos[ii, 2]],
            color="0.55", alpha=0.18, lw=0.65,
        )
    sc = ax3.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2], c=dist[idx], cmap="viridis", s=14, alpha=0.95, depthshade=False)
    ax3.scatter([observer[0]], [observer[1]], [observer[2]], marker="*", s=160, c="k", depthshade=False)
    corners = _draw_cube(ax3, box_lo, box_hi)
    if geometry == "regular_cartesian":
        pts_for_equal = np.vstack([corners, pos, observer[None, :]])
    else:
        pts_for_equal = np.vstack([corners, pos, observer[None, :]])
    _set_3d_equal(ax3, pts_for_equal)
    ax3.set_title("RM line-of-sight rays inside reconstruction domain")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=22, azim=-58)
    cbar = fig.colorbar(sc, ax=ax3, fraction=0.032, pad=0.06)
    cbar.set_label("distance from observer")

    shell_unique = np.unique(shell)
    jitter = np.linspace(-0.15, 0.15, max(3, len(dist)))
    for s in shell_unique:
        m = shell == s
        jj = np.resize(jitter, m.sum())
        axr.scatter(dist[m], np.full(m.sum(), s, dtype=float) + jj, c=dist_colors[m], s=18, alpha=0.85, edgecolors="none")
        med = float(np.median(dist[m]))
        axr.axvline(med, color="0.55", ls="--", lw=0.8, alpha=0.4)
        axr.text(med, s + 0.23, f"{med:.2f}", ha="center", va="bottom", fontsize=8, color="0.35")
    axr.set_xlabel("distance from observer")
    axr.set_ylabel("radial shell index")
    axr.set_title("Observer-centric radial placement of pulsars")
    axr.set_yticks(shell_unique)
    axr.grid(alpha=GRID_ALPHA)

    fig.savefig(outdir / "00_pulsar_geometry_and_sightlines.png", bbox_inches="tight")
    plt.close(fig)


def plot_summary(outdir: Path, rows, div, std, paired):
    stats_div = rows[0]
    stats_std = rows[1]
    btrue = finite_arr(div["bpar_true"])
    bdiv = finite_arr(div["bpar_est"])
    bstd = finite_arr(std["bpar_est"])
    finite = np.isfinite(btrue) & np.isfinite(bdiv) & np.isfinite(bstd)
    abs_true = np.abs(btrue[finite])
    thresh = float(stats_div["mask_abs_bparallel_threshold"])
    strong_mask = np.zeros_like(btrue, dtype=bool)
    strong_mask[finite] = abs_true > thresh

    delta_all = bootstrap_accuracy_difference(btrue, bdiv, bstd, None, seed=2)
    delta_strong = bootstrap_accuracy_difference(btrue, bdiv, bstd, strong_mask, seed=3)

    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.0, 0.85]})
    ax1, ax2, ax3 = axs

    x = np.array([0, 1])
    xlabels = ["all endpoints", "excluding weakest\n$|B_\\parallel|$"]
    offset = 0.08
    div_acc = np.array([stats_div["sign_accuracy"], stats_div["sign_accuracy_masked"]])
    div_lo = np.array([stats_div["sign_accuracy_low"], stats_div["sign_accuracy_masked_low"]])
    div_hi = np.array([stats_div["sign_accuracy_high"], stats_div["sign_accuracy_masked_high"]])
    std_acc = np.array([stats_std["sign_accuracy"], stats_std["sign_accuracy_masked"]])
    std_lo = np.array([stats_std["sign_accuracy_low"], stats_std["sign_accuracy_masked_low"]])
    std_hi = np.array([stats_std["sign_accuracy_high"], stats_std["sign_accuracy_masked_high"]])
    ax1.errorbar(x - offset, div_acc, yerr=np.vstack([np.maximum(0.0, div_acc - div_lo), np.maximum(0.0, div_hi - div_acc)]), fmt="o", color=DIVFREE_COLOR, ms=6.5, lw=1.8, capsize=4, label="div-free prior")
    ax1.plot(x - offset, div_acc, color=DIVFREE_COLOR, lw=1.4, alpha=0.8)
    ax1.errorbar(x + offset, std_acc, yerr=np.vstack([np.maximum(0.0, std_acc - std_lo), np.maximum(0.0, std_hi - std_acc)]), fmt="s", color=STANDARD_COLOR, ms=6.2, lw=1.8, capsize=4, label="standard vector prior")
    ax1.plot(x + offset, std_acc, color=STANDARD_COLOR, lw=1.4, alpha=0.8)
    for xx, yy in zip(x - offset, div_acc):
        ax1.text(xx, yy + 0.008, f"{yy:.3f}", ha="center", va="bottom", fontsize=9, color=DIVFREE_COLOR)
    for xx, yy in zip(x + offset, std_acc):
        ax1.text(xx, yy + 0.008, f"{yy:.3f}", ha="center", va="bottom", fontsize=9, color=STANDARD_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel("endpoint sign accuracy")
    ax1.set_ylim(0.78, 1.02)
    ax1.set_title("Endpoint magnetic-field sign recovery")
    ax1.grid(axis="y", alpha=GRID_ALPHA)
    ax1.text(x[0], 0.785, f"n={stats_div['sign_total']}", ha="center", va="bottom", fontsize=8.5, color="0.35")
    ax1.text(x[1], 0.785, f"n={stats_div['masked_sign_total']}", ha="center", va="bottom", fontsize=8.5, color="0.35")

    gain_x = np.array([0, 1])
    gains = np.array([delta_all["delta"], delta_strong["delta"]])
    gain_lo = np.array([delta_all["lo"], delta_strong["lo"]])
    gain_hi = np.array([delta_all["hi"], delta_strong["hi"]])
    ax2.axhline(0.0, color="0.45", lw=1.0, ls="--")
    ax2.errorbar(gain_x, gains, yerr=np.vstack([np.maximum(0.0, gains - gain_lo), np.maximum(0.0, gain_hi - gains)]), fmt="o", color=DIVFREE_COLOR, ms=7, lw=1.8, capsize=4)
    ax2.plot(gain_x, gains, color=DIVFREE_COLOR, lw=1.3)
    for xx, yy in zip(gain_x, gains):
        ax2.text(xx, yy + 0.0035 * np.sign(yy if yy != 0 else 1), f"{yy:+.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(gain_x)
    ax2.set_xticklabels(["all endpoints", "stronger-signal\nsubset"])
    ax2.set_ylabel(r"accuracy gain: div-free $-$ standard")
    lo = min(-0.02, float(np.nanmin(gain_lo)) - 0.01)
    hi = max(0.06, float(np.nanmax(gain_hi)) + 0.01)
    ax2.set_ylim(lo, hi)
    ax2.set_title("How much the div-free prior helps")
    ax2.grid(axis="y", alpha=GRID_ALPHA)
    ax2.text(
        0.03, 0.97,
        f"discordant endpoints: {paired['divfree_only']} vs {paired['standard_only']}\nexact McNemar p = {paired['mcnemar_p_value']:.3g}",
        transform=ax2.transAxes, ha="left", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.82", alpha=0.95),
    )

    chi = np.array([stats_div["reduced_chi2"], stats_std["reduced_chi2"]])
    ax3.axhline(1.0, ls="--", lw=1.0, color="0.4")
    ax3.vlines([0, 1], 0, chi, colors=[DIVFREE_COLOR, STANDARD_COLOR], lw=2.0, alpha=0.85)
    ax3.scatter([0, 1], chi, c=[DIVFREE_COLOR, STANDARD_COLOR], s=55, zorder=3)
    for xx, yy in zip([0, 1], chi):
        ax3.text(xx, yy + 0.03, f"{yy:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["div-free", "standard"])
    ax3.set_ylabel(r"reduced $\chi^2$")
    ax3.set_ylim(0, max(1.15, float(np.nanmax(chi)) * 1.25))
    ax3.set_title("RM data-space fit")
    ax3.grid(axis="y", alpha=GRID_ALPHA)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Same mock RM data reconstructed with two GraphGP priors", y=1.08, fontsize=15)
    fig.savefig(outdir / "01_prior_comparison_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_scatter(outdir: Path, div, std):
    btrue = finite_arr(div["bpar_true"])
    bdiv = finite_arr(div["bpar_est"])
    bstd = finite_arr(std["bpar_est"])
    lim = float(np.nanpercentile(np.abs(np.concatenate([btrue, bdiv, bstd])), 99.2))
    lim = max(lim, 1e-6)
    stats_div = sign_stats(bdiv, btrue)
    stats_std = sign_stats(bstd, btrue)
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True, sharey=True, constrained_layout=True)
    legend_handles = None
    legend_labels = None
    for ax, brec, title, stats in [
        (axs[0], bdiv, "div-free prior", stats_div),
        (axs[1], bstd, "standard vector prior", stats_std),
    ]:
        finite = np.isfinite(brec) & np.isfinite(btrue)
        correct = np.sign(brec[finite]) == np.sign(btrue[finite])
        x = btrue[finite]
        y = brec[finite]
        h1 = ax.scatter(x[correct], y[correct], s=14, alpha=0.35, color=CORRECT_COLOR, label="correct sign")
        h2 = ax.scatter(x[~correct], y[~correct], s=32, alpha=0.90, marker="x", color=WRONG_COLOR, label="wrong sign")
        legend_handles = [h1, h2]
        legend_labels = ["correct sign", "wrong sign"]
        ax.axhline(0, lw=1, color="0.45")
        ax.axvline(0, lw=1, color="0.45")
        ax.plot([-lim, lim], [-lim, lim], "--", lw=1.1, color="0.25")
        ax.set_title(f"{title}: sign acc. {stats['acc']:.3f}")
        ax.set_xlabel(r"true endpoint $B_\parallel$")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=GRID_ALPHA)
    axs[0].set_ylabel(r"reconstructed endpoint $B_\parallel$")
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.18, 0.98), frameon=False, ncol=2)
    fig.savefig(outdir / "02_endpoint_sign_scatter.png", bbox_inches="tight")
    plt.close(fig)


def plot_rm_fit(outdir: Path, div, std):
    rm_obs = finite_arr(div["rm_obs"])
    rm_err = finite_arr(div["rm_noise"])
    lim = float(np.nanpercentile(np.abs(np.concatenate([rm_obs, div["rm_pred"], std["rm_pred"]])), 99.2))
    lim = max(lim, 1e-6)
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, d, title in [
        (axs[0], div, f"div-free prior: $\\chi^2_\\nu={float(scalar(div['reduced_chi2'])):.3f}$"),
        (axs[1], std, f"standard vector prior: $\\chi^2_\\nu={float(scalar(std['reduced_chi2'])):.3f}$"),
    ]:
        ax.errorbar(rm_obs, d["rm_pred"], xerr=rm_err, fmt=".", ms=3.2, alpha=0.32, elinewidth=0.42, color=CORRECT_COLOR)
        ax.plot([-lim, lim], [-lim, lim], "--", lw=1.1, color="0.25")
        ax.set_title(title)
        ax.set_xlabel("observed RM")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=GRID_ALPHA)
    axs[0].set_ylabel("predicted RM")
    fig.savefig(outdir / "03_rm_fit.png", bbox_inches="tight")
    plt.close(fig)


def plot_cartesian_slices(outdir: Path, div, std):
    if scalar_str(div, "geometry") != "regular_cartesian":
        return
    Btrue = finite_arr(div["B_true"])
    Bdiv = finite_arr(div["B_est"])
    Bstd = finite_arr(std["B_est"])
    n = Btrue.shape[0]
    z = n // 2
    panels = [
        (Btrue[:, :, z, 2], "truth $B_z$"),
        (Bdiv[:, :, z, 2], "div-free prior"),
        (Bstd[:, :, z, 2], "standard prior"),
        (Bdiv[:, :, z, 2] - Bstd[:, :, z, 2], "div-free minus standard"),
    ]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([p[0].ravel() for p in panels[:3]])), 99))
    fig, axs = plt.subplots(1, 4, figsize=(14.0, 3.7), constrained_layout=True)
    for ax, (arr, title) in zip(axs, panels):
        v = float(np.nanpercentile(np.abs(arr), 99)) if "minus" in title else vmax
        im = ax.imshow(arr.T, origin="lower", vmin=-v, vmax=v, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Central $z$-slice comparison on the regular Cartesian grid", y=1.02)
    fig.savefig(outdir / "04_cartesian_Bz_slice.png", bbox_inches="tight")
    plt.close(fig)


def plot_healpix_sky(outdir: Path, div, std):
    if scalar_str(div, "geometry") != "healpix_logr":
        return
    phi = np.degrees(finite_arr(div["phi_pix"]))
    theta = np.degrees(finite_arr(div["theta_pix"]))
    pix = np.asarray(div["pulsar_pixel_id"], dtype=int)
    shell = np.asarray(div["pulsar_shell_id"], dtype=int)
    outer = shell == shell.max()
    if outer.sum() == 0:
        outer = np.ones_like(shell, dtype=bool)
    btrue = finite_arr(div["bpar_true"])[outer]
    bdiv = finite_arr(div["bpar_est"])[outer]
    bstd = finite_arr(std["bpar_est"])[outer]
    pix_outer = pix[outer]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([btrue, bdiv, bstd])), 98.5))
    vmax = max(vmax, 1e-6)
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.2), sharex=True, sharey=True, constrained_layout=True)
    titles = ["truth endpoint $B_\\parallel$", "div-free prior", "standard prior"]
    for ax, values, title in zip(axs, [btrue, bdiv, bstd], titles):
        sc = ax.scatter(phi[pix_outer], theta[pix_outer], c=values, s=20, vmin=-vmax, vmax=vmax, cmap="coolwarm", edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel(r"$\phi$ [deg]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_ylim(180, 0)
        ax.grid(alpha=0.15)
    cbar = fig.colorbar(sc, ax=axs, shrink=0.88, pad=0.02)
    cbar.set_label(r"endpoint $B_\parallel$")
    fig.suptitle("Outer-shell line-of-sight field on the HEALPix sky", y=1.03)
    fig.savefig(outdir / "04_healpix_outer_shell_sky.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divfree", type=Path, required=True)
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    div = load_npz(args.divfree)
    std = load_npz(args.standard)

    stats_div = sign_stats(div["bpar_est"], div["bpar_true"])
    stats_std = sign_stats(std["bpar_est"], std["bpar_true"])
    rows = [metric_row("divfree_rbf", div, stats_div), metric_row("standard_vector_rbf", std, stats_std)]
    paired = paired_counts(div, std)

    csv, txt = write_summary(args.outdir, rows, paired)
    plot_pulsar_geometry(args.outdir, div)
    plot_summary(args.outdir, rows, div, std, paired)
    plot_endpoint_scatter(args.outdir, div, std)
    plot_rm_fit(args.outdir, div, std)
    plot_cartesian_slices(args.outdir, div, std)
    plot_healpix_sky(args.outdir, div, std)

    print(f"Wrote {txt}")
    print(f"Wrote {csv}")
    print(f"sign accuracy divfree={rows[0]['sign_accuracy']:.4f}, standard={rows[1]['sign_accuracy']:.4f}, "
          f"delta={rows[0]['sign_accuracy'] - rows[1]['sign_accuracy']:+.4f}")
    print(f"paired counts: divfree_only={paired['divfree_only']}, standard_only={paired['standard_only']}, "
          f"mcnemar_p={paired['mcnemar_p_value']:.4g}")


if __name__ == "__main__":
    main()
