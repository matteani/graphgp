#!/usr/bin/env python3
"""
Local along-ray B_parallel recovery diagnostic for GraphGP + NIFTy RM reconstructions.

This compares two reconstructed magnetic fields against the known mock truth along the
full RM ray, not only at the pulsar endpoint.

For each pulsar ray x(s) = x_obs + s n_hat, it evaluates

    B_parallel(s) = B(x(s)) dot n_hat

for truth, div-free reconstruction and standard-vector reconstruction.

Outputs:
  05_along_ray_recovery_summary.png
  06_along_ray_gain_distribution.png
  07_along_ray_example_profiles.png
  along_ray_recovery_summary.txt

Example:
  python 04_along_ray_recovery.py \
    --divfree outputs/local_cartesian_recon_divfree_prior.npz \
    --standard outputs/local_cartesian_recon_standard_prior.npz \
    --outdir figures/local_cartesian_compare/along_ray \
    --n-ray-samples 128
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    from rm_common import load_npz
except Exception:
    def load_npz(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}


plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 240,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DIVFREE_COLOR = "#1f77b4"
STANDARD_COLOR = "#d95f02"
TRUTH_COLOR = "#222222"
GRID_ALPHA = 0.18


def scalar(x):
    arr = np.asarray(x)
    return arr.item() if arr.shape == () else arr


def scalar_str(d: Dict[str, np.ndarray], key: str, default: str = "unknown") -> str:
    return str(scalar(d.get(key, default)))


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-30)


def observer_from_file(d: Dict[str, np.ndarray]) -> np.ndarray:
    if "observer" in d:
        return np.asarray(d["observer"], dtype=float)
    if scalar_str(d, "geometry") == "regular_cartesian":
        return np.array([0.5, 0.5, 0.5], dtype=float)
    return np.zeros(3, dtype=float)


def get_ray_catalogue(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "pulsar_pos" not in d:
        raise KeyError("File does not contain pulsar_pos; cannot construct rays.")
    pos = np.asarray(d["pulsar_pos"], dtype=float)
    if "pulsar_dir" in d:
        dirs = normalize_rows(np.asarray(d["pulsar_dir"], dtype=float))
    else:
        obs = observer_from_file(d)
        dirs = normalize_rows(pos - obs[None, :])
    if "pulsar_dist" in d:
        dist = np.asarray(d["pulsar_dist"], dtype=float)
    else:
        obs = observer_from_file(d)
        dist = np.linalg.norm(pos - obs[None, :], axis=1)
    return pos, dirs, dist


def trilinear_interp_regular(Bgrid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Trilinear interpolation on a [0,1]^3 voxel-centred Cartesian grid."""
    n = Bgrid.shape[0]
    u = pts * n - 0.5
    i0_raw = np.floor(u).astype(int)
    t = u - i0_raw
    i0 = np.clip(i0_raw, 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    t = np.where((i0_raw < 0) | (i0_raw >= n - 1), 0.0, t)
    t = np.clip(t, 0.0, 1.0)

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    c000 = Bgrid[x0, y0, z0]
    c100 = Bgrid[x1, y0, z0]
    c010 = Bgrid[x0, y1, z0]
    c110 = Bgrid[x1, y1, z0]
    c001 = Bgrid[x0, y0, z1]
    c101 = Bgrid[x1, y0, z1]
    c011 = Bgrid[x0, y1, z1]
    c111 = Bgrid[x1, y1, z1]

    c00 = c000 * (1 - tx)[:, None] + c100 * tx[:, None]
    c10 = c010 * (1 - tx)[:, None] + c110 * tx[:, None]
    c01 = c001 * (1 - tx)[:, None] + c101 * tx[:, None]
    c11 = c011 * (1 - tx)[:, None] + c111 * tx[:, None]
    c0 = c00 * (1 - ty)[:, None] + c10 * ty[:, None]
    c1 = c01 * (1 - ty)[:, None] + c11 * ty[:, None]
    return c0 * (1 - tz)[:, None] + c1 * tz[:, None]


def inverse_distance_interp(points: np.ndarray, B: np.ndarray, pts: np.ndarray, *, k: int = 4) -> np.ndarray:
    """Simple inverse-distance interpolation for irregular HEALPix/log-r points."""
    tree = cKDTree(points)
    k = min(k, len(points))
    dist, idx = tree.query(pts, k=k)
    dist = np.asarray(dist)
    idx = np.asarray(idx)
    if k == 1:
        return B[idx]
    weights = 1.0 / np.maximum(dist, 1e-12) ** 2
    out = np.sum(B[idx] * weights[..., None], axis=1) / np.sum(weights, axis=1)[:, None]
    exact = dist[:, 0] < 1e-12
    if np.any(exact):
        out[exact] = B[idx[exact, 0]]
    return out


def make_interpolator(d: Dict[str, np.ndarray], B: np.ndarray):
    geom = scalar_str(d, "geometry")
    B = np.asarray(B, dtype=float)
    if geom == "regular_cartesian" and B.ndim == 4:
        return lambda pts: trilinear_interp_regular(B, pts)
    points = np.asarray(d["points"], dtype=float)
    Bflat = B.reshape((len(points), 3))
    return lambda pts: inverse_distance_interp(points, Bflat, pts, k=4)


def ray_sample_indices(n_rays: int, max_rays: int, seed: int) -> np.ndarray:
    if max_rays <= 0 or n_rays <= max_rays:
        return np.arange(n_rays)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rays, size=max_rays, replace=False))


def sample_along_rays(
    d: Dict[str, np.ndarray],
    B: np.ndarray,
    *,
    n_ray_samples: int,
    ray_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    obs = observer_from_file(d)
    _, dirs, dist = get_ray_catalogue(d)
    frac = (np.arange(n_ray_samples, dtype=float) + 0.5) / n_ray_samples
    ray_ids = np.repeat(ray_idx, n_ray_samples)
    frac_all = np.tile(frac, len(ray_idx))
    dirs_sel = dirs[ray_ids]
    dist_sel = dist[ray_ids]
    pts = obs[None, :] + dirs_sel * (dist_sel * frac_all)[:, None]
    interp = make_interpolator(d, B)
    Bpts = interp(pts)
    bpar = np.sum(Bpts * dirs_sel, axis=1)
    return {"ray_ids": ray_ids, "frac": frac_all, "pts": pts, "bpar": bpar}


def per_ray_metrics(true: np.ndarray, rec: np.ndarray, ray_ids: np.ndarray) -> Dict[str, np.ndarray]:
    unique = np.unique(ray_ids)
    sign_acc = np.full(len(unique), np.nan)
    corr = np.full(len(unique), np.nan)
    nrmse = np.full(len(unique), np.nan)
    mean_abs_true = np.full(len(unique), np.nan)
    for i, rid in enumerate(unique):
        m = ray_ids == rid
        t = true[m]
        r = rec[m]
        finite = np.isfinite(t) & np.isfinite(r)
        if finite.sum() == 0:
            continue
        t = t[finite]
        r = r[finite]
        sign_acc[i] = np.mean(np.sign(t) == np.sign(r))
        mean_abs_true[i] = np.mean(np.abs(t))
        denom = np.sqrt(np.mean(t ** 2))
        nrmse[i] = np.sqrt(np.mean((r - t) ** 2)) / max(denom, 1e-12)
        if len(t) > 2 and np.std(t) > 0 and np.std(r) > 0:
            corr[i] = np.corrcoef(t, r)[0, 1]
    return {"ray_idx": unique, "sign_acc": sign_acc, "corr": corr, "nrmse": nrmse, "mean_abs_true": mean_abs_true}


def mean_ci(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if len(x) == 1:
        return m, m, m
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, *, nboot: int = 5000, seed: int = 0) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    n = len(a)
    if n == 0:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    delta = float(np.mean(a - b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    boot = np.mean(a[idx] - b[idx], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"delta": delta, "lo": float(lo), "hi": float(hi), "n": int(n)}


def pick_representative_rays(metrics_div, metrics_std, *, n: int = 6) -> np.ndarray:
    ray_idx = metrics_div["ray_idx"]
    gain = metrics_div["sign_acc"] - metrics_std["sign_acc"]
    finite = np.isfinite(gain)
    if finite.sum() == 0:
        return ray_idx[:n]
    rays = ray_idx[finite]
    gain_f = gain[finite]
    signal = metrics_div["mean_abs_true"][finite]
    order_gain = np.argsort(gain_f)
    selected = []
    # strong wins for div-free
    for j in order_gain[::-1][: max(1, n // 3)]:
        selected.append(rays[j])
    # cases where standard wins or div-free does least well
    for j in order_gain[: max(1, n // 4)]:
        selected.append(rays[j])
    # high-signal rays for visually meaningful profiles
    for j in np.argsort(signal)[::-1]:
        selected.append(rays[j])
        if len(set(selected)) >= n:
            break
    out = []
    for r in selected:
        if r not in out:
            out.append(r)
        if len(out) == n:
            break
    return np.asarray(out, dtype=int)


def plot_summary(outdir: Path, div_metrics, std_metrics):
    labels = ["along-ray sign accuracy", "along-ray correlation", "normalized RMSE"]
    keys = ["sign_acc", "corr", "nrmse"]
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)
    for ax, key, title in zip(axs, keys, labels):
        div_mean, div_lo, div_hi = mean_ci(div_metrics[key])
        std_mean, std_lo, std_hi = mean_ci(std_metrics[key])
        vals = np.array([div_mean, std_mean])
        lo = np.array([div_lo, std_lo])
        hi = np.array([div_hi, std_hi])
        yerr = np.vstack([np.maximum(0, vals - lo), np.maximum(0, hi - vals)])
        ax.errorbar([0, 1], vals, yerr=yerr, fmt="none", capsize=5, lw=1.8, color="0.2")
        ax.scatter([0], [div_mean], s=75, color=DIVFREE_COLOR, zorder=3, label="div-free prior")
        ax.scatter([1], [std_mean], s=75, marker="s", color=STANDARD_COLOR, zorder=3, label="standard vector prior")
        for xx, yy in zip([0, 1], vals):
            ax.text(xx, yy, f" {yy:.3f}", va="center", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["div-free", "standard"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=GRID_ALPHA)
        if key in ("sign_acc", "corr"):
            ymin = max(-1.0 if key == "corr" else 0.0, np.nanmin(lo) - 0.08)
            ymax = min(1.02, np.nanmax(hi) + 0.08)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0.0, np.nanmax(hi) * 1.25)
    handles, lab = axs[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Local line-of-sight recovery, averaged per pulsar ray", y=1.10, fontsize=15)
    fig.savefig(outdir / "05_along_ray_recovery_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_gain_distribution(outdir: Path, div_metrics, std_metrics):
    gains = [
        (div_metrics["sign_acc"] - std_metrics["sign_acc"], "Sign-accuracy gain", "div-free minus standard"),
        (div_metrics["corr"] - std_metrics["corr"], "Correlation gain", "div-free minus standard"),
        (std_metrics["nrmse"] - div_metrics["nrmse"], "Error reduction", "standard RMSE minus div-free RMSE"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.0), constrained_layout=True)
    for ax, (gain, title, xlabel) in zip(axs, gains):
        gain = np.asarray(gain, dtype=float)
        gain = gain[np.isfinite(gain)]
        ax.axvline(0.0, color="0.35", ls="--", lw=1.0)
        ax.hist(gain, bins=28, alpha=0.82, color=DIVFREE_COLOR)
        if len(gain):
            ax.axvline(np.mean(gain), color="k", lw=1.4)
            ax.text(0.02, 0.96, f"mean = {np.mean(gain):+.3f}\npositive rays = {np.mean(gain > 0):.1%}", transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("number of rays")
        ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.suptitle("Does the div-free prior improve local recovery along individual rays?", y=1.04, fontsize=15)
    fig.savefig(outdir / "06_along_ray_gain_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_example_profiles(outdir: Path, samples_true, samples_div, samples_std, div_metrics, std_metrics, *, n_examples: int = 6):
    rays = pick_representative_rays(div_metrics, std_metrics, n=n_examples)
    ncols = 2
    nrows = int(math.ceil(len(rays) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(12.0, 3.1 * nrows), sharex=True, constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, rid in zip(axs, rays):
        m = samples_true["ray_ids"] == rid
        f = samples_true["frac"][m]
        order = np.argsort(f)
        f = f[order]
        t = samples_true["bpar"][m][order]
        d = samples_div["bpar"][m][order]
        s = samples_std["bpar"][m][order]
        ax.axhline(0.0, color="0.45", lw=0.9)
        ax.plot(f, t, color=TRUTH_COLOR, lw=2.0, label="truth")
        ax.plot(f, d, color=DIVFREE_COLOR, lw=1.7, label="div-free")
        ax.plot(f, s, color=STANDARD_COLOR, lw=1.7, label="standard")
        i_div = np.where(div_metrics["ray_idx"] == rid)[0][0]
        i_std = np.where(std_metrics["ray_idx"] == rid)[0][0]
        gain = float(div_metrics["sign_acc"][i_div] - std_metrics["sign_acc"][i_std])
        ax.set_title(f"ray {rid}: sign-accuracy gain {gain:+.2f}")
        ax.set_xlabel("fractional distance along ray")
        ax.set_ylabel(r"$B_\parallel(s)$")
        ax.grid(alpha=GRID_ALPHA)
    for ax in axs[len(rays):]:
        ax.axis("off")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(r"Local $B_\parallel(s)$ profiles along representative RM sightlines", y=1.06, fontsize=15)
    fig.savefig(outdir / "07_along_ray_example_profiles.png", bbox_inches="tight")
    plt.close(fig)


def save_summary(outdir: Path, div_metrics, std_metrics):
    txt = outdir / "along_ray_recovery_summary.txt"
    deltas = {
        "sign_accuracy_gain": paired_bootstrap_delta(div_metrics["sign_acc"], std_metrics["sign_acc"], seed=1),
        "correlation_gain": paired_bootstrap_delta(div_metrics["corr"], std_metrics["corr"], seed=2),
        "nrmse_reduction": paired_bootstrap_delta(std_metrics["nrmse"], div_metrics["nrmse"], seed=3),
    }
    with txt.open("w") as f:
        f.write("Local along-ray B_parallel recovery\n")
        f.write("===================================\n")
        for label, met in [("divfree", div_metrics), ("standard", std_metrics)]:
            f.write(f"\n{label}\n")
            for key in ["sign_acc", "corr", "nrmse"]:
                m, lo, hi = mean_ci(met[key])
                f.write(f"  {key}_mean: {m}\n")
                f.write(f"  {key}_low95: {lo}\n")
                f.write(f"  {key}_high95: {hi}\n")
            f.write(f"  n_rays: {int(np.isfinite(met['sign_acc']).sum())}\n")
        f.write("\nPaired div-free improvements\n")
        for name, d in deltas.items():
            f.write(f"  {name}: {d['delta']}  [{d['lo']}, {d['hi']}]  n={d['n']}\n")
    print(f"Wrote {txt}")
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divfree", type=Path, required=True)
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-ray-samples", type=int, default=128)
    ap.add_argument("--max-rays", type=int, default=0, help="subsample rays for speed; 0 means all")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    div = load_npz(args.divfree)
    std = load_npz(args.standard)

    if "B_true" not in div or "B_est" not in div or "B_est" not in std:
        raise KeyError("Need B_true and B_est arrays in reconstruction files.")

    _, _, dist = get_ray_catalogue(div)
    rays = ray_sample_indices(len(dist), args.max_rays, args.seed)

    s_true = sample_along_rays(div, np.asarray(div["B_true"]), n_ray_samples=args.n_ray_samples, ray_idx=rays)
    s_div = sample_along_rays(div, np.asarray(div["B_est"]), n_ray_samples=args.n_ray_samples, ray_idx=rays)
    s_std = sample_along_rays(div, np.asarray(std["B_est"]), n_ray_samples=args.n_ray_samples, ray_idx=rays)

    ray_ids = s_true["ray_ids"]
    div_metrics = per_ray_metrics(s_true["bpar"], s_div["bpar"], ray_ids)
    std_metrics = per_ray_metrics(s_true["bpar"], s_std["bpar"], ray_ids)

    deltas = save_summary(args.outdir, div_metrics, std_metrics)
    plot_summary(args.outdir, div_metrics, std_metrics)
    plot_gain_distribution(args.outdir, div_metrics, std_metrics)
    plot_example_profiles(args.outdir, s_true, s_div, s_std, div_metrics, std_metrics, n_examples=args.n_examples)

    print("\nMain figures:")
    print(f"  {args.outdir / '05_along_ray_recovery_summary.png'}")
    print(f"  {args.outdir / '06_along_ray_gain_distribution.png'}")
    print(f"  {args.outdir / '07_along_ray_example_profiles.png'}")
    print("\nHeadline:")
    print(f"  sign-accuracy gain = {deltas['sign_accuracy_gain']['delta']:+.4f} [{deltas['sign_accuracy_gain']['lo']:+.4f}, {deltas['sign_accuracy_gain']['hi']:+.4f}]")
    print(f"  correlation gain   = {deltas['correlation_gain']['delta']:+.4f} [{deltas['correlation_gain']['lo']:+.4f}, {deltas['correlation_gain']['hi']:+.4f}]")
    print(f"  RMSE reduction     = {deltas['nrmse_reduction']['delta']:+.4f} [{deltas['nrmse_reduction']['lo']:+.4f}, {deltas['nrmse_reduction']['hi']:+.4f}]")


if __name__ == "__main__":
    main()
