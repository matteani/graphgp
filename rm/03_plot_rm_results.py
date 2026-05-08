#!/usr/bin/env python3
"""Plot GraphGP mock RM data and NIFTy reconstruction outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rm_graphgp_nifty_utils import load_npz


def imshow_component(ax, arr, title, comp=2):
    n = arr.shape[0]
    zmid = n // 2
    im = ax.imshow(arr[:, :, zmid, comp].T, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_geometry(d, outdir):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    obs = d["observer"]
    pp = d["pulsar_pos"]
    ax.scatter([obs[0]], [obs[1]], [obs[2]], s=90, marker="*", label="observer")
    ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], s=18, label="mock pulsars")
    for p in pp[: min(90, len(pp))]:
        ax.plot([obs[0], p[0]], [obs[1], p[1]], [obs[2], p[2]], lw=0.5, alpha=0.35)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Mock pulsar sightlines")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "01_mock_pulsar_geometry.png", dpi=180)
    plt.close(fig)


def plot_slices(d, outdir):
    B_true = d["B_true"]
    B_est = d.get("B_est", None)
    if B_est is None:
        B_est = d.get("B_map", None)
    if B_est is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.8, 5))
        imshow_component(ax, B_true, "truth $B_z$ slice", comp=2)
        fig.tight_layout()
        fig.savefig(outdir / "02_truth_slice.png", dpi=180)
        plt.close(fig)
        return

    vmax = np.nanpercentile(np.abs(np.concatenate([B_true[..., 2].ravel(), B_est[..., 2].ravel()])), 99)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    n = B_true.shape[0]
    zmid = n // 2
    panels = [
        (B_true[:, :, zmid, 2], "truth $B_z$"),
        (B_est[:, :, zmid, 2], "NIFTy/GraphGP estimate $B_z$"),
        (B_est[:, :, zmid, 2] - B_true[:, :, zmid, 2], "estimate - truth"),
    ]
    for ax, (img, title) in zip(axs, panels):
        im = ax.imshow(img.T, origin="lower", extent=[0, 1, 0, 1], vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Central z-slice, z-index={zmid}")
    fig.tight_layout()
    fig.savefig(outdir / "02_reconstruction_slices.png", dpi=180)
    plt.close(fig)


def plot_rm_fit(d, outdir):
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4))
    rm_true = d["rm_true"]
    rm_obs = d["rm_obs"]
    rm_noise = d["rm_noise"]
    if "rm_pred" in d:
        y = d["rm_pred"]
        label = "NIFTy prediction"
    else:
        y = rm_obs
        label = "observed mock RM"
    ax.errorbar(rm_true, rm_obs, yerr=rm_noise, fmt=".", alpha=0.45, label="mock observed")
    ax.scatter(rm_true, y, s=18, label=label)
    lo = min(np.min(rm_true), np.min(rm_obs), np.min(y))
    hi = max(np.max(rm_true), np.max(rm_obs), np.max(y))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true noiseless RM")
    ax.set_ylabel("observed / predicted RM")
    ax.set_title("RM data-space fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "03_rm_fit.png", dpi=180)
    plt.close(fig)


def plot_sign_recovery(d, outdir):
    if "bpar_est" not in d:
        return
    btrue = d["bpar_true"]
    best = d["bpar_est"]
    acc = float(np.mean(np.sign(btrue) == np.sign(best)))
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4))
    ax.scatter(btrue, best, s=22, alpha=0.75)
    lo = min(np.min(btrue), np.min(best))
    hi = max(np.max(btrue), np.max(best))
    ax.axhline(0, color="0.5", lw=1)
    ax.axvline(0, color="0.5", lw=1)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true endpoint $B_\\parallel$")
    ax.set_ylabel("estimated endpoint $B_\\parallel$")
    ax.set_title(f"Endpoint line-of-sight sign recovery: {acc:.3f}")
    fig.tight_layout()
    fig.savefig(outdir / "04_endpoint_sign_recovery.png", dpi=180)
    plt.close(fig)


def plot_metrics(d, outdir):
    labels, values = [], []
    for key, label in [
        ("reduced_chi2", "RM reduced chi2"),
        ("sign_accuracy", "endpoint sign accuracy"),
        ("epsilon_rms", "truth epsilon rms"),
        ("epsilon_est_rms", "estimate epsilon rms"),
    ]:
        if key in d:
            labels.append(label)
            values.append(float(d[key]))
    if not values:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4))
    x = np.arange(len(values))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Diagnostics")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "05_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("rm_nifty_reconstruction.npz"))
    ap.add_argument("--outdir", type=Path, default=Path("rm_figures"))
    args = ap.parse_args()

    d = load_npz(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_geometry(d, args.outdir)
    plot_slices(d, args.outdir)
    plot_rm_fit(d, args.outdir)
    plot_sign_recovery(d, args.outdir)
    plot_metrics(d, args.outdir)
    print(f"Saved figures to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
