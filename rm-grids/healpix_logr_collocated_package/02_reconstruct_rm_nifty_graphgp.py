#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import jax.random as jr

from rm_common import (
    require_graphgp,
    import_jft,
    shape_with_dtype,
    make_initial_position,
    compose_likelihood,
    call_optimize_kl,
    extract_xi_from_samples,
    make_graphgp_covariance,
    canonical_prior_kind,
    response_apply_numpy,
    response_apply_jax,
    divergence_fd_cartesian,
    sign_accuracy_masked,
    load_npz,
    save_npz,
    json_safe,
)


def scalar_string(x):
    arr = np.asarray(x)
    return str(arr.item()) if arr.shape == () else str(arr)


def infer_prior_kind(data: dict, requested: str) -> str:
    if requested != "auto":
        return canonical_prior_kind(requested)
    # For prior comparison, default to div-free even if truth metadata is present.
    return "divfree_rbf"


def main() -> None:
    ap = argparse.ArgumentParser(description="NIFTy.re RM reconstruction with selectable GraphGP vector prior.")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prior-kind", type=str, default="auto", choices=["auto", "divfree_rbf", "standard_vector_rbf", "divfree", "standard", "nondivfree"])
    ap.add_argument("--n-vi-iterations", type=int, default=4)
    ap.add_argument("--n-vi-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--save-xi", action="store_true")
    args = ap.parse_args()

    d = load_npz(args.data)
    gp = require_graphgp()
    jft = import_jft()

    points = jnp.asarray(d["points"], dtype=jnp.float64)
    N = int(points.shape[0])
    geometry = scalar_string(d.get("geometry", "unknown"))
    graph_n0 = int(np.asarray(d["graph_n0"]).item())
    graph_k = int(np.asarray(d["graph_k"]).item())
    ell = float(np.asarray(d["ell"]).item())
    sigma2 = float(np.asarray(d["sigma2"]).item())
    periodic = bool(np.asarray(d["periodic"]).item())

    prior_kind = infer_prior_kind(d, args.prior_kind)
    cov_elem, prior_kind = make_graphgp_covariance(gp, prior_kind, ell=ell, sigma2=sigma2, periodic=periodic)

    print(
        f"[recon] geometry={geometry}, prior={prior_kind}, N={N}, n0={graph_n0}, k={graph_k}, "
        f"ndata={len(d['rm_obs'])}",
        flush=True,
    )
    graph = gp.build_graph(points, n0=graph_n0, k=graph_k)

    rows = jnp.asarray(d["response_rows"], dtype=jnp.int64)
    cols = jnp.asarray(d["response_cols"], dtype=jnp.int64)
    vals = jnp.asarray(d["response_vals"], dtype=jnp.float64)
    rm_obs = jnp.asarray(d["rm_obs"], dtype=jnp.float64)
    rm_noise = jnp.asarray(d["rm_noise"], dtype=jnp.float64)
    n_data = int(rm_obs.shape[0])

    def graphgp_prior_to_B(xi):
        return gp.generate_vector(graph, cov_elem, xi, cuda=False, fast_jit=True)

    def graphgp_prior_to_rm(xi):
        B = graphgp_prior_to_B(xi)
        return response_apply_jax(rows, cols, vals, B.reshape(-1), n_data)

    class GraphGPRMForwardModel(jft.Model):
        def __init__(self):
            domain = {"xi": shape_with_dtype(jft, (N, 3), jnp.float64)}
            super().__init__(domain=domain)

        def __call__(self, x):
            return graphgp_prior_to_rm(x["xi"])

    model = GraphGPRMForwardModel()

    def noise_cov_inv(x):
        return x / (rm_noise**2)

    likelihood = compose_likelihood(jft, rm_obs, noise_cov_inv, model)

    key = jr.PRNGKey(args.seed)
    key, sub = jr.split(key)
    pos_init = make_initial_position(jft, model.domain, sub)

    print("[recon] Running NIFTy.re optimize_kl", flush=True)
    samples, aux = call_optimize_kl(
        jft,
        likelihood,
        pos_init,
        key,
        n_vi_iterations=args.n_vi_iterations,
        n_vi_samples=args.n_vi_samples,
        verbosity=args.verbosity,
    )
    print(f"[recon] NIFTy output type: {type(samples)}", flush=True)

    xi_est = extract_xi_from_samples(samples)
    xi_est_jax = jnp.asarray(xi_est, dtype=jnp.float64)
    B_est_flat_vec = np.asarray(graphgp_prior_to_B(xi_est_jax))  # (N, 3)
    B_est_flat = B_est_flat_vec.reshape(-1)
    rm_pred = np.asarray(graphgp_prior_to_rm(xi_est_jax))

    endpoint_rows = np.asarray(d["endpoint_rows"], dtype=np.int64)
    endpoint_cols = np.asarray(d["endpoint_cols"], dtype=np.int64)
    endpoint_vals = np.asarray(d["endpoint_vals"], dtype=np.float64)
    bpar_est = response_apply_numpy(endpoint_rows, endpoint_cols, endpoint_vals, B_est_flat, len(d["bpar_true"]))
    acc_all, acc_masked = sign_accuracy_masked(bpar_est, d["bpar_true"])

    resid = (rm_pred - np.asarray(d["rm_obs"])) / np.asarray(d["rm_noise"])
    reduced_chi2 = float(np.mean(resid**2))
    B_rms = float(np.sqrt(np.mean(np.sum(B_est_flat_vec**2, axis=-1))))

    if geometry == "regular_cartesian":
        n = int(np.asarray(d["n"]).item())
        B_est = B_est_flat_vec.reshape((n, n, n, 3))
        div = divergence_fd_cartesian(B_est)
        div_rms = float(np.sqrt(np.mean(div**2)))
        eps_rms = float(ell * div_rms / (B_rms + 1e-12))
    elif geometry == "healpix_logr":
        n_radial = int(np.asarray(d["n_radial"]).item())
        npix = int(np.asarray(d["npix"]).item())
        B_est = B_est_flat_vec.reshape((n_radial, npix, 3))
        div_rms = np.nan
        eps_rms = np.nan
    else:
        B_est = B_est_flat_vec
        div_rms = np.nan
        eps_rms = np.nan

    out_data = dict(d)
    out_data.update(
        B_est=B_est,
        rm_pred=rm_pred,
        bpar_est=bpar_est,
        residual_sigma=resid,
        reduced_chi2=np.asarray(reduced_chi2),
        sign_accuracy=np.asarray(acc_all),
        sign_accuracy_masked=np.asarray(acc_masked),
        B_est_rms=np.asarray(B_rms),
        div_est_rms=np.asarray(div_rms),
        epsilon_est_rms=np.asarray(eps_rms),
        reconstruction_prior_kind=np.asarray(prior_kind),
        nifty_method=np.asarray("nifty.re.optimize_kl"),
        nifty_config_json=np.asarray(json.dumps(json_safe(vars(args)))),
    )
    if args.save_xi:
        out_data["xi_est"] = xi_est

    save_npz(args.out, **out_data)
    print(f"[recon] Saved reconstruction: {args.out}", flush=True)
    print(f"[recon] prior={prior_kind}", flush=True)
    print(f"[recon] reduced chi2={reduced_chi2:.4g}", flush=True)
    print(f"[recon] endpoint B_parallel sign accuracy={acc_all:.4f} masked={acc_masked:.4f}", flush=True)
    print(f"[recon] ell*div_rms/B_rms={eps_rms:.4g}", flush=True)


if __name__ == "__main__":
    main()
