#!/usr/bin/env python3
"""NIFTy.re reconstruction of mock RM data with a GraphGP prior model.

Latent parameter:
    xi ~ N(0, I), shape (N_voxels, 3)

GraphGP prior model:
    xi -> B = graphgp.generate_vector(graph, cov_elem, xi)

RM likelihood:
    rm_obs ~ Normal(A @ B.reshape(-1), diag(rm_noise^2))

This is exactly the standardized latent-space setup NIFTy.re is designed for:
the prior term is the standard Gaussian in xi, and all correlation structure is
encoded in the deterministic model xi -> B -> RM.
"""
from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from rm_graphgp_nifty_utils import (
    require_graphgp,
    load_npz,
    save_npz,
    response_apply_jax,
    endpoint_los_component,
    divergence_fd,
)

def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def import_jft():
    """Import NIFTy.re from either the NIFTy 9 or NIFTy 8 namespace."""
    try:
        import nifty.re as jft  # type: ignore
        return jft
    except Exception:
        try:
            import nifty8.re as jft  # type: ignore
            return jft
        except Exception as exc:
            raise ImportError(
                "Could not import NIFTy.re as `nifty.re` or `nifty8.re`."
            ) from exc


def shape_with_dtype(jft, shape, dtype):
    if hasattr(jft, "ShapeWithDtype"):
        try:
            return jft.ShapeWithDtype(shape, dtype)
        except TypeError:
            return jft.ShapeWithDtype(shape, dtype=dtype)
    # Very old/future fallback; most NIFTy.re versions have ShapeWithDtype.
    return shape


def make_initial_position(jft, domain, key):
    if hasattr(jft, "random_like"):
        raw = jft.random_like(key, domain)
    else:
        raise AttributeError("This NIFTy.re install has no `random_like`; please adapt initialization.")
    return jft.Vector(raw) if hasattr(jft, "Vector") else raw


def extract_xi_from_samples(samples: Any) -> np.ndarray:
    """Best-effort extraction of the posterior central position/mean xi.

    NIFTy.re sample containers have changed slightly across versions. This
    function handles the common cases and fails with a useful message otherwise.
    """
    for attr in ("pos", "position", "mean", "center"):
        if hasattr(samples, attr):
            obj = getattr(samples, attr)
            try:
                return np.asarray(obj["xi"])
            except Exception:
                pass

    if hasattr(samples, "sample_stat"):
        for reducer in ("mean", np.mean):
            try:
                stat = samples.sample_stat(reducer)
                return np.asarray(stat["xi"])
            except Exception:
                pass

    # Some versions return an iterable of residual/absolute sample vectors.
    try:
        xis = [np.asarray(s["xi"]) for s in samples]
        if xis:
            return np.mean(np.stack(xis, axis=0), axis=0)
    except Exception:
        pass

    raise RuntimeError(
        "Could not extract xi from NIFTy.re output. Inspect the printed type of `samples` "
        "and adapt `extract_xi_from_samples` for your NIFTy version."
    )


def _filter_kwargs(func, kwargs):
    """Return only keyword arguments accepted by `func`.

    This is intentionally defensive because NIFTy.re's `optimize_kl` keyword
    names changed between releases. If a function accepts **kwargs, the full
    dictionary is returned.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def call_optimize_kl(jft, likelihood, pos_init, key, args):
    """Call NIFTy.re optimize_kl using the installed API signature.

    NIFTy.re >=8/9 uses a keyword-only interface of the form

        optimize_kl(likelihood, position_or_samples, *, key,
                    n_total_iterations, n_samples, ...)

    Some older examples/wrappers used slightly different names. This function
    inspects the installed function and only passes supported keywords.
    """
    if not hasattr(jft, "optimize_kl"):
        raise AttributeError("This NIFTy.re install has no `optimize_kl`.")

    base_kwargs = dict(
        key=key,
        n_total_iterations=args.n_vi_iterations,
        n_samples=args.n_vi_samples,
        point_estimates=(),
        # Disable plotting in batch/script mode when these options exist.
        plot_energy_history=False,
        plot_minisanity_history=False,
    )

    # Support installations that renamed n_total_iterations.
    attempts = []
    attempts.append(base_kwargs)
    kw2 = dict(base_kwargs)
    kw2["n_iterations"] = kw2.pop("n_total_iterations")
    attempts.append(kw2)
    kw3 = dict(base_kwargs)
    kw3["iterations"] = kw3.pop("n_total_iterations")
    attempts.append(kw3)

    errors = []
    for kw in attempts:
        call_kwargs = _filter_kwargs(jft.optimize_kl, kw)
        try:
            print("Calling jft.optimize_kl with kwargs:", sorted(call_kwargs.keys()))
            out = jft.optimize_kl(likelihood, pos_init, **call_kwargs)
            if isinstance(out, tuple):
                return out[0], out[1:]
            return out, ()
        except TypeError as exc:
            errors.append(str(exc))

    raise TypeError("All optimize_kl call signatures failed:\n" + "\n---\n".join(errors))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("rm_graphgp_mock.npz"))
    ap.add_argument("--out", type=Path, default=Path("rm_nifty_reconstruction.npz"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-vi-iterations", type=int, default=16)
    ap.add_argument("--n-vi-samples", type=int, default=4)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--save-xi", action="store_true", help="save posterior xi estimate; may be large")
    args = ap.parse_args()

    d = load_npz(args.data)
    gp = require_graphgp()
    jft = import_jft()

    points = jnp.asarray(d["points"], dtype=jnp.float64)
    n = int(d["n"])
    Nvox = int(points.shape[0])
    graph_n0 = int(d["graph_n0"])
    graph_k = int(d["graph_k"])
    ell = float(d["ell"])
    sigma2 = float(d["sigma2"])
    periodic = bool(d["periodic"])

    print(f"Building GraphGP prior model: Nvox={Nvox}, n0={graph_n0}, k={graph_k}")
    graph = gp.build_graph(points, n0=graph_n0, k=graph_k)
    cov_elem = gp.extras.make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=periodic)

    rows = jnp.asarray(d["response_rows"], dtype=jnp.int64)
    cols = jnp.asarray(d["response_cols"], dtype=jnp.int64)
    vals = jnp.asarray(d["response_data"], dtype=jnp.float64)
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
            domain = {"xi": shape_with_dtype(jft, (Nvox, 3), jnp.float64)}
            super().__init__(domain=domain)

        def __call__(self, x):
            return graphgp_prior_to_rm(x["xi"])

    model = GraphGPRMForwardModel()

    def noise_cov_inv(x):
        return x / (rm_noise**2)

    # NIFTy.re likelihood composition changed across versions.
    # Current jft likelihoods are themselves Models and are composed with a
    # forward model via `.amend(model)`, not necessarily via the `@` operator.
    # See the NIFTy.re examples, e.g. `jft.Poissonian(data).amend(forward)`.
    try:
        gaussian = jft.Gaussian(rm_obs, noise_cov_inv)
    except TypeError:
        gaussian = jft.Gaussian(data=rm_obs, noise_cov_inv=noise_cov_inv)

    if hasattr(gaussian, "amend"):
        likelihood = gaussian.amend(model)
    else:
        # Very old NIFTy-style fallback.
        likelihood = gaussian @ model

    key = jr.PRNGKey(args.seed)
    key, sub = jr.split(key)
    pos_init = make_initial_position(jft, model.domain, sub)

    print("Running NIFTy.re optimize_kl...")
    samples, aux = call_optimize_kl(jft, likelihood, pos_init, key, args)
    print(f"NIFTy output sample object type: {type(samples)}")

    xi_est = extract_xi_from_samples(samples)
    xi_est_jax = jnp.asarray(xi_est, dtype=jnp.float64)
    B_est = np.asarray(graphgp_prior_to_B(xi_est_jax)).reshape((n, n, n, 3))
    rm_pred = np.asarray(graphgp_prior_to_rm(xi_est_jax))

    resid = (rm_pred - np.asarray(d["rm_obs"])) / np.asarray(d["rm_noise"])
    bpar_est = endpoint_los_component(B_est, d["pulsar_pos"], d["pulsar_dir"])
    sign_accuracy = float(np.mean(np.sign(bpar_est) == np.sign(d["bpar_true"])))
    reduced_chi2 = float(np.mean(resid**2))
    div = divergence_fd(B_est)
    B_rms = float(np.sqrt(np.mean(np.sum(B_est**2, axis=-1))))
    div_rms = float(np.sqrt(np.mean(div**2)))
    eps_rms = float(ell * div_rms / (B_rms + 1e-12))

    out_data = dict(d)
    out_data.update(
        B_est=B_est,
        rm_pred=rm_pred,
        bpar_est=bpar_est,
        reduced_chi2=np.asarray(reduced_chi2),
        sign_accuracy=np.asarray(sign_accuracy),
        B_est_rms=np.asarray(B_rms),
        div_est_rms=np.asarray(div_rms),
        epsilon_est_rms=np.asarray(eps_rms),
        nifty_method=np.asarray("nifty.re.optimize_kl"),
        nifty_config_json=np.asarray(json.dumps(json_safe(vars(args)))),  
    )
    
    if args.save_xi:
        out_data["xi_est"] = xi_est

    save_npz(args.out, **out_data)
    print(f"Saved reconstruction: {args.out}")
    print(f"reduced chi2={reduced_chi2:.4g}")
    print(f"endpoint B_parallel sign accuracy={sign_accuracy:.3f}")
    print(f"reconstructed ell*div_rms/B_rms={eps_rms:.4g}")


if __name__ == "__main__":
    main()
