#!/usr/bin/env python3
"""
demo-derivdiag.py

GraphGP demo: sample u(x) = (∂x Bx, ∂y By, ∂z Bz) directly by differentiating the
divergence-free covariance tensor.

Expected: u_x + u_y + u_z ~ 0 (up to jitter), since div(B)=0 in the underlying model.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import graphgp as gp


def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=True):
    """
    Your existing divergence-free RBF tensor:

      C_ij(r) = σ² [ a(r) δ_ij + b(r) r_i r_j ]
      a(r) = (1 - r²/(2ℓ²)) exp(-r²/(2ℓ²))
      b(r) = (1/(2ℓ²)) exp(-r²/(2ℓ²))

    Returns cov_elem(x,y)->(3,3).
    """
    ell2 = ell * ell
    c = 1.0 / (2.0 * ell2)

    def cov(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)  # unit box minimum-image wrap (piecewise)

        s = jnp.dot(r, r)        # r^2
        base = jnp.exp(-c * s)   # exp(-r^2/(2ell^2))

        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - c * s) * base
        b = c * base
        return sigma2 * (a * I + b * jnp.outer(r, r))

    return cov


def make_diag_deriv_covariance_from_div_free_rbf(ell, sigma2=1.0, periodic=True, jitter=1e-10):
    """
    Build cov_elem_u(x,y)->(3,3) for u=(∂xBx, ∂yBy, ∂zBz) by analytic differentiation:

      K_ab(r) = Cov(∂_{x_a} B_a(x), ∂_{y_b} B_b(y)) = -∂_{r_a}∂_{r_b} C_ab(r)

    IMPORTANT: For an exactly divergence-free B, this K is singular (one zero eigenvalue).
    We add `jitter*I` so GraphGP can Cholesky-sample it.
    """
    ell2 = ell * ell
    c = 1.0 / (2.0 * ell2)
    I = jnp.eye(3)

    # We reuse the closed form C_ij(r) = g(s) * A_ij(r), with s=r^2, g=σ² exp(-c s),
    # A=(1-c s)I + c r r^T, and differentiate analytically.
    def cov_u(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)

        s = jnp.dot(r, r)
        f = jnp.exp(-c * s)
        g = sigma2 * f

        A = (1.0 - c * s) * I + c * jnp.outer(r, r)

        # g_p = ∂_{r_p} g = -2 c g r_p
        g_p = -2.0 * c * g * r                          # (3,)

        # g_pq = ∂_{r_q}∂_{r_p} g = 4 c^2 g r_p r_q - 2 c g δ_pq
        g_pq = 4.0 * (c**2) * g * jnp.outer(r, r) - 2.0 * c * g * I  # (3,3)

        # A_p (tensor): (A_p)_{ij} = ∂_{r_p} A_{ij}
        # = -2 c r_p δ_ij + c(δ_{ip} r_j + r_i δ_{jp})
        def A_p_of(p):
            e = I[p]  # basis vector
            return -2.0 * c * r[p] * I + c * (jnp.outer(e, r) + jnp.outer(r, e))

        A_p = jax.vmap(A_p_of)(jnp.arange(3))            # (p,i,j) = (3,3,3)

        # A_pq (tensor): ∂_{r_q}∂_{r_p} A_{ij}
        # = -2 c δ_pq δ_ij + c(δ_{ip}δ_{jq} + δ_{iq}δ_{jp})
        term1 = (-2.0 * c) * jnp.einsum("pq,ij->pqij", I, I)
        term2 = (c) * jnp.einsum("ip,jq->pqij", I, I)
        term3 = (c) * jnp.einsum("iq,jp->pqij", I, I)
        A_pq = term1 + term2 + term3                     # (p,q,i,j)

        # D_pq C_ij = ∂_{r_q}∂_{r_p} (g A_ij)
        #          = g_pq A_ij + g_p A_q,ij + g_q A_p,ij + g A_pq,ij
        D = (
            jnp.einsum("pq,ij->pqij", g_pq, A)
            + jnp.einsum("p,qij->pqij", g_p, A_p)
            + jnp.einsum("q,pij->pqij", g_p, A_p)
            + g * A_pq
        )

        # Our desired K_ab = - D_{p=a,q=b,i=a,j=b}
        K = jnp.zeros((3, 3), dtype=g.dtype)
        for a in range(3):
            for b in range(3):
                K = K.at[a, b].set(-D[a, b, a, b])

        # jitter to make SPD for Cholesky (div-free makes it singular otherwise)
        K = K + jitter * I
        return K

    return jax.jit(cov_u)


def divergence_central(B, spacing):
    """Central-difference divergence for a gridded vector field B[nx,ny,nz,3] with periodic wrap."""
    dx, dy, dz = spacing
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    dBx_dx = (jnp.roll(Bx, -1, axis=0) - jnp.roll(Bx, 1, axis=0)) / (2.0 * dx)
    dBy_dy = (jnp.roll(By, -1, axis=1) - jnp.roll(By, 1, axis=1)) / (2.0 * dy)
    dBz_dz = (jnp.roll(Bz, -1, axis=2) - jnp.roll(Bz, 1, axis=2)) / (2.0 * dz)
    return dBx_dx + dBy_dy + dBz_dz


def main():
    outdir = Path("./figures-derivdiag")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- grid of points in [0,1)^3 ----
    n = 50
    coords_1d = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))  # (N,3)
    N = points.shape[0]

    # ---- GraphGP graph ----
    n0 = min(250, int(0.2 * N))
    k = 10
    graph = gp.build_graph(points, n0=n0, k=k)

    # ---- underlying div-free B kernel params ----
    ell = 0.15
    sigma2 = 1.0

    # (A) optional: sample B itself to compare
    cov_B = make_div_free_rbf_covariance(ell, sigma2=sigma2, periodic=True)

    # (B) sample u=(∂xBx,∂yBy,∂zBz) using differentiated covariance
    jitter = 1e3
    cov_u = make_diag_deriv_covariance_from_div_free_rbf(ell, sigma2=sigma2, periodic=True, jitter=jitter)

    key = jr.PRNGKey(0)
    key, kB, kU = jr.split(key, 3)

    xiB = jr.normal(kB, (N, 3))
    xiU = jr.normal(kU, (N, 3))

    # NOTE: cuda=True is not implemented for vector refinement in your code; keep cuda=False.
    B = gp.generate_vector(graph, cov_B, xiB, cuda=False, fast_jit=True)   # (N,3)
    U = gp.generate_vector(graph, cov_u, xiU, cuda=False, fast_jit=True)   # (N,3)

    B3 = B.reshape((n, n, n, 3))
    U3 = U.reshape((n, n, n, 3))

    # ---- diagnostics ----
    spacing = (1.0 / n, 1.0 / n, 1.0 / n)

    divB_fd = divergence_central(B3, spacing)
    divB_fd_rms = jnp.sqrt(jnp.mean(divB_fd**2))

    divU = U3[..., 0] + U3[..., 1] + U3[..., 2]
    divU_rms = jnp.sqrt(jnp.mean(divU**2))

    Urms = jnp.sqrt(jnp.mean(jnp.sum(U3**2, axis=-1)))

    print("N:", int(N), "n0:", int(n0), "k:", int(k))
    print("ell:", float(ell), "sigma2:", float(sigma2), "jitter:", jitter)
    print("Finite-difference div(B) rms:", float(divB_fd_rms))
    print("Generated (Ux+Uy+Uz) rms:", float(divU_rms))
    print("Generated U rms:", float(Urms))
    print("Ratio (Ux+Uy+Uz)_rms / U_rms:", float(divU_rms / (Urms + 1e-12)))

    # ---- plots: mid-z slice ----
    zslice = n // 2

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    axs[0].imshow(U3[:, :, zslice, 0]); axs[0].set_title("Ux = ∂xBx (sampled)")
    axs[1].imshow(U3[:, :, zslice, 1]); axs[1].set_title("Uy = ∂yBy (sampled)")
    axs[2].imshow(U3[:, :, zslice, 2]); axs[2].set_title("Uz = ∂zBz (sampled)")
    axs[3].imshow(divU[:, :, zslice]);  axs[3].set_title("Ux+Uy+Uz (should be ~0)")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "U_and_sum_slice.png")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].imshow(divB_fd[:, :, zslice]); axs[0].set_title("FD div(B) on sampled B")
    axs[1].imshow(divU[:, :, zslice]);    axs[1].set_title("Sampled (Ux+Uy+Uz)")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "div_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
