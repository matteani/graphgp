# refine_vector_streaming.py
import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable
Array = jnp.ndarray
CovElem = Callable[[Array, Array], Array]  # (d,), (d,) -> (ncomp, ncomp)

def compute_cov_matrix_elem(cov_elem: CovElem, points_a: Array, points_b: Array) -> Array:
    """
    Build block covariance for lists of points:
    points_a: (Na, d), points_b: (Nb, d)
    returns (Na*ncomp, Nb*ncomp)
    """
    blocks = jax.vmap(lambda xa: jax.vmap(lambda yb: cov_elem(xa, yb))(points_b))(points_a)
    ncomp = blocks.shape[-1]
    blocks = jnp.transpose(blocks, (0, 2, 1, 3))  # (Na, ncomp, Nb, ncomp)
    return blocks.reshape((points_a.shape[0] * ncomp, points_b.shape[0] * ncomp))

def _local_conditional_mats_vector(cov_elem: CovElem, joint_points: Array, *, k: int, ncomp: int, jitter: float):
    """
    joint_points: (k+1, d)
    returns R (ncomp, k*ncomp) and cholD (ncomp, ncomp)
    """
    K = compute_cov_matrix_elem(cov_elem, joint_points, joint_points)
    m = k * ncomp
    Kcc = K[:m, :m]
    Kcf = K[:m, m:]
    Kff = K[m:, m:]
    A = jnp.linalg.solve(Kcc, Kcf)   # (m, ncomp)
    D = Kff - Kcf.T @ A              # (ncomp, ncomp)
    D = D + jitter * jnp.eye(ncomp, dtype=D.dtype)
    cholD = jnp.linalg.cholesky(D)
    R = A.T
    return R, cholD

@jax.jit
def refine_vector_streaming(
    points: Array,            # (N, d) tree order
    neighbors: Array,         # (N-n0, k)
    offsets: Array,           # (num_offsets,) offset integers in [0..N]
    cov_elem: CovElem,
    initial_values: Array,    # (n0, ncomp)
    xi: Array,                # (N-n0, ncomp)
    *,
    jitter: float = 1e-6,
):
    n0 = points.shape[0] - neighbors.shape[0]
    k = neighbors.shape[1]
    ncomp = initial_values.shape[1]
    d = points.shape[1]

    # batch sizes
    batch_sizes = offsets[1:] - offsets[:-1]
    max_batch = jnp.max(batch_sizes)

    # pad arrays so we can always slice fixed-size windows
    points_pad = jnp.pad(points, ((0, int(max_batch)), (0, 0)))
    neighbors_pad = jnp.pad(neighbors, ((0, int(max_batch)), (0, 0)), mode="edge")
    xi_pad = jnp.pad(xi, ((0, int(max_batch)), (0, 0)))
    values = jnp.zeros((points_pad.shape[0], ncomp), dtype=initial_values.dtype)
    values = values.at[:n0, :].set(initial_values)

    nbatches = offsets.shape[0] - 1

    def body(i, vals):
        start = offsets[i]
        end = offsets[i+1]
        B = end - start
        i0 = start - n0

        # fixed-size windows (max_batch)
        neigh = lax.dynamic_slice(neighbors_pad, (i0, 0), (int(max_batch), k))         # (max_batch, k)
        coarse_pts = points_pad[neigh]                                                  # (max_batch, k, d)
        fine_pts = lax.dynamic_slice(points_pad, (start, 0), (int(max_batch), d))      # (max_batch, d)
        joint = jnp.concatenate([coarse_pts, fine_pts[:, None, :]], axis=1)            # (max_batch, k+1, d)

        # compute R and cholD for the batch
        R_batch, cholD_batch = jax.vmap(
            lambda jp: _local_conditional_mats_vector(cov_elem, jp, k=k, ncomp=ncomp, jitter=jitter),
            in_axes=0,
            out_axes=(0,0),
        )(joint)  # shapes: (max_batch, ncomp, m), (max_batch, ncomp, ncomp)

        neigh_vals = vals[neigh].reshape((int(max_batch), k*ncomp))
        mean = jnp.einsum("bim,bm->bi", R_batch, neigh_vals)
        z = lax.dynamic_slice(xi_pad, (i0, 0), (int(max_batch), ncomp))
        noise = jnp.einsum("bij,bj->bi", cholD_batch, z)

        out = mean + noise

        # Only write the first B rows (mask)
        mask = (jnp.arange(int(max_batch))[:, None] < B)
        current = lax.dynamic_slice(vals, (start, 0), (int(max_batch), ncomp))
        out = jnp.where(mask, out, current)
        vals = lax.dynamic_update_slice(vals, out, (start, 0))
        return vals

    values = lax.fori_loop(0, nbatches, body, values)
    return values[:points.shape[0], :]
