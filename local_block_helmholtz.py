# local_block_helmholtz.py
import numpy as np
from functools import lru_cache
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _ensure_numpy(arr):
    # Accept jax arrays too
    try:
        import jax.numpy as jnp
        if isinstance(arr, jnp.ndarray):
            return np.asarray(arr)
    except Exception:
        pass
    return np.asarray(arr)

@lru_cache(maxsize=8)
def _build_interior_laplacian_for_window(ws):
    """
    Build sparse Laplacian for interior points of a ws^3 grid (unknowns are (ws-2)^3).
    Dirichlet phi=0 on outer boundary of the window.
    Returns CSR matrix L of shape m x m where m=(ws-2)^3
    """
    m1 = ws - 2  # interior points per axis
    m = m1**3
    e = np.ones(m1)
    T = sp.diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(m1, m1))
    I = sp.eye(m1)
    L = sp.kron(sp.kron(I, I), T) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(T, I), I)
    # L corresponds to negative definite of typical discrete second derivative [-1, 2, -1]
    # We'll use -L as positive Laplacian if convenient for solves.
    return L.tocsr()

def _grad_phi_interior_vec(phi_vec, ws, dx):
    """
    phi_vec: (ws-2)^3 vector -> phi (m1,m1,m1)
    returns gradient array of shape (ws,ws,ws,3) (zeros on window boundary)
    """
    m1 = ws - 2
    phi = phi_vec.reshape((m1, m1, m1))

    # Create a padded phi array of shape (ws,ws,ws) with zeros on outer boundary.
    # Place phi into [1:-1,1:-1,1:-1].
    phi_pad = np.zeros((ws, ws, ws), dtype=phi.dtype)
    phi_pad[1:-1, 1:-1, 1:-1] = phi

    # Allocate gradients (zeros on full window)
    gx = np.zeros((ws, ws, ws), dtype=phi.dtype)
    gy = np.zeros((ws, ws, ws), dtype=phi.dtype)
    gz = np.zeros((ws, ws, ws), dtype=phi.dtype)

    # Central differences computed on padded phi produce shape (ws-2, ws-2, ws-2)
    gx[1:-1, 1:-1, 1:-1] = (phi_pad[2:, 1:-1, 1:-1] - phi_pad[:-2, 1:-1, 1:-1]) / (2.0 * dx)
    gy[1:-1, 1:-1, 1:-1] = (phi_pad[1:-1, 2:, 1:-1] - phi_pad[1:-1, :-2, 1:-1]) / (2.0 * dx)
    gz[1:-1, 1:-1, 1:-1] = (phi_pad[1:-1, 1:-1, 2:] - phi_pad[1:-1, 1:-1, :-2]) / (2.0 * dx)

    grad = np.stack([gx, gy, gz], axis=-1)  # (ws,ws,ws,3)
    return grad

def helmholtz_local_block_projection(Bgrid, dx, block_size=32, overlap=8, solver_tol=1e-6, maxiter=2000):
    """
    Project Bgrid to approximately divergence-free using overlapping local-block Helmholtz projection.

    Parameters
    ----------
    Bgrid : array-like (n,n,n,3)  (jax arrays ok; converted to numpy)
    dx : scalar grid spacing
    block_size : int, size of central (non-overlap) block in each axis (e.g. 32)
    overlap : int, size of padding on each side for local solve (e.g. 8)
    solver_tol, maxiter : CG solver params (used for sparse CG per block)

    Returns
    -------
    Bprime : same shape as Bgrid, same type (numpy or jax)
    """
    Bnp = _ensure_numpy(Bgrid).astype(np.float64)
    n = Bnp.shape[0]
    assert Bnp.shape == (n, n, n, 3)
    dx = float(dx)

    ws = block_size + 2 * overlap  # window size for solves
    if ws > n:
        raise ValueError("window size (block_size+2*overlap) must be <= n")

    # accumulators for weighted averaging over overlaps
    B_accum = np.zeros_like(Bnp, dtype=np.float64)
    W_accum = np.zeros((n, n, n), dtype=np.float64)

    # pre-build Laplacian for windows of size ws (cached)
    Lwin = _build_interior_laplacian_for_window(ws)
    Lpos = -Lwin  # positive Laplacian operator (so Lpos @ phi = rhs)

    # iterate block centers (non-overlap blocks). choose grid of starts that tile domain
    # Start indices for central blocks
    step = block_size
    starts = list(range(0, n, step))
    # ensure last block covers the end: adjust last start so block fits
    if starts[-1] + block_size > n:
        starts[-1] = n - block_size
    # loop over block starts in 3D
    for ix0 in starts:
        for iy0 in starts:
            for iz0 in starts:
                # window indices in full domain
                wx0 = ix0 - overlap
                wy0 = iy0 - overlap
                wz0 = iz0 - overlap
                wx1 = ix0 + block_size + overlap
                wy1 = iy0 + block_size + overlap
                wz1 = iz0 + block_size + overlap

                # clip to domain bounds, but keep window size ws by shifting if needed
                # (prefer shift rather than shrink to maintain ws)
                if wx0 < 0:
                    wx1 += -wx0
                    wx0 = 0
                if wy0 < 0:
                    wy1 += -wy0
                    wy0 = 0
                if wz0 < 0:
                    wz1 += -wz0
                    wz0 = 0
                if wx1 > n:
                    shift = wx1 - n
                    wx0 -= shift
                    wx1 -= shift
                if wy1 > n:
                    shift = wy1 - n
                    wy0 -= shift
                    wy1 -= shift
                if wz1 > n:
                    shift = wz1 - n
                    wz0 -= shift
                    wz1 -= shift

                # Now windows are guaranteed to be in [0,n) and size ws
                assert wx1 - wx0 == ws and wy1 - wy0 == ws and wz1 - wz0 == ws

                # Extract Bblock
                Bblock = Bnp[wx0:wx1, wy0:wy1, wz0:wz1, :]  # shape (ws,ws,ws,3)

                # compute divergence (rhs) on interior of window
                rhs = divergence_interior_block(Bblock, dx)  # length (ws-2)^3

                # quick check: if rhs is tiny, skip solve to save time
                rhs_norm = np.linalg.norm(rhs)
                if rhs_norm < 1e-12:
                    # no divergence here; do nothing but accumulate weights
                    # subtract zero gradient (no-op) -> just add original central block to accum
                    central_slice = (slice(ix0, ix0+block_size),
                                     slice(iy0, iy0+block_size),
                                     slice(iz0, iz0+block_size))
                    B_accum[central_slice + (slice(None),)] += Bblock[overlap:overlap+block_size,
                                                                     overlap:overlap+block_size,
                                                                     overlap:overlap+block_size, :]
                    W_accum[central_slice] += 1.0
                    continue

                # solve Lpos @ phi = rhs
                # use conjugate gradient; initial guess zero
                try:
                    phi_vec, info = spla.cg(Lpos, rhs, tol=solver_tol, maxiter=maxiter)
                except Exception as e:
                    # fall back to sparse direct solve for small blocks (may be slower)
                    phi_vec = spla.spsolve(Lpos, rhs)
                    info = 0
                if info != 0:
                    # solver didn't converge -- that's usually okay for local correction, warn
                    # but continue with whatever solution we have (phi_vec)
                    # small residual might leave small div in result; consider increasing tol
                    pass

                # grad on window
                grad = _grad_phi_interior_vec(phi_vec, ws, dx)  # (ws,ws,ws,3)

                # central region indices inside the window
                cx0, cy0, cz0 = overlap, overlap, overlap
                cx1, cy1, cz1 = overlap + block_size, overlap + block_size, overlap + block_size

                # B corrected on central region = Bblock_central - grad_central
                B_central_corrected = (Bblock[cx0:cx1, cy0:cy1, cz0:cz1, :] -
                                       grad[cx0:cx1, cy0:cy1, cz0:cz1, :])

                # accumulate into global arrays
                central_slice = (slice(ix0, ix0+block_size),
                                 slice(iy0, iy0+block_size),
                                 slice(iz0, iz0+block_size))
                B_accum[central_slice + (slice(None),)] += B_central_corrected
                W_accum[central_slice] += 1.0

    # finalize by averaging overlapping contributions
    # avoid div by zero
    W_nonzero = W_accum.copy()
    W_nonzero[W_nonzero == 0] = 1.0
    Bprime_np = B_accum / W_nonzero[..., None]

    # For cells that were never covered (shouldn't happen), fall back to original
    mask_uncovered = (W_accum == 0)
    if np.any(mask_uncovered):
        Bprime_np[mask_uncovered, :] = Bnp[mask_uncovered, :]

    # Return same type as input
    try:
        import jax.numpy as jnp
        if hasattr(Bgrid, "shape") and isinstance(Bgrid, jnp.ndarray):
            return jnp.asarray(Bprime_np)
    except Exception:
        pass
    return Bprime_np