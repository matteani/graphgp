# %%
import jax
from refine_vector import compute_cov_matrix_elem
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import graphgp as gp
import itertools

from local_block_helmholtz import helmholtz_local_block_projection


# %%
def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=True):
    ell2 = ell * ell
    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)  # assumes coords in [0,1)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)

        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))
    return cov_elem

def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=True, image_sum=True):
    """
    PSD divergence-free kernel in 3D built from RBF scalar kernel.
    sigma2 is the per-component variance at r=0, so K(x,x)=sigma2*I.
    If periodic and image_sum=True, uses 27-image periodic summation (PSD-safe).
    """
    ell2 = ell * ell
    ell4 = ell2 * ell2
    I3 = jnp.eye(3)

    # internal amplitude so that K(0) = sigma2 * I
    # For 3D: K(0) = A * (2/ell^2) I  =>  A = sigma2 * ell^2 / 2
    A = sigma2 * ell2 / 2.0

    shifts = [(0, 0, 0)]
    if periodic and image_sum:
        shifts = list(itertools.product([-1, 0, 1], repeat=3))

    def cov_elem(x, y):
        r0 = y - x  # points assumed in [0,1)^3 if periodic
        K = jnp.zeros((3, 3), dtype=r0.dtype)

        for s in shifts:
            s = jnp.array(s, dtype=r0.dtype)
            r = r0 + s

            # If you prefer min-image instead of image sum:
            if periodic and (not image_sum):
                r = r - jnp.round(r)

            rr2 = jnp.dot(r, r)
            base = jnp.exp(-0.5 * rr2 / ell2)

            a = (2.0 / ell2 - rr2 / ell4) * base
            b = (1.0 / ell4) * base
            K = K + A * (a * I3 + b * jnp.outer(r, r))

        return K
    
    # Attach metadata the CUDA dispatcher will read
    cov_elem._graphgp_cuda = {
        "kernel": "divfree_rbf",
        "ell": float(ell),
        "sigma2": float(sigma2),
        "periodic": bool(periodic),
        "image_sum": bool(image_sum),
    }

    return cov_elem

def divergence_central(B, spacing):
    dx, dy, dz = spacing
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    dBx_dx = (jnp.roll(Bx, -1, axis=0) - jnp.roll(Bx,  1, axis=0)) / (2.0 * dx)
    dBy_dy = (jnp.roll(By, -1, axis=1) - jnp.roll(By,  1, axis=1)) / (2.0 * dy)
    dBz_dz = (jnp.roll(Bz, -1, axis=2) - jnp.roll(Bz,  1, axis=2)) / (2.0 * dz)
    return dBx_dx + dBy_dy + dBz_dz

def divergence_nonperiodic(B, dx):
    # B: (n,n,n,3)
    dBx = (B[2:,1:-1,1:-1,0] - B[:-2,1:-1,1:-1,0]) / (2*dx)
    dBy = (B[1:-1,2:,1:-1,1] - B[1:-1,:-2,1:-1,1]) / (2*dx)
    dBz = (B[1:-1,1:-1,2:,2] - B[1:-1,1:-1,:-2,2]) / (2*dx)
    return dBx + dBy + dBz

# %%
rng = jr.key(0)

# Regular periodic grid of points in [0,1)^3
n = 30 #10
xs = 1*(jnp.arange(n) + 0.5) / n
print(xs)
X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))
N = points.shape[0]

# Build GraphGP dependency graph
graph = gp.build_graph(points, n0=250, k=30) #k=20

ell = (100/n)*0.2
sigma2=1.0
# Divergence-free covariance element
cov_elem = make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=False) #0.4

""""
K = gp.compute_cov_matrix_elem(cov_elem, points, points)
K = 0.5 * (K + K.T)
evals = jnp.linalg.eigvalsh(K)
print("min eigenvalue:", float(evals[0]))
print("K finite:", bool(jnp.all(jnp.isfinite(K))))
print("K sym err:", float(jnp.max(jnp.abs(K - K.T))))
print("diag min/max:", float(jnp.min(jnp.diag(K))), float(jnp.max(jnp.diag(K))))
"""

# White parameters xi (N, 3)
rng, key = jr.split(rng)
xi = jr.normal(key, (N, 3))

# Generate vector field values (N, 3)
B = gp.generate_vector(graph, cov_elem, xi, cuda=True, fast_jit=True)
#B = gp.generate_dense_vector(points, cov_elem, xi)

# Reshape to grid and compute diagnostics
Bgrid = B.reshape((n, n, n, 3))
spacing = (1.0 / n, 1.0 / n, 1.0 / n)
dx = spacing[0]
"""
# choose block size & overlap: tradeoff accuracy vs speed
Bgrid = helmholtz_local_block_projection(Bgrid, dx,
                                        block_size=32,   # central block size
                                        overlap=8,       # padding for local solve
                                        solver_tol=1e-6)
"""

divB = divergence_nonperiodic(Bgrid, dx)
Bmag = jnp.linalg.norm(Bgrid, axis=-1)
B_rms = float(jnp.sqrt(jnp.mean(Bmag**2)))
divB_rms = float(jnp.sqrt(jnp.mean(divB**2)))
ratio = ell * divB_rms / B_rms
print("B_rms:", B_rms)
print("divB_rms:", divB_rms)
print("divB_rms / B_rms:", ratio)
print("divB max abs:", float(jnp.max(jnp.abs(divB))))

def divergence_fft(B):
    """
    Spectral divergence on periodic grid.

    B: array of shape (nx, ny, nz, 3)
    returns: real-space divergence (nx, ny, nz)
    """
    nx, ny, nz = B.shape[:3]

    # FFT of vector field
    Bhat = jnp.fft.fftn(B, axes=(0,1,2))

    # Fourier wavenumbers
    kx = 2*jnp.pi * jnp.fft.fftfreq(nx)
    ky = 2*jnp.pi * jnp.fft.fftfreq(ny)
    kz = 2*jnp.pi * jnp.fft.fftfreq(nz)

    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")

    # Compute i k · Bhat
    divhat = 1j * (
        KX * Bhat[...,0] +
        KY * Bhat[...,1] +
        KZ * Bhat[...,2]
    )

    # Transform back to real space
    div = jnp.fft.ifftn(divhat, axes=(0,1,2)).real

    return div

def divergence_fft(B, spacing=(1.0, 1.0, 1.0)):
    nx, ny, nz = B.shape[:3]
    dx, dy, dz = spacing

    Bhat = jnp.fft.fftn(B, axes=(0,1,2))

    kx = 2*jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2*jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kz = 2*jnp.pi * jnp.fft.fftfreq(nz, d=dz)

    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")

    divhat = 1j * (KX*Bhat[...,0] + KY*Bhat[...,1] + KZ*Bhat[...,2])
    return jnp.fft.ifftn(divhat, axes=(0,1,2)).real

div_fft = divergence_fft(Bgrid, spacing)
div_fft_rms = float(jnp.sqrt(jnp.mean(div_fft**2)))
ratio_fft = div_fft_rms / B_rms
print("div_fft_rms:", div_fft_rms)
print("div_fft_rms / B_rms:", ratio_fft)
print("div_fft max abs:", float(jnp.max(jnp.abs(div_fft))))

# div_fft already computed
zslice = n // 2

plt.figure(figsize=(5,5))
plt.imshow(divB[:, :, zslice], cmap="seismic", norm = mcolors.TwoSlopeNorm(vcenter = 0, vmin=-25, vmax=25))
plt.title("Finite Differences Divergence (mid plane)")
plt.colorbar()
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(div_fft[:, :, zslice], cmap="seismic", norm = mcolors.TwoSlopeNorm(vcenter = 0, vmin=-25, vmax=25))
plt.title("FFT divergence (mid plane)")
plt.colorbar()
plt.show()

def longitudinal_fraction_spectrum(B):
    # B: (nx,ny,nz,3)
    nx,ny,nz = B.shape[:3]
    Bhat = jnp.fft.fftn(B, axes=(0,1,2))
    # wavenumbers (radians)
    kx = 2*jnp.pi * jnp.fft.fftfreq(nx)
    ky = 2*jnp.pi * jnp.fft.fftfreq(ny)
    kz = 2*jnp.pi * jnp.fft.fftfreq(nz)
    KX,KY,KZ = jnp.meshgrid(kx,ky,kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    # avoid k=0
    k2_safe = jnp.where(k2==0, 1.0, k2)
    # Bhat components
    Bxhat = Bhat[...,0]
    Byhat = Bhat[...,1]
    Bzhat = Bhat[...,2]
    # longitudinal (k·Bhat)^2 / k^2
    kdotB = KX*Bxhat + KY*Byhat + KZ*Bzhat
    Elong = jnp.abs(kdotB)**2 / k2_safe   # energy in longitudinal part per mode
    Etot = (jnp.abs(Bxhat)**2 + jnp.abs(Byhat)**2 + jnp.abs(Bzhat)**2)
    # bin by |k| (radial shell)
    kmag = jnp.sqrt(k2)
    kflat = kmag.ravel()
    elong_flat = Elong.ravel()
    etot_flat = Etot.ravel()
    # build histogram bins
    kbins = jnp.linspace(0, jnp.max(kmag), 50)
    import numpy as np
    kb = np.asarray(kbins)
    # use numpy to aggregate (simple)
    import numpy as onp
    inds = onp.digitize(onp.asarray(kflat), kb) - 1
    elong_bin = onp.zeros(len(kb)-1)
    etot_bin = onp.zeros(len(kb)-1)
    for i in range(len(kb)-1):
        mask = (inds == i)
        if mask.sum()>0:
            elong_bin[i] = onp.sum(onp.asarray(elong_flat[mask]))
            etot_bin[i] = onp.sum(onp.asarray(etot_flat[mask]))
    frac = onp.where(etot_bin>0, elong_bin / etot_bin, 0.0)
    return kb, frac, elong_bin, etot_bin

kb, frac, eL, eT = longitudinal_fraction_spectrum(Bgrid)

plt.figure()
plt.plot(kb[:-1], frac)
plt.xlabel('|k|')
plt.ylabel('Longitudinal fraction per shell')
plt.yscale('log')
plt.title('Longitudinal energy fraction vs k')
plt.show()

divB_osr = divB[1:-1, 1:-1, 1:-1]
div_fft_osr = div_fft[1:-1, 1:-1, 1:-1]

plt.figure()
plt.hist(divB.ravel(), bins=500, density=True, alpha=0.5)
plt.hist(divB_osr.ravel(), bins=500, alpha=0.5, density=True, label="OSR")
plt.title("Histogram of divergence")
plt.show()

plt.figure()
plt.hist(div_fft.ravel(), bins=500, density=True, alpha=0.5)
plt.hist(div_fft_osr.ravel(), bins=500, alpha=0.5, density=True, label="OSR")
plt.title("Histogram of spectral divergence")
plt.show()


# %%
# Plot a mid-plane slice, like the ICR demo
zslice = n // 2
Bx = Bgrid[:, :, zslice, 0]
By = Bgrid[:, :, zslice, 1]
Bz = Bgrid[:, :, zslice, 2]
Babs = jnp.linalg.norm(Bgrid[:, :, zslice, :], axis=-1)

# Quiver of (Bx, By) on the slice
xx = jnp.arange(n)
yy = jnp.arange(n)
XX, YY = jnp.meshgrid(xx, yy, indexing="ij")
plt.figure(figsize=(5, 5))
plt.quiver(XX, YY, Bx, By, scale=50)
plt.title(f"Vector field at z={zslice}")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig("temp-figures/demo_vector_quiver.png")
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
axs[0].imshow(Bx);   axs[0].set_title("B_x")
axs[1].imshow(By);   axs[1].set_title("B_y")
axs[2].imshow(Bz);   axs[2].set_title("B_z")
axs[3].imshow(Babs); axs[3].set_title("|B|")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.savefig("temp-figures/demo_vector_fields.png")
plt.show()

