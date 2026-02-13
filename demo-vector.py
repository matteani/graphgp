# %%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import graphgp as gp

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

def divergence_central(B, spacing):
    dx, dy, dz = spacing
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    dBx_dx = (jnp.roll(Bx, -1, axis=0) - jnp.roll(Bx,  1, axis=0)) / (2.0 * dx)
    dBy_dy = (jnp.roll(By, -1, axis=1) - jnp.roll(By,  1, axis=1)) / (2.0 * dy)
    dBz_dz = (jnp.roll(Bz, -1, axis=2) - jnp.roll(Bz,  1, axis=2)) / (2.0 * dz)
    return dBx_dx + dBy_dy + dBz_dz

# %%
rng = jr.key(0)

# Regular periodic grid of points in [0,1)^3
n = 75
xs = (jnp.arange(n) + 0.5) / n
X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))
N = points.shape[0]

# Build GraphGP dependency graph
graph = gp.build_graph(points, n0=250, k=20)

# Divergence-free covariance element
cov_elem = make_div_free_rbf_covariance(ell=0.05, sigma2=1.0, periodic=True)

# White parameters xi (N, 3)
rng, key = jr.split(rng)
xi = jr.normal(key, (N, 3))

# Generate vector field values (N, 3)
B = gp.generate_vector(graph, cov_elem, xi, fast_jit=True)

# Reshape to grid and compute diagnostics
Bgrid = B.reshape((n, n, n, 3))
spacing = (1.0 / n, 1.0 / n, 1.0 / n)

divB = divergence_central(Bgrid, spacing)
Bmag = jnp.linalg.norm(Bgrid, axis=-1)
print("B_rms:", float(jnp.sqrt(jnp.mean(Bmag**2))))
print("divB_rms:", float(jnp.sqrt(jnp.mean(divB**2))))
print("divB max abs:", float(jnp.max(jnp.abs(divB))))

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

