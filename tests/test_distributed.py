import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.sharding import Mesh

import graphgp as gp

jax.config.update("jax_enable_x64", True)


def _setup(dtype=jnp.float64):
    n = 48
    points = jnp.linspace(0.0, 1.0, n, dtype=dtype)[:, None]
    graph = gp.build_graph(points, n0=8, k=4)
    covariance = gp.extras.matern_kernel(
        p=0,
        variance=1.0,
        cutoff=0.3,
        r_min=1e-4,
        r_max=2.0,
        n_bins=64,
        jitter=1e-5,
    )
    return graph, covariance


@pytest.mark.parametrize("n_partitions", [1, 2, 4])
def test_recompute_matches_generate(n_partitions):
    graph, covariance = _setup()
    owners = np.arange(len(graph.points)) % n_partitions
    plan = gp.distributed.partition_graph(graph, owners)
    xi = jr.normal(jr.key(1), (2, len(graph.points)), dtype=jnp.float64)

    expected = jax.vmap(lambda z: gp.generate(graph, covariance, z))(xi)
    actual = gp.distributed.generate_recompute(plan, covariance, xi)

    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_exchange_matches_generate_and_is_sharded():
    n_partitions = min(jax.device_count(), 4)
    graph, covariance = _setup()
    owners = np.arange(len(graph.points)) % n_partitions
    plan = gp.distributed.partition_graph(graph, owners)
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    xi = jr.normal(jr.key(2), (len(graph.points),), dtype=jnp.float64)

    expected = gp.generate(graph, covariance, xi)
    actual = gp.distributed.generate(plan, covariance, xi, mesh=mesh)

    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual.sharding.mesh.shape["space"] == n_partitions


def test_exchange_batched_adjoint_and_covariance_gradient():
    n_partitions = min(jax.device_count(), 4)
    graph, covariance = _setup()
    owners = np.repeat(np.arange(n_partitions), len(graph.points) // n_partitions)
    plan = gp.distributed.partition_graph(graph, owners)
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    cov_bins, cov_vals = covariance

    k1, k2, k3 = jr.split(jr.key(3), 3)
    xi = jr.normal(k1, (2, len(graph.points)), dtype=jnp.float64)
    xi_tangent = jr.normal(k2, xi.shape, dtype=xi.dtype)
    value_tangent = jr.normal(k3, xi.shape, dtype=xi.dtype)

    def generate(x, values=cov_vals):
        return gp.distributed.generate(plan, (cov_bins, values), x, mesh=mesh)

    _, jvp = jax.jvp(generate, (xi,), (xi_tangent,))
    _, pullback = jax.vjp(generate, xi)
    vjp = pullback(value_tangent)[0]
    lhs = jnp.vdot(value_tangent, jvp)
    rhs = jnp.vdot(vjp, xi_tangent)
    assert jnp.allclose(lhs, rhs, rtol=1e-8, atol=1e-8)

    cov_grad = jax.grad(lambda values: jnp.sum(generate(xi, values)))(cov_vals)
    assert jnp.all(jnp.isfinite(cov_grad))


def test_partition_stats_and_validation():
    graph, _ = _setup()
    owners = np.arange(len(graph.points)) % 2
    plan = gp.distributed.partition_graph(graph, owners, output_shape=(6, 8))
    stats = gp.distributed.partition_stats(plan)

    assert stats.owned_nodes == (24, 24)
    assert stats.cross_partition_parents > 0
    assert stats.ancestor_replication_factor >= 1.0
    with pytest.raises(ValueError, match="dense range"):
        gp.distributed.partition_graph(graph, owners + 1)
    with pytest.raises(ValueError, match="product"):
        gp.distributed.partition_graph(graph, owners, output_shape=(7, 7))
