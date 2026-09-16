import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.sharding import Mesh

import graphgp as gp

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def vector_setup():
    # A small valid DAG with unequal batches and a final batch of just one node.
    points = jnp.arange(27, dtype=jnp.float64).reshape(9, 3) / 30
    neighbors = jnp.array([[0, 1], [1, 2], [3, 4], [0, 3], [2, 4], [5, 7]])
    indices = jnp.array([8, 0, 6, 1, 5, 2, 7, 3, 4])
    graph = gp.Graph(points, neighbors, (3, 5, 8, 9), indices)
    gp.check_graph(graph)
    covariance = gp.extras.matern_kernel(p=0, variance=1.0, cutoff=0.3, r_min=1e-4, r_max=3.0, n_bins=128, jitter=1e-5)
    component_cov = jnp.array([[1.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 0.8]])
    cov_elem = gp.make_separable_vector_covariance(covariance, component_cov)
    xi = jr.normal(jr.key(13), (9, 3))
    return graph, covariance, component_cov, cov_elem, xi


def test_block_covariance_layout(vector_setup):
    graph, covariance, component_cov, cov_elem, _ = vector_setup
    a, b = graph.points[:2], graph.points[2:5]
    expected = jnp.kron(gp.compute_cov_matrix(covariance, a, b), component_cov)
    np.testing.assert_allclose(gp.compute_cov_matrix_elem(cov_elem, a, b), expected, atol=1e-12)


@pytest.mark.parametrize("fast_jit", [False, True])
def test_vector_matches_independent_scalar_fields(vector_setup, fast_jit):
    graph, covariance, component_cov, cov_elem, xi = vector_setup
    independent = jax.vmap(lambda z: gp.generate(graph, covariance, z, fast_jit=False), in_axes=1, out_axes=1)(xi)
    expected = independent @ jnp.linalg.cholesky(component_cov).T
    actual = jax.jit(lambda z: gp.generate_vector(graph, cov_elem, z, fast_jit=fast_jit))(xi)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_vector_inverse_and_logdet(vector_setup):
    graph, _, _, cov_elem, xi = vector_setup
    generate = jax.jit(lambda z: gp.generate_vector(graph, cov_elem, z))
    values = generate(xi)
    recovered = jax.jit(lambda v: gp.generate_vector_inv(graph, cov_elem, v))(values)
    np.testing.assert_allclose(recovered, xi, rtol=1e-10, atol=1e-10)
    jacobian = jax.jacfwd(generate)(jnp.zeros_like(xi)).reshape(27, 27)
    expected = jnp.linalg.slogdet(jacobian)[1]
    actual = jax.jit(lambda: gp.generate_vector_logdet(graph, cov_elem))()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    direction = jr.normal(jr.key(4), xi.shape)
    np.testing.assert_allclose(jax.jvp(generate, (xi,), (direction,))[1], generate(direction), atol=1e-10)
    cotangent = jax.grad(lambda z: jnp.sum(generate(z) ** 2))(xi)
    np.testing.assert_allclose(cotangent.reshape(-1), 2 * jacobian.T @ values.reshape(-1), atol=1e-10)


def test_dense_vector_inverse_and_logdet(vector_setup):
    graph, _, _, cov_elem, xi = vector_setup
    values = gp.generate_dense_vector(graph.points, cov_elem, xi)
    np.testing.assert_allclose(gp.generate_dense_vector_inv(graph.points, cov_elem, values), xi, atol=1e-10)
    matrix = gp.compute_cov_matrix_elem(cov_elem, graph.points, graph.points)
    np.testing.assert_allclose(
        gp.generate_dense_vector_logdet(graph.points, cov_elem), jnp.linalg.slogdet(matrix)[1] / 2
    )


def test_magnetic_field_generation(vector_setup):
    graph, _, _, _, xi = vector_setup
    cov_elem = gp.extras.make_div_free_rbf_covariance(ell=0.15, periodic=False)
    fast = jax.jit(lambda z: gp.generate_vector(graph, cov_elem, z))(xi)
    slow = gp.generate_vector(graph, cov_elem, xi, fast_jit=False)
    assert bool(jnp.all(jnp.isfinite(fast)))
    np.testing.assert_allclose(fast, slow, atol=1e-10)
    np.testing.assert_allclose(gp.generate_vector_inv(graph, cov_elem, fast), xi, atol=1e-10)
    # The analytic tensor is solenoidal; a finite sampled realization remains approximate.
    x, y = graph.points[0], graph.points[1]
    derivative = jax.jacfwd(lambda z: cov_elem(z, y))(x)
    np.testing.assert_allclose(jnp.einsum("iji->j", derivative), 0.0, atol=1e-10)


@pytest.mark.parametrize("entry_point", ["generate", "generate_recompute"])
def test_distributed_rejects_vector_fields(vector_setup, entry_point):
    graph, _, _, cov_elem, xi = vector_setup
    plan = gp.distributed.partition_graph(graph, np.arange(9) % 1)
    kwargs = {"mesh": Mesh(np.asarray(jax.devices()[:1]), ("space",))} if entry_point == "generate" else {}
    with pytest.raises(NotImplementedError, match="single device"):
        getattr(gp.distributed, entry_point)(plan, cov_elem, xi, **kwargs)


def test_vector_shape_validation(vector_setup):
    graph, _, _, cov_elem, xi = vector_setup
    with pytest.raises(ValueError, match="shape"):
        gp.generate_vector(graph, cov_elem, xi[:, 0])
    with pytest.raises(ValueError, match="Leading dimension"):
        gp.generate_vector(graph, cov_elem, xi[:-1])
    with pytest.raises(ValueError, match="component"):
        gp.generate_vector(graph, cov_elem, xi[:, :2])
