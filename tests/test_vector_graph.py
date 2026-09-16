import jax.numpy as jnp
import numpy as np

import graphgp as gp


def test_periodic_predecessors():
    points = jnp.array([[0.01], [0.99], [0.25], [0.75], [0.45], [0.55]])
    tree, split_dims, _ = gp.build_tree(points)
    actual = gp.query_preceding_neighbors(tree, split_dims, n0=2, k=2, periodic=True, boxsize=1.0)
    for i, neighbors in enumerate(np.asarray(actual), start=2):
        delta = np.asarray(tree[:i] - tree[i])
        delta = delta - np.round(delta)
        expected = np.argsort(np.sum(delta ** 2, axis=1))[:2]
        assert set(neighbors) == set(expected)
    gp.check_graph(gp.build_graph(points, n0=2, k=2, periodic=True, boxsize=1.0))


def test_interpolation_vector_keeps_source_values():
    source = jnp.array([[0.01], [0.45], [0.75]])
    target = jnp.array([[0.99], [0.25], [0.55]])
    graph = gp.build_interpolation_graph(source, target, k=2, periodic=True, boxsize=1.0)
    gp.check_graph(graph)
    np.testing.assert_allclose(graph.points[:3], source)
    covariance = gp.extras.matern_kernel(
        p=0, variance=1.0, cutoff=0.3, r_min=1e-4, r_max=2.0, n_bins=64, jitter=1e-5
    )
    cov_elem = gp.make_separable_vector_covariance(covariance, jnp.eye(2))
    initial = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    args = (graph.points, graph.neighbors, graph.offsets, cov_elem, initial, jnp.zeros((3, 2)))
    fast = gp.refine_vector(*args)
    slow = gp.refine_vector(*args, fast_jit=False)
    assert bool(jnp.all(jnp.isfinite(fast)))
    np.testing.assert_allclose(fast[:3], initial)
    np.testing.assert_allclose(fast, slow, atol=1e-10)
