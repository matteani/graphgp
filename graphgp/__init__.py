# graphgp/__init__.py
"""
Public GraphGP API.

Exports scalar & vector-valued generation / refine utilities and helper modules.
"""

from .graph import build_graph, Graph
from . import extras
from . import tree

# --- scalar operations (existing API) ---
from .refine import (
    compute_cov_matrix,
    generate,
    generate_dense,
    generate_inv,
    generate_dense_inv,
    generate_logdet,
    generate_dense_logdet,
    refine,
    refine_inv,
    refine_logdet,
)

# --- vector-valued extensions (added) ---
from .refine import (
    compute_cov_matrix_elem,
    make_separable_vector_covariance,
    generate_vector,
    generate_vector_inv,
    generate_vector_logdet,
    generate_dense_vector,
    generate_dense_vector_inv,
    generate_dense_vector_logdet,
    refine_vector,
    refine_vector_inv,
    refine_vector_logdet,
)

# Public names
__all__ = [
    # graph builder / helpers
    "build_graph",
    "Graph",
    "extras",
    "tree",
    # scalar
    "compute_cov_matrix",
    "generate",
    "generate_dense",
    "generate_inv",
    "generate_dense_inv",
    "generate_logdet",
    "generate_dense_logdet",
    "refine",
    "refine_inv",
    "refine_logdet",
    # vector-valued
    "compute_cov_matrix_elem",
    "make_separable_vector_covariance",
    "generate_vector",
    "generate_vector_inv",
    "generate_vector_logdet",
    "generate_dense_vector",
    "generate_dense_vector_inv",
    "generate_dense_vector_logdet",
    "refine_vector",
    "refine_vector_inv",
    "refine_vector_logdet",
]
