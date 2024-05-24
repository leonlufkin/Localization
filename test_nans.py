import numpy as np
import pandas as pd

import jax.numpy as jnp
import jax
from jax import jit

sizes = (10 * 2**i for i in range(9))
numpy_nans = []
jax_nans = []


def generate_matrix(D):
    C = jnp.abs(
        jnp.tile(jnp.arange(D)[:, jnp.newaxis], (1, D))
        - jnp.tile(jnp.arange(D), (D, 1))
    )
    C = jnp.float32(C)  # Using `float64` prevents underflow.
    return jnp.exp(-(C**2) / (3.3**2))


@jit
def jax_svd(matrix):
    return jnp.linalg.svd(matrix)


@jit
def jax_cholesky(matrix):
    return jnp.linalg.cholesky(matrix)


def numpy_svd(matrix):
    return np.linalg.svd(matrix)


def numpy_cholesky(matrix):
    return np.linalg.cholesky(matrix)


isnans = []
for D in sizes:
    matrix_jax = generate_matrix(D)
    matrix_numpy = np.array(matrix_jax)

    # JAX.
    u_jax, s_jax, vh_jax = jax_svd(matrix_jax)
    jax_svd_isnan = any(
        (
            jnp.any(jnp.isnan(u_jax)),
            jnp.any(jnp.isnan(s_jax)),
            jnp.any(jnp.isnan(vh_jax)),
        )
    )

    l_jax = jax_cholesky(matrix_jax)
    jax_cholesky_isnan = bool(jnp.any(jnp.isnan(l_jax)))

    # NumPy
    u_numpy, s_numpy, vh_numpy = numpy_svd(matrix_numpy)
    numpy_svd_isnan = any(
        (
            jnp.any(jnp.isnan(u_numpy)),
            jnp.any(jnp.isnan(s_numpy)),
            jnp.any(jnp.isnan(vh_numpy)),
        )
    )

    l_numpy = numpy_cholesky(matrix_numpy)
    numpy_cholesky_isnan = bool(np.any(np.isnan(l_numpy)))

    isnans += [
        {
            "dimension": D,
            "JAX SVD": jax_svd_isnan,
            "JAX Cholesky": jax_cholesky_isnan,
            "NumPy SVD": numpy_svd_isnan,
            "NumPy Cholesky": numpy_cholesky_isnan,
        }
    ]


print("### NaNs:")
print(pd.DataFrame(isnans))
