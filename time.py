
from time import time

import jax
from jax import numpy as jnp
from functools import partial

# Initialize parameters
N = 100
key = jax.random.PRNGKey(0)
mean = jnp.zeros((N,), dtype=jnp.float64)
var = jnp.array([[jnp.exp(-100 * (row/N - col/N)**2 - 10) for col in range(N)] for row in range(N)])

# JIT compile the sampling functions
mvn_cholesky = jax.jit(lambda key: jax.random.multivariate_normal(key, mean, var, method='cholesky'))
mvn_svd = jax.jit(lambda key: jax.random.multivariate_normal(key, mean, var, method='svd'))

# Warm up the JIT
jax.block_until_ready(mvn_cholesky(key))
jax.block_until_ready(mvn_svd(key))

# Time Cholesky method
start = time()
jax.block_until_ready(mvn_cholesky(key))
cholesky_time = time() - start

# Time SVD method
start = time()
jax.block_until_ready(mvn_svd(key))
svd_time = time() - start

print(f"Cholesky method took {cholesky_time:.6f} seconds.")
print(f"SVD method took {svd_time:.6f} seconds.")

