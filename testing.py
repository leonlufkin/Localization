import pandas as pd
from functools import partial
from time import time
import jax
from jax import numpy as jnp

def generate_parameters(N):
    mean = jnp.zeros((N,), dtype=jnp.float32)
    cov = jnp.array([[jnp.exp(-100 * (row/N - col/N)**2 - 10) for col in range(N)] for row in range(N)])
    return mean, cov

def contains_nans(sample):
    return bool(jnp.any(jnp.isnan(sample)))

Ns = (2**i for i in range(8))
results = []

def analyze(func, key):
    # Warm-up.
    func(key)

    # Time.
    start = time()
    sample = func(key).block_until_ready()
    time = time() - start

    nans = contains_nans(sample)

    return str(func), time, nans



for N in Ns:
    mean, cov = generate_parameters(N)
    key = jax.random.PRNGKey(N)

    # JIT compile the sampling functions
    mvn_cholesky = jax.jit(partial(jax.random.multivariate_normal, mean=mean, cov=cov, method='cholesky'))
    mvn_svd = jax.jit(partial(jax.random.multivariate_normal, mean=mean, cov=cov, method='svd'))

    ### ADD Here

    results.append({
        'dimension': N,
        ### ADD Here
    })

df = pd.DataFrame(results)
print(df)
