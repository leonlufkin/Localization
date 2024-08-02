
"""A collection of inverse CDFs (aka quantile functions, or QDFs) for various distributions."""

import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import erf, erfinv

class QDF:
    def __init__(self, qdf, cdf, key=jax.random.PRNGKey(0)):
        self.qdf = qdf
        self.cdf = cdf
        self.key = key
        self.__name__ = self.__class__.__name__
    
    def __call__(self, x):
        return self.qdf(x)
    
    def cdf(self, x):
        return self.cdf(x)
    
class NormalQDF(QDF):
    def __init__(self):
        normal_cdf = lambda x: 0.5 * (1 + erf(x / jnp.sqrt(2)))
        normal_qdf = lambda x: jnp.sqrt(2) * erfinv(2 * x - 1)
        super().__init__(normal_qdf, normal_cdf)
        
class UniformQDF(QDF):
    def __init__(self):
        uniform_cdf = lambda x: max(0, min(1, (x+jnp.sqrt(3))/(2*jnp.sqrt(3)) ))
        uniform_qdf = lambda x: jnp.sqrt(3) * (2 * x - 1)
        super().__init__(uniform_qdf, uniform_cdf)
        
class BernoulliQDF(QDF):
    def __init__(self):
        bernoulli_cdf = lambda x: 0.5 * (1 + jnp.sign(x))
        bernoulli_qdf = lambda x: erf( 100 * (x-0.5) ) # needs to be differentiable
        super().__init__(bernoulli_qdf, bernoulli_cdf)

class LaplaceQDF(QDF):
    def __init__(self):
        laplace_cdf = lambda x: 0.5 * (1 + jnp.sign(x) * (1 - jnp.exp(-jnp.abs(x)/jnp.sqrt(2))))
        laplace_qdf = lambda x: jnp.sign(x - 0.5) * jnp.log(1 - 2 * jnp.abs(x - 0.5)) / jnp.sqrt(2)
        super().__init__(laplace_qdf, laplace_cdf)

class AlgQDF(QDF):
    def __init__(self, k=5):
        self.k = k
        alg_qdf = lambda x: (2*x-1) / ( 1 - jnp.abs(2*x-1)**k )**(1/k)
        alg_cdf = lambda x: 0.5 * (1 + x / (1 + jnp.abs(x)**k)**(1/k))
        # Standardize (no explicit form for variance)
        p = jax.random.uniform(jax.random.PRNGKey(42), (1000000,))
        x = alg_qdf(p)
        std = jnp.std(x)
        print(f'Approximate standard deviation: {std}')
        # Save standardized QDF and CDF
        super().__init__(lambda x: alg_qdf(x) / std, lambda x: alg_cdf(x * std))
        self.__name__ = f'AlgQDF{k}'


