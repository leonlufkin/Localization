from functools import partial

import numpy as np
import scipy.stats as osp_stats

from jax import lax
from jax import numpy as jnp
from jax._src.numpy.util import _wraps, promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import gammaln


@_wraps(osp_stats.multivariate_t.logpdf, update_doc=False, lax_description="""
In the JAX version, the `allow_singular` argument is not implemented.
""")
def logpdf(x: ArrayLike, loc: ArrayLike, shape: ArrayLike, df: int, allow_singular: None = None) -> ArrayLike:
  if allow_singular is not None:
    raise NotImplementedError("allow_singular argument of multivariate_normal.logpdf")
  x, loc, shape, df = promote_dtypes_inexact(x, loc, shape, df)
  p = loc.shape[-1] if loc.shape else 1
  
  c = gammaln((df + p) / 2) - gammaln(df / 2) - p / 2 * jnp.log(df * np.pi) # does not include covariance term
  
  if not loc.shape:
    return (c - jnp.log(shape) / 2 
            - (df + p) / 2 * jnp.log1p((x - loc)**2 / (df * shape)))
      
  else:
    if not np.shape(shape):
      y = x - loc
      return (c - p * jnp.log(shape) / 2 
            - (df + p) / 2 * jnp.log1p(jnp.einsum('...i,...i->...', y, y)/ (df * shape)))
    else:
      if shape.ndim < 2 or shape.shape[-2:] != (p, p):
        raise ValueError("multivariate_normal.logpdf got incompatible shapes")
      L = lax.linalg.cholesky(shape)
      y = jnp.vectorize(
        partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
        signature="(n,n),(n)->(n)"
      )(L, x - loc)
      return (c - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1)
            - (df + p) / 2 * jnp.log1p(jnp.einsum('...i,...i->...', y, y) / df))


@_wraps(osp_stats.multivariate_t.pdf, update_doc=False)
def pdf(x: ArrayLike, loc: ArrayLike, shape: ArrayLike, df: int) -> Array:
    return lax.exp(logpdf(x, loc, shape, df))
