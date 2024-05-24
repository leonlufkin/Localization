
"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

from localization.datasets.base import Dataset, ExemplarType
from localization.datasets.qdfs import NormalQDF, UniformQDF, BernoulliQDF, LaplaceQDF, AlgQDF
from localization.utils import build_DRT, build_gaussian_covariance, iterate_kron

from jax.scipy.special import erf
normal_cdf = lambda x: 0.5 * (1 + erf(x / jnp.sqrt(2)))

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))


class NortaDataset(Dataset):
  """Normal-to-Anything (NORTA) dataset."""

  def __init__(
    self,
    key: Array,
    xi: tuple[float] = (0.1, 1.1,),
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    adjust: tuple[float, float] | None = (-1.0, 1.0),
    marginal_qdf: callable = NormalQDF(),
    dim: int = 1,
    **kwargs
  ):
    """Initializes a `NortaDataset` instance."""
    super().__init__(
      key=key,
      num_exemplars=num_exemplars,
      num_dimensions=num_dimensions,
      )

    DRT = build_DRT(num_dimensions)
    DRT_ = DRT
    for _ in range(dim-1):
      DRT_ = jnp.kron(DRT_, DRT)
    self.DRT = DRT_

    def Z(g):
      return jnp.sqrt( (2/jnp.pi) * jnp.arcsin( (2*g**2) / (1 + (2*g**2)) ) )
      
    def generate_non_gaussian(key, xi, L, d):
      C = iterate_kron(build_gaussian_covariance(L, xi), d)
      evals = jnp.diag(self.DRT.T @ C @ self.DRT)
      sqrt_C = self.DRT @ jnp.diag(jnp.sqrt(jnp.maximum(evals, 0))) @ self.DRT.T
      
      z_id = jax.random.normal(key, (L ** d,))
      z = sqrt_C @ z_id
      x = marginal_qdf( normal_cdf( z ) )
      return x

    self.num_classes = len(xi)
    self.generate_xi = [None for _ in range(self.num_classes)]
    for i, xi in enumerate(xi):
      self.generate_xi[i] = jax.jit(
        jax.vmap(
          partial(generate_non_gaussian,
                  xi=xi, L=num_dimensions, d=dim,
                  ),
        )
      )
    
    # Adjust support
    # no adjustment
    if adjust is None:
      self.adjust = lambda x: x
    # adjust to fall within interval
    elif isinstance(adjust, tuple):
      z = 1 # Z(gain)
      self.adjust = jax.jit(
        jax.vmap(
          partial(lambda x, adjust: (x * z + 1) * (adjust[1] - adjust[0]) / 2 + adjust[0],
                  adjust=adjust)
          )
      )
    # apply function (should only take in x and return x)
    else:
      self.adjust = jax.jit(jax.vmap(adjust))

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  # @profile
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

    # index, n = self.process_index(index)
    
    if isinstance(index, slice):
      if index.stop is None:
        raise ValueError("Slice `index.stop` must be specified.")
      index = slice_to_array(index, len(self))
      n = index.shape[0]
    elif isinstance(index, int):
      # index = jnp.array([index])
      n = 1
    else:
      index = jnp.array(index)
      n = index.shape[0]
    
    class_keys = jax.random.split(self.key, self.num_classes)
    fold_in = jax.vmap(jax.random.fold_in, in_axes=(None, 0))
    keys = [ fold_in(class_key, index) for class_key in class_keys ]
    if isinstance(index, int):
      keys = [ jnp.expand_dims(keys_, axis=0) for keys_ in keys ]
        
    # generate exemplars and labels
    exemplars = jnp.concatenate([ self.generate_xi[i](key=keys_) for i, keys_ in enumerate(keys) ], axis=0)
    labels = jnp.concatenate([ i * jnp.ones(n) for i in range(self.num_classes) ], axis=0)
    # subsample
    perm = jax.random.permutation(class_keys[0], exemplars.shape[0])
    exemplars = exemplars[perm[:n]]
    labels = labels[perm[:n]]
    # adjust support
    exemplars = self.adjust(exemplars)

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    xi = (0.1, 5,)
    qdf = AlgQDF(6)
    print("xi:", xi)
    dataset = NortaDataset(key=key, xi=xi, marginal_qdf=qdf, num_dimensions=40, num_exemplars=10000)
    x, y = dataset[:100000]
    xx = (x.T @ x) / len(x)
    
    # Compute excess kurtosis
    x_ = x.flatten()
    print("Mean:", jnp.mean(x_)) # mean
    sigma2 = jnp.var(x_) # variance
    print("Variance:", sigma2) # variance
    k = jnp.mean(x_**4) / sigma2**2 - 3
    print("Excess kurtosis:", k) # kurtosis
    
    # Do the same but for the __sum/mean__ of all the entries
    x_ = jnp.sum(x, axis=1)
    print("Mean:", jnp.mean(x_)) # mean
    sigma2 = jnp.var(x_) # variance
    print("Variance:", sigma2) # variance
    k = jnp.mean(x_**4) / sigma2**2 - 3
    print("Excess kurtosis:", k) # kurtosis
    # Setting k=6 shows that the kurtosis of preactivations can be positive
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    im = ax1.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    ax2.plot(x[0])
    ax3.hist(x[:,0], bins=50, density=True)
    fig.savefig(f'../datasets/norta/covariance_{xi}_{qdf.__name__}.png')
    plt.close()
    
    