"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from localization.datasets.base import Dataset, ExemplarType
from localization.utils import build_DRT
# from nets.datasets.base import ExemplarType

from jax.scipy.special import erf as gain_function
from localization.utils import build_gaussian_covariance


def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))


class TDataset(Dataset):
  """Dataset of exemplars and labels from a T distribution."""

  def __init__(
    self,
    key: Array,
    xi: tuple[float] = (0.1, 1.1),
    df: float = 3.,
    class_proportion: float = 0.5,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    support: tuple[float, float] = (-1.0, 1.0),
    **kwargs
  ):
    """Initializes a `NonlinearGPDataset` instance."""
    super().__init__(
      key=key,  # TODO(eringrant): Use a separate key.
      num_exemplars=num_exemplars,
      )

    self.num_dimensions = num_dimensions 
    self.DRT = build_DRT(num_dimensions)
    
    if df <= 2:
      raise ValueError("Degrees of freedom must be greater than 2.")
    
    # use prior to November 23, 2023
    def generate_t_old(key, xi, L, df):
      C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
      C = jnp.minimum(C, L - C)
      C = jnp.exp(-C ** 2 / (xi ** 2))
      C = (df-2) / df * C # rescaling so that covariance is C
      x = jax.random.multivariate_t(key, jnp.zeros(L), C, df, method="svd") # FIXME: using svd for numerical stability, breaks if xi > 2.5 ish
      return x
    
    def generate_t(key, xi, L, df):
      C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
      C = jnp.minimum(C, L - C)
      C = jnp.exp(-C ** 2 / (xi ** 2))
      C = (df-2) / df * C # rescaling so that covariance is C
      evals = jnp.diag(self.DRT.T @ C @ self.DRT)
      sqrt_C = self.DRT @ jnp.diag(jnp.sqrt(jnp.maximum(evals, 0))) @ self.DRT.T
      
      normal_key, chi_key = jax.random.split(key)
      z_id = jax.random.normal(normal_key, (L,))
      normal_samples = sqrt_C @ z_id
      chi2_samples = jax.random.chisquare(chi_key, df)
      x = normal_samples / jnp.sqrt(chi2_samples / df)
      return x
    
    # self.generate_xi1 = jax.jit(
    #   jax.vmap(
    #     partial(generate_t,
    #             xi=xi1, L=num_dimensions, df=df,
    #             ),
    #   )
    # )
    
    # self.generate_xi2 = jax.jit(
    #   jax.vmap(
    #     partial(generate_t,
    #             xi=xi2, L=num_dimensions, df=df,
    #             ),
    #   )
    # )
    
    self.num_classes = len(xi)
    self.generate_xi = [None for _ in range(self.num_classes)]
    for i, xi in enumerate(xi):
      self.generate_xi[i] = jax.jit(
        jax.vmap(
          partial(generate_t,
                  xi=xi, L=num_dimensions, df=df,
                  ),
        )
      )
    
    # Adjust support
    # z = Z(gain)
    # self.adjust = jax.jit(
    #   jax.vmap(
    #     partial(lambda x, support: (x * z + 1) * (support[1] - support[0]) / 2 + support[0],
    #             support=support)
    #     )
    # )
    self.adjust = lambda x: x

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  # @profile
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

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

    # keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, index)
    # if isinstance(index, int):
    #   keys = jnp.expand_dims(keys, axis=0)
        
    # # generate xi1 and xi2
    # xi1 = self.generate_xi1(key=keys)
    # xi2 = self.generate_xi2(key=keys)
    # # concatenate xi1 and xi2
    # exemplars = jnp.concatenate((xi1, xi2), axis=0)
    # labels = jnp.concatenate((jnp.ones(n), jnp.zeros(n)), axis=0)
    # # subsample
    # perm = jax.random.permutation(keys[0], exemplars.shape[0])
    # exemplars = exemplars[perm[:n]]
    # labels = labels[perm[:n]]
    
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
    # exemplars = self.adjust(exemplars)

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    xi, df = (5, 5), 5
    print("xi, gain:", xi, df)
    dataset = TDataset(key=key, xi=xi, df=df, num_dimensions=40, num_exemplars=100000)
    x, y = dataset[:100000]
    xx = (x.T @ x) / len(x)
    
    # examining covariance
    import matplotlib.pyplot as plt
    im = plt.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    plt.suptitle("Empirical covariance")
    plt.savefig(f'../../thoughts/distributions/figs/multi_t{xi1}_{xi2}_{df}.png')
    plt.close()
    
    # plotting difference between empirical and theoretical covariance
    # note the difference is blotchy, suggesting issues with how we are generating randomness
    im = plt.imshow(xx - build_gaussian_covariance(40, xi1))
    cbar = plt.colorbar(im)
    plt.suptitle("Difference between empirical and theoretical covariance")
    plt.savefig(f'../../thoughts/distributions/figs/multi_t_diff{xi1}_{xi2}_{df}.png')
    plt.close()
    
    
    