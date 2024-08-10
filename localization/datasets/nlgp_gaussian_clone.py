"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from jaxnets.datasets.base import Dataset, ExemplarType
from localization.utils import build_DRT, build_non_gaussian_covariance, iterate_kron
# from nets.datasets.base import DatasetSplit
# from nets.datasets.base import ExemplarType
# from nets.datasets.base import ExemplarLabeling
# from nets.datasets.base import HoldoutClassLabeling

from jax.scipy.special import erf as gain_function

#from line_profiler import LineProfiler
#
#profiler = LineProfiler()
#def profile(func):
#   def inner(*args, **kwargs):
#       profiler.add_function(func)
#       profiler.enable_by_count()
#       return func(*args, **kwargs)
#   return inner

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))

class NLGPGaussianCloneDataset(Dataset):
  """Nonlinear Gaussian Process dataset."""

  def __init__(
    self,
    key: Array,
    # xi1: float = 0.1,
    # xi2: float = 1.1,
    # xi=
    gain: float = 1.,
    class_proportion: float = 0.5,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    support: tuple[float, float] = (-1.0, 1.0),
    dim: int = 1,
    **kwargs
  ):
    """Initializes a `NonlinearGPDataset` instance."""
    super().__init__(
      key=key,  # TODO: Use a separate key.
      num_exemplars=num_exemplars,
      )

    # self.exemplar_noise_scale = exemplar_noise_scale
    self.num_dimensions = num_dimensions 
    self.DRT = iterate_kron(build_DRT(num_dimensions), dim)

    # Compile a function for sampling exemplars at `Dataset.__init__`.
    def Z(g):
      return jnp.sqrt( (2/jnp.pi) * jnp.arcsin( (2*g**2) / (1 + (2*g**2)) ) )
    
    # old way, used prior to Nov 2, 2023
    def generate_gaussian_old(key, xi, L, g):
      C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
      C = jnp.minimum(C, L - C)
      C = jnp.exp(-C ** 2 / (xi ** 2))
      Sigma = (2/jnp.pi) / (Z(g) ** 2) * jnp.arcsin( (2*g**2) / (1 + (2*g**2)) * C )
      x = jax.random.multivariate_normal(key, jnp.zeros(L), Sigma, method="svd")
      return x

    # new way, equivalent in distribution but lets us make more direct comparisons to Gaussian clone
    # randomness will be different, so it may yield slightly different results than before
    def generate_gaussian(key, xi, L, g, d):
      # C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
      # C = jnp.minimum(C, L - C)
      # C = jnp.exp(-C ** 2 / (xi ** 2))
      # Sigma = (2/jnp.pi) / (Z(g) ** 2) * jnp.arcsin( (2*g**2) / (1 + (2*g**2)) * C )
      Sigma = iterate_kron(build_non_gaussian_covariance(L, xi, g), d)
      evals = jnp.diag(self.DRT.T @ Sigma @ self.DRT)
      sqrt_Sigma = self.DRT @ jnp.diag(jnp.sqrt(evals)) @ self.DRT.T
      
      z_id = jax.random.normal(key, (L ** d,))
      x = sqrt_Sigma @ z_id
      return x
    
    # self.generate_xi1 = jax.jit(
    #   jax.vmap(
    #     partial(generate_gaussian,
    #             xi=xi1, L=num_dimensions, g=gain, d=dim,
    #             ),
    #   )
    # )
    # self.generate_xi2 = jax.jit(
    #   jax.vmap(
    #     partial(generate_gaussian,
    #             xi=xi2, L=num_dimensions, g=gain, d=dim,
    #             ),
    #   )
    # )
    
    self.num_classes = len(xi)
    self.generate_xi = [None for _ in range(self.num_classes)]
    for i, xi in enumerate(xi):
      self.generate_xi[i] = jax.jit(
        jax.vmap(
          partial(generate_gaussian,
                  xi=xi, L=num_dimensions, g=gain, d=dim,
                  ),
        )
      )
    
    # Adjust support
    # z = Z(gain)
    # self.adjust_support = jax.jit(
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
    # # adjust support
    # exemplars = self.adjust(exemplars)

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
    xi, gain = (5, 1,), 3
    print("xi, gain:", xi, gain)
    from nonlinear_gp import NonlinearGPDataset
    dataset = NonlinearGPDataset(key=key, xi=xi, gain=gain, num_dimensions=40, num_exemplars=10000)
    control_dataset = NLGPGaussianCloneDataset(key=key, xi=xi, gain=gain, num_dimensions=40, num_exemplars=10000)
    
    # original
    x, y = dataset[:10000]
    xx = (x.T @ x) / len(x)
    print(xx)
    
    # control
    x_, y_ = control_dataset[:10000]
    xx_ = (x_.T @ x_) / len(x_)
    print(xx_)
    
    # difference
    print( xx / xx_ )
    print( jnp.linalg.norm(xx - xx_) )

    

