"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from localization.datasets.base import Dataset, ExemplarType
from localization.utils import build_DRT, build_gaussian_covariance, build_sine_covariance, iterate_kron
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


class NonlinearGPDataset(Dataset):
  """Gaussian control for Nonlinear Gaussian Process dataset."""

  def __init__(
    self,
    key: Array,
    xi: tuple[float] = (0.1, 1.1,),
    gain: float = 1.,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    adjust: tuple[float, float] | None = (-1.0, 1.0),
    dim: int = 1,
    **kwargs
  ):
    """Initializes a `NonlinearGPDataset` instance."""
    super().__init__(
      key=key,  # TODO(eringrant): Use a separate key.
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
      
    def generate_non_gaussian(key, xi, L, g, d):
      C = iterate_kron(build_gaussian_covariance(L, xi), d)
      # C = iterate_kron(build_sine_covariance(L, xi), d)
      evals = jnp.diag(self.DRT.T @ C @ self.DRT)
      sqrt_C = self.DRT @ jnp.diag(jnp.sqrt(jnp.maximum(evals, 0))) @ self.DRT.T
      
      z_id = jax.random.normal(key, (L ** d,))
      z = sqrt_C @ z_id
      x = gain_function(g * z) / (Z(g) ** 2)
      return x

    self.num_classes = len(xi)
    self.generate_xi = [None for _ in range(self.num_classes)]
    for i, xi in enumerate(xi):
      self.generate_xi[i] = jax.jit(
        jax.vmap(
          partial(generate_non_gaussian,
                  xi=xi, L=num_dimensions, g=gain, d=dim,
                  ),
        )
      )
    
    # Adjust support
    # no adjustment
    if adjust is None:
      self.adjust = lambda x: x
    # adjust to fall within interval
    elif isinstance(adjust, tuple):
      z = Z(gain)
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
      # TODO(leonl): Deal with the case where `index.stop` is `None`.
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
    xi, gain = (5, 3, 1), 3
    print("xi, gain:", xi, gain)
    dataset = NonlinearGPDataset(key=key, xi=xi, gain=gain, num_dimensions=40, num_exemplars=10000)
    x, y = dataset[:10000]
    xx = (x.T @ x) / len(x)
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im = ax1.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    ax2.plot(x[0])
    fig.savefig(f'../datasets/nlgp/covariance_{xi}_{gain}.png')
    plt.close()
    
    