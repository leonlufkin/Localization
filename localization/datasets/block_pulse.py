"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from jax import random
from functools import partial

# export PYTHONPATH="${PYTHONPATH}:./" # <- NEED TO RUN THIS ON COMMAND LINE SO IT CAN FIND THE DATASET MODULE
from jaxnets.datasets.base import Dataset, ExemplarType
# from nets.datasets.base import ExemplarType

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))

def generate_pulse(key, xi, L):
  start = jax.random.randint(key, shape=(), minval=0, maxval=L//xi) * xi
  stop = start + xi
  l = jnp.arange(L)
  X = jnp.where(start <= l, 1., 0.) * jnp.where(l < stop, 1., 0.) + jnp.where(l < stop - L, 1., 0.)
  return X


class BlockDataset(Dataset):
  """
  Finite block dataset.
  All data-points are pre-generated and stored in memory.
  Hopefully, there are not too many.
  
  (The SinglePulseDataset is finite too, but this one is "more finite.")
  """

  def __init__(
    self,
    key: Array,
    xi1: int = 4,
    xi2: int = 3,
    class_proportion: float = 0.5,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    support: tuple[float, float] = (0.0, 1.0),
    **kwargs
  ):
    """
    Initializes a `FiniteDataset` instance.
    
    Parameters
    ----------
    key: jax PRNG Key
    xi1: length pulses in class 1
    xi2: length pulses in class -1
    class_proportion: proportion of examples to draw from class 1
    num_dimension: dimension of exemplars generated
    """
    # num_exemplars_ = 2 * num_dimensions
    
    super().__init__(
      key=key,  # TODO: Use a separate key.
      # num_exemplars=(num_dimensions // xi1 + num_dimensions // xi2)# * 2
      num_exemplars=num_exemplars
    )

    # Print the extra kwargs
    print("kwargs:", kwargs)

    # Construct all exemplars
    xi1 = int(xi1 * num_dimensions)
    xi2 = int(xi2 * num_dimensions)
    if num_dimensions % xi1 != 0 or num_dimensions % xi2 != 0:
      raise ValueError("num_dimensions must be divisible by xi1 and xi2")
    
    self.num_dimensions = num_dimensions
    tile_ = jnp.tile(jnp.arange(num_dimensions)[jnp.newaxis, :], (num_dimensions, 1))
    tile = (tile_ + jnp.arange(num_dimensions)[:, jnp.newaxis]) % num_dimensions
    exemplars1 = (jnp.where(tile < xi1, 1., 0.) + jnp.where(tile < xi1 - num_dimensions, 1., 0.))[::xi1] # this makes the pulses occupy disjoint blocks
    exemplars2 = (jnp.where(tile < xi2, 1., 0.) + jnp.where(tile < xi2 - num_dimensions, 1., 0.))[::xi2] # "                                          "
    exemplars_ = jnp.concatenate((exemplars1, exemplars2), axis=0)
    labels_ = jnp.concatenate((jnp.ones(len(exemplars1)), jnp.zeros(len(exemplars2))), axis=0)
    # self.exemplars = jnp.concatenate((exemplars_, 1-exemplars_), axis=0)
    # self.labels = jnp.concatenate((labels_, labels_), axis=0)
    self.exemplars = exemplars_
    self.labels = labels_
        
    # Adjust support
    adjust_support = lambda x: x * (support[1] - support[0]) + support[0]
    self.exemplars = adjust_support(self.exemplars)
    

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

    if isinstance(index, slice):
      if index.stop is None:
        raise ValueError("Slice `index.stop` must be specified.")
      index = slice_to_array(index, len(self))
    
    index = index % len(self.exemplars)
    exemplars = self.exemplars[index]
    labels = self.labels[index]

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    xi1, xi2 = 0.25, 0.1
    print("xi1, xi2:", xi1, xi2)
    dataset = BlockDataset(key=key, xi1=xi1, xi2=xi2, num_dimensions=40, support=(0.0, 1.0))
    print(len(dataset))
    x, y = dataset[:len(dataset)]
    xx = (x.T @ x) / len(x)
    import matplotlib.pyplot as plt
    im = plt.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    plt.savefig(f'../../thoughts/towards_gdln/figs/block_pulse_covariance_{xi1}_{xi2}.png')
    plt.close()
    
    im = plt.imshow(x[:200], cmap='gray')
    cbar = plt.colorbar(im)
    plt.savefig(f'../../thoughts/towards_gdln/figs/block_pulse_dataset_{xi1}_{xi2}.png')
    plt.close()
    
    
    
    
