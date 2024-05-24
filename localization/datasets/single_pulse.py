"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from jax import random
from functools import partial

# export PYTHONPATH="${PYTHONPATH}:./" # <- NEED TO RUN THIS ON COMMAND LINE SO IT CAN FIND THE DATASET MODULE
from localization.datasets.base import Dataset, ExemplarType
# from nets.datasets.base import ExemplarType

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))

def generate_pulse(key, xi_lower, xi_upper, L):
  start_key, stop_key = jax.random.split(key, 2)
  start = jax.random.randint(start_key, shape=(), minval=0, maxval=L)
  stop = start + jax.random.randint(stop_key, shape=(), minval=xi_lower, maxval=xi_upper) + 1
  l = jnp.arange(L)
  X = jnp.where(start <= l, 1., 0.) * jnp.where(l < stop, 1., 0.) + jnp.where(l < stop - L, 1., 0.)
  return X


class SinglePulseDataset(Dataset):
  """Single Pulse Dataset."""

  def __init__(
    self,
    key: Array,
    xi1: tuple[float, float] = (0.1, 0.2),
    xi2: tuple[float, float] = (0.0, 0.1),
    class_proportion: float = 0.5,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    support: tuple[float, float] = (0.0, 1.0),
    **kwargs
  ):
    """
    Initializes a `SinglePulseDataset` instance.
    
    Parameters
    ----------
    key: jax PRNG Key
    xi1: lower/upper bound of interval to sample class 1's length scale from (inclusive/exclusive)
    xi2: lower/upper bound of interval to sample class -1's length scale from (inclusive/exclusive)
    class_proportion: proportion of examples to draw from class 1
    num_dimension: dimension of exemplars generated
    num_exemplars: number of distinct exemplars to generate
    """
    super().__init__(
      key=key,  # TODO(eringrant): Use a separate key.
      num_exemplars=num_exemplars,
    )

    # Print the extra kwargs
    print("kwargs:", kwargs)

    self.num_dimensions = num_dimensions
    
    # Compile a function for sampling exemplars at `Dataset.__init__`.
    xi1_lower, xi1_upper = int(xi1[0] * num_dimensions), int(xi1[1] * num_dimensions)
    xi2_lower, xi2_upper = int(xi2[0] * num_dimensions), int(xi2[1] * num_dimensions)
    
    self.generate_xi1 = jax.jit(
      jax.vmap(
        partial(generate_pulse, 
                xi_lower=xi1_lower,
                xi_upper=xi1_upper,
                L=num_dimensions)
        )
    )
    
    self.generate_xi2 = jax.jit(
      jax.vmap(
        partial(generate_pulse, 
                xi_lower=xi2_lower,
                xi_upper=xi2_upper,
                L=num_dimensions)
        )
    )
    
    # Adjust support
    self.adjust_support = jax.jit(
      jax.vmap(
        partial(lambda x, support: x * (support[1] - support[0]) + support[0],
                support=support)
        )
    )

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
    else:
      index = index
    n = index.shape[0]

    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, index)
    if isinstance(index, int):
        keys = jnp.expand_dims(keys, axis=0)

    # generate xi1 and xi2
    xi1 = self.generate_xi1(key=keys)
    xi2 = self.generate_xi2(key=keys)
    # concatenate xi1 and xi2
    exemplars = jnp.concatenate((xi1, xi2), axis=0)
    labels = jnp.concatenate((jnp.ones(n), jnp.zeros(n)), axis=0)
    # subsample
    perm = jax.random.permutation(keys[0], exemplars.shape[0])
    exemplars = exemplars[perm[:n]]
    labels = labels[perm[:n]] 
    # randomly flip sign of exemplars
    sign = jax.random.bernoulli(keys[0], p=0.5, shape=(n,1))
    exemplars = jnp.where(sign, exemplars, 1-exemplars)
    # adjust support of exemplars
    exemplars = self.adjust_support(exemplars)

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    xi1, xi2 = (0.4, 0.5), (0.0, 0.1)
    dataset = SinglePulseDataset(key=key, xi1=xi1, xi2=xi2, num_dimensions=40, num_exemplars=100000)
    x, y = dataset[:100000]
    xx = (x.T @ x) / len(x)
    import matplotlib.pyplot as plt
    im = plt.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    plt.savefig(f'../thoughts/towards_gdln/figs/sp_covariance_({xi1[0]},{xi1[1]}]_({xi2[0]},{xi2[1]}].png')
    plt.close()
    
    # print(dataset[:1000][0].min())
    # print(dataset[:1000][0].max())
    # from nets import samplers
    # sampler = samplers.EpochSampler(
    #     key=key,
    #     dataset=dataset,
    #     num_epochs=1,
    # )
    # print(len(sampler))
    # print(sampler[:1][0])
    # print(sampler[:1][1])
    # print(sampler[:1][0].shape, sampler[:1][1].shape)
