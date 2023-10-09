"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from jax import random
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from localization.datasets.base import Dataset
# from nets.datasets.base import DatasetSplit
from nets.datasets.base import ExemplarType
# from nets.datasets.base import ExemplarLabeling
# from nets.datasets.base import HoldoutClassLabeling

from jax.scipy.special import erf as gain_function

from line_profiler import LineProfiler
#
profiler = LineProfiler()
def profile(func):
   def inner(*args, **kwargs):
       profiler.add_function(func)
       profiler.enable_by_count()
       return func(*args, **kwargs)
   return inner

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))


class NonlinearGPDataset(Dataset):
  """Nonlinear Gaussian Process dataset."""

  def __init__(
    self,
    key: Array,
    xi1: float = 0.1,
    xi2: float = 1.1,
    gain: float = 1.,
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

    # self.exemplar_noise_scale = exemplar_noise_scale
    self.num_dimensions = num_dimensions 

    # Compile a function for sampling exemplars at `Dataset.__init__`.
    def Z(g):
        return jnp.sqrt( (2/jnp.pi) * jnp.arcsin( (g**2) / (1 + (g**2)) ) )
    
    # @profile
    def generate_non_gaussian(key, xi, L, g):
        C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
        C = jnp.exp(-C ** 2 / (xi ** 2))
        z = jax.random.multivariate_normal(key, jnp.zeros(L), C, method="svd") # FIXME: using svd for numerical stability, breaks if xi > 2.5 ish
        x = gain_function(g * z) / Z(g)
        return x

    # @profile
    def generate_non_gaussian_branching(key, class_proportion, L, g, xi1, xi2):
      label_key, exemplar_key = jax.random.split(key, 2)
      label = jax.random.bernoulli(label_key, p=class_proportion)
      xi = jnp.where(label, xi1, xi2)
      # label = 1.
      # xi = xi1
      exemplar = generate_non_gaussian(exemplar_key, xi, L, g)
      label = 2 * jnp.float32(label) - 1
      return exemplar, label

    self.generate_xi = jax.jit(
      jax.vmap(
            partial(generate_non_gaussian_branching, 
                    class_proportion=class_proportion,
                    L=num_dimensions, g=gain,
                    xi1=xi1, xi2=xi2
                    ),
        )
    )
    
    self.generate_xi1 = jax.jit(
      jax.vmap(
        partial(generate_non_gaussian,
                xi=xi1, L=num_dimensions, g=gain,
                ),
      )
    )
    self.generate_xi2 = jax.jit(
      jax.vmap(
        partial(generate_non_gaussian,
                xi=xi2, L=num_dimensions, g=gain,
                ),
      )
    )
    
    # Adjust support
    z = Z(gain)
    self.adjust_support = jax.jit(
      jax.vmap(
        partial(lambda x, support: (x * z + 1) * (support[1] - support[0]) / 2 + support[0],
                support=support)
        )
    )
    

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  # @profile
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

    if isinstance(index, slice):
      # TODO(leonl): Deal with the case where `index.stop` is `None`.
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
    # adjust support
    exemplars = self.adjust_support(exemplars)

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    dataset = NonlinearGPDataset(key=key, xi1=0.1, xi2=1.1, gain=1., num_dimensions=100, num_exemplars=1e4)
    batch_size = 1000
    print(len(dataset))
    # import time
    # start_time = time.time()
    # def run_(dataset, batch_size):
      # for i in range(0, len(dataset), batch_size):
      #   x, y = dataset[i:i+batch_size]
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    # import cProfile
    # cProfile.runctx('run_(dataset, batch_size)', 
    #                 locals={},
    #                 globals={'run_': run_, 'dataset': dataset, 'batch_size': batch_size},
    #                 filename='nlgp_jax_stats',
    #                 )
    
    # cProfile.run('dataset.__getitem__(slice(0, len(dataset)))', 
    #                 # locals={},
    #                 # globals={'dataset': dataset, },
    #                 filename='nlgp_jax_stats',
    #                 )

    # import pstats
    # p = pstats.Stats('nlgp_jax_stats')
    # from pstats import SortKey
    # p.sort_stats(SortKey.CUMULATIVE).print_stats(.01)
    
    # dataset.__getitem__(slice(0, len(dataset)))
    for i in range(0, len(dataset), batch_size):
      x, y = dataset.__getitem__(slice(i, i+batch_size))
    print(profiler.print_stats())
    
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

