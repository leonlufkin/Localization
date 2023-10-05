"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax.random import KeyArray

import jax
import jax.numpy as jnp
from jax import random
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from datasets.base import Dataset
# from nets.datasets.base import DatasetSplit
from nets.datasets.base import ExemplarType
# from nets.datasets.base import ExemplarLabeling
# from nets.datasets.base import HoldoutClassLabeling

from jax.scipy.special import erf as gain_function

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
    key: KeyArray,
    xi1: float = 0.1,
    xi2: float = 1.1,
    gain: float = 1.,
    class_proportion: float = 0.5,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
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
    
    def generate_non_gaussian(key, xi, L, g):
        C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
        C = jnp.exp(-C ** 2 / (xi ** 2))
        z = jax.random.multivariate_normal(key, jnp.zeros(L), C, method="svd") # FIXME: using svd for numerical stability, breaks if xi > 2.5 ish
        x = gain_function(g * z) / Z(g)
        return x

    def generate_non_gaussian_branching(key, class_proportion, L, g):
      label_key, exemplar_key = jax.random.split(key, 2)
      label = jax.random.bernoulli(label_key, p=class_proportion)
      xi = jnp.where(label, xi1, xi2)
      exemplar = generate_non_gaussian(exemplar_key, xi, L, g)
      label = 2 * jnp.float32(label) - 1
      return exemplar, label

    self.generate_xi = jax.jit(
    jax.vmap(
            partial(generate_non_gaussian_branching, 
                    class_proportion=class_proportion,
                    L=num_dimensions, g=gain)
        )
    )

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

    if isinstance(index, slice):
      # TODO(leonl): Deal with the case where `index.stop` is `None`.
      if index.stop is None:
        raise ValueError("Slice `index.stop` must be specified.")
      index = slice_to_array(index, len(self))
    else:
      index = index

    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, index)
    if isinstance(index, int):
        keys = jnp.expand_dims(keys, axis=0)

    exemplars, labels = self.generate_xi(
        key=keys,
    )

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    dataset = NonlinearGPDataset(key=key, xi1=0.1, xi2=1.1, gain=1., num_dimensions=100)
    print(len(dataset))
    print(dataset[:10])
    from nets import samplers
    sampler = samplers.EpochSampler(
        key=key,
        dataset=dataset,
        num_epochs=1,
    )
    print(len(sampler))
    print(sampler[:1][0])
    print(sampler[:1][1])
    print(sampler[:1][0].shape, sampler[:1][1].shape)

