import jax
import jax.numpy as jnp

from jax import Array
from localization.datasets.base import ExemplarType

from localization.samplers.base import SingletonSampler, slice_to_array
from localization.datasets import Dataset

class OtherSampler(SingletonSampler):
  """Sampler of example-label pairs over multiple epochs."""

  def __init__(
    self,
    key: Array,
    dataset: Dataset,
  ):
    """Sampler of example-label pairs over multiple epochs."""
    self.key = key
    self.dataset = dataset
    self.epoch_count = 0
    self.index_in_epoch = 0

    self.dataset_size = len(self.dataset)

  def __len__(self) -> int:
    """Return the number of example-label pairs in `Sampler`."""
    # if self.num_epochs is None:
    #   return int(float("inf"))  # Infinite length if num_epochs is not set
    return self.dataset_size

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Return exemplar-class pairs at index `index` of `Sampler`."""
    # TODO(eringrant): Simplify this while maintaining type-validity.
    if isinstance(index, slice):
      transformed_index = slice_to_array(index, len(self))
    else:
      transformed_index = index

    index_in_epoch = transformed_index % self.dataset_size # we shouldn't get indices larger than the dataset size, but we add this just in case

    return self.dataset[index_in_epoch]