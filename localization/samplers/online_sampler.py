import jax
import jax.numpy as jnp

from jax import Array
from nets.datasets.base import ExemplarType

from nets.samplers.base import SingletonSampler, slice_to_array
from nets.datasets import Dataset

class OnlineSampler(SingletonSampler):
  """Sampler of example-label pairs over multiple epochs."""

  def __init__(
    self,
    key: Array,
    dataset: Dataset,
    num_epochs: int | None = None,
  ):
    """Sampler of example-label pairs over multiple epochs."""
    self.key = key
    self.dataset = dataset
    self.num_epochs = num_epochs
    self.epoch_count = 0
    self.index_in_epoch = 0

    self.dataset_size = len(self.dataset)

  def __len__(self) -> int:
    """Return the number of example-label pairs in `Sampler`."""
    if self.num_epochs is None:
      return int(float("inf"))  # Infinite length if num_epochs is not set
    return self.dataset_size

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Return exemplar-class pairs at index `index` of `Sampler`."""
    # TODO(eringrant): Simplify this while maintaining type-validity.
    if isinstance(index, slice):
      transformed_index = slice_to_array(index, len(self))
    else:
      transformed_index = index

    epoch_idx = transformed_index // self.dataset_size
    if not isinstance(epoch_idx, int):
      unique_vals = jnp.unique(epoch_idx)
      if unique_vals.size != 1:
        # TODO(eringrant): Implement this case.
        raise ValueError("Array should contain only one unique value.")
      epoch_idx = unique_vals[0]
    index_in_epoch = transformed_index % self.dataset_size

    if self.num_epochs is not None and epoch_idx >= self.num_epochs:
      raise StopIteration("Reached the end of data generation.")

    epoch_key = jax.random.fold_in(self.key, epoch_idx)
    permuted_index = jax.random.permutation(
      epoch_key,
      jnp.arange(self.dataset_size),
    )[index_in_epoch]

    return self.dataset[permuted_index]