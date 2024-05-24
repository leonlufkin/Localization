"""`Sampler`s are sequences of samples from a `Dataset`."""
from __future__ import annotations

from collections.abc import Sequence
from typing_extensions import Protocol
from jax import Array
# from nets.datasets.base import ExemplarType
from localization.datasets.base import ExemplarType

from enum import Enum
from enum import unique

import copy
from functools import partial

import jax
from jax import numpy as jnp
from jax import nn as jnn

from localization.datasets import Dataset

# The initial number of sequences to instantiate within an infinite `Sampler`.
# Changing this parameter impacts PRNGs for sequence sampling.
# TODO(eringrant): Change to use `jax.random.fold_in`.
MAX_NUM_SEQS = int(1e7)


@unique
class QueryType(Enum):
  """Types of queries that can be generated from a `Sampler`."""

  # Generate the query in the same manner as the context.
  NATURALISTIC = 1
  # Generate a query from classes in the context.
  SUPPORTED = 2
  # Generate a query from classes that do not occur in the context.
  UNSUPPORTED = 3


def zipfian_weights(num_classes: int, zipf_exponent: float) -> Array:
  """Compute Zipfian weights for `num_classes` classes."""
  return jnp.exp(-zipf_exponent * jnp.log(jnp.arange(1, num_classes + 1)))


def zipfian_distribution(num_classes: int, zipf_exponent: float) -> Array:
  """Compute Zipfian distribution for `num_classes` classes."""
  weights = zipfian_weights(num_classes, zipf_exponent)
  return weights / jnp.sum(weights)


def generate_exemplar_idx_sequence(
  key: Array,
  label_seq: Array,
  dataset_labels: Array,
) -> Array:
  """Generate a sequence of exemplar indices.

  Args:
    key: A key for randomness in sampling.
    label_seq: A sequence of `context_len` class labels for which to sample
      exemplars.
    dataset_labels: Class labels corresponding to the dataset of exemplars
      from which to sample.

  Returns:
    An array of indices into `dataset_labels` corresponding to sampled
    exemplars.
  """
  # Identify valid class exemplars for each element of the sequence
  # via a [context_len + 1, dataset_len] mask.
  exemplar_mask_seq = jnn.one_hot(
    dataset_labels, num_classes=dataset_labels.max(), dtype=jnp.int_
  ).T[label_seq]

  @partial(jax.vmap, in_axes=0)
  def _sample_exemplar(key, p):
    # Note: This samples exemplars independently, and thus with replacement,
    # which gives the possibility of duplicate exemplars in a sequence.
    return jax.random.choice(key, jnp.arange(len(dataset_labels)), shape=(), p=p)

  # Sample an exemplar per class element in the sequence.
  exemplar_idx_seq = _sample_exemplar(
    jax.random.split(key, len(exemplar_mask_seq)),
    jax.nn.softmax(jnp.log(exemplar_mask_seq)),
  )

  # A [context_len + 1] array of indices into `dataset_labels`,
  # corresponding to sampled exemplars.
  return exemplar_idx_seq


class ClassSampler(Protocol):
  """Protocol for a function that generates a sequence of class indices."""

  def __call__(self, key: Array, num_classes: int) -> Array:
    """Generate a sequence of class indices."""
    ...


class ExemplarSampler(Protocol):
  """Protocol for a function that generates a sequence of exemplar indices."""

  def __call__(self, key: Array, label_seq: Array, dataset_labels: Array) -> Array:
    """Generate a sequence of exemplar indices."""
    ...


def generate_sequence(
  key: Array,
  dataset_labels: Array,
  classes_to_sample: Array,
  generate_class_idx_sequence_fn: ClassSampler,
  generate_exemplar_idx_sequence_fn: ExemplarSampler,
) -> Array:
  """Generate a sequence of examples.

  Args:
    key: A key for randomness in sampling.
    dataset_labels: Class labels corresponding to the dataset of exemplars
      from which to sample; the ith element of `dataset_labels` is the class
      label of the ith example in the dataset.
    classes_to_sample: The subset of class labels in `dataset_labels` to be
      sampled, perhaps corresponding to a class split.
    generate_class_idx_sequence_fn: A function that generates a sequence of
      class indices.
    generate_exemplar_idx_sequence_fn: A function that generates a sequence of
      exemplar indices.

  Returns:
    An array of indices into `dataset_labels` corresponding to sampled
    exemplars.
  """
  class_key, exemplar_key = jax.random.split(key, 2)

  # Sample class indices.
  class_idx_seq = generate_class_idx_sequence_fn(
    key=class_key, num_classes=classes_to_sample.size
  )

  # Translate class indices into class labels.
  label_seq = jnp.take(classes_to_sample, class_idx_seq)

  # Sample exemplar indices.
  exemplar_idx_seq = generate_exemplar_idx_sequence_fn(
    key=exemplar_key,
    label_seq=label_seq,
    dataset_labels=dataset_labels,
  )

  return exemplar_idx_seq


class Sampler(Sequence):
  """Sampler of sequences drawn from a `nets.datasets.Dataset`."""


class SingletonSampler(Sampler):
  """Sampler of a sequence of examples."""


# TODO(eringrant): Is this faster than a recursive call to `__getitem__`?
def slice_to_array(s: slice, array_length: int) -> Array:
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))


class EpochSampler(SingletonSampler):
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

    self.permute_dataset_index = jax.jit(
        partial(
          jax.random.permutation,
          x=jnp.arange(self.dataset_size),
        ))

  def __len__(self) -> int:
    """Return the number of example-label pairs in `Sampler`."""
    if self.num_epochs is None:
      return int(float("inf"))  # Infinite length if num_epochs is not set
    return self.num_epochs * self.dataset_size

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

    #epoch_key = jax.random.fold_in(self.key, epoch_idx)
    #permuted_index = self.permute_dataset_index(epoch_key)[index_in_epoch]
    permuted_index = index_in_epoch
    # print(permuted_index)

    return self.dataset[permuted_index]


class SequenceSampler(Sampler):
  """Sampler of context + query sequences for in-context learning."""

