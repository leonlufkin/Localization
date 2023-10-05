"""`Dataset`s are sequences of unique examples."""
from typing import Any
from collections.abc import Sequence
from nptyping import NDArray
from nptyping import Bool
from nptyping import Floating
from nptyping import Int
from jax.random import KeyArray
from jaxtyping import Array

from enum import Enum
from enum import unique
from functools import cached_property
from functools import partial
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.nn as jnn


# Type hints.
IndexType = int | Sequence[int] | slice
ExemplarType = tuple[NDArray[Any, Floating], NDArray[Any, Int]]


@unique
class DatasetSplit(Enum):
  """Which split of the underlying dataset to use."""

  TRAIN = 1
  VALID = 2
  TEST = 3
  ALL = 4


class Dataset:
  """A `Dataset` of class exemplars from which to draw sequences."""

#   _exemplars: Sequence[Path] | NDArray
#   _labels: NDArray

#   num_train_classes: int
#   prop_train_labels: float
#   num_test_classes: int
#   prop_test_labels: float
#   num_valid_classes: int
#   prop_valid_labels: float

  num_exemplars: int

  def __init__(
    self,
    key: KeyArray,
    # split: DatasetSplit,
    # num_train_classes: int,
    # prop_train_labels: float,
    # num_test_classes: int,
    # prop_test_labels: float,
    # num_valid_classes: int = 0,
    # prop_valid_labels: float = 0,
    # num_exemplars_per_class: int = 400,
    num_exemplars=1000
  ):
    """A `Dataset` of class exemplars from which to draw sequences.

    Args:
      key: A key for randomness in sampling.
      split: Which split of the underlying dataset to use.
      exemplar_labeling: How to assign class labels to exemplars from the underlying
          dataset.
      holdout_class_labeling: How to assign class labels to holdout (validation and
          testing) splits of this `Dataset`.
      num_train_classes: Number of training classes in this `Dataset`.
      prop_train_labels: Size of the training label set proportional to the underlying
          class set. If 1.0, then labels are identical to the underlying class labels;
          if < 1.0, then labels are wrapped in increasing order.
      num_valid_classes: Number of validation classes in this `Dataset`.
      prop_valid_labels: Size of the validation label set proportional to the
          underlying class set. If 1.0, then labels are identical to the underlying
          class labels; if < 1.0, then labels are wrapped in increasing order.
      num_test_classes: Number of testing classes in this `Dataset`.
      prop_test_labels: Size of the testing label set proportional to the underlying
          class set. If 1.0, then labels are identical to the underlying class labels;
          if < 1.0, then labels are wrapped in increasing order.
      num_exemplars_per_class: Number of exemplars per class to draw from the
          underlying dataset.
    """
    self.key = key
    self.num_exemplars = num_exemplars

    # self.num_train_classes = num_train_classes
    # self.num_valid_classes = num_valid_classes
    # self.num_test_classes = num_test_classes
    # self.num_exemplars_per_class = num_exemplars_per_class

    # # TODO(eringrant): Empty valid class set?
    # if not all(
    #   0.0 < p <= 1.0
    #   for p in (
    #     prop_train_labels,
    #     prop_valid_labels,
    #     prop_test_labels,
    #   )
    # ):
    #   raise ValueError(
    #     "One of `prop_{train,valid,test}_labels` was invalid: "
    #     f"{prop_train_labels}, {prop_valid_labels}, {prop_test_labels}."
    #   )

    # num_train_labels, train_indices = get_wrapped_indices(
    #   prop_train_labels, num_train_classes
    # )
    # num_valid_labels, valid_indices = get_wrapped_indices(
    #   prop_valid_labels,
    #   num_valid_classes,
    #   offset=0
    #   if holdout_class_labeling == HoldoutClassLabeling.TRAIN_LABELS
    #   else num_train_labels,
    # )
    # num_test_labels, test_indices = get_wrapped_indices(
    #   prop_test_labels,
    #   num_test_classes,
    #   offset=0
    #   if holdout_class_labeling == HoldoutClassLabeling.TRAIN_LABELS
    #   else num_train_labels + num_valid_labels,
    # )

    # indices = jnp.concatenate((train_indices, valid_indices, test_indices))
    # modulus = jnp.eye(self.num_classes, dtype=int)[indices, :]

    # self.wrap_labels = jax.jit(
    #   partial(
    #     wrap_labels,
    #     num_classes=self.num_classes,
    #     modulus=modulus,
    #   )
    # )

  def __len__(self) -> int:
    """Number of exemplars in this `Dataset`."""
    return int(self.num_exemplars) # len(self._exemplars)

#   @property
#   def num_classes(self) -> int:
#     """Number of classes in this `Dataset`."""
#     return self.num_train_classes + self.num_valid_classes + self.num_test_classes

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Shape of an exemplar."""
    raise NotImplementedError("To be implemented by the subclass.")

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""
    raise NotImplementedError("To be implemented by the subclass.")