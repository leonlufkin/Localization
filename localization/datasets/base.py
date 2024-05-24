"""`Dataset`s are sequences of unique examples."""
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Sequence
# from nptyping import NDArray
# from nptyping import Bool
# from nptyping import Floating
# from nptyping import Int
from numpy.typing import NDArray#, Bool, Floating, Int
from jax import Array

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
ExemplarType = tuple[NDArray[Any], NDArray[Any]]


def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))

class Dataset:
  """A `Dataset` of class exemplars from which to draw sequences."""

  num_exemplars: int
  num_dimensions: int

  def __init__(
    self,
    key: Array,
    num_exemplars=1000,
    num_dimensions=40,
  ):
    """A `Dataset` of class exemplars from which to draw sequences.

    Args:
      key: A key for randomness in sampling.
      num_exemplars_per_class: Number of exemplars per class to draw from the
          underlying dataset.
    """
    self.key = key
    self.num_exemplars = num_exemplars
    self.num_dimensions = num_dimensions

  def __len__(self) -> int:
    """Number of exemplars in this `Dataset`."""
    return int(self.num_exemplars) # len(self._exemplars)

  def process_index(self, index: int | slice) -> tuple[Array, int]:
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
      
    return index, n

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Shape of an exemplar."""
    raise NotImplementedError("To be implemented by the subclass.")

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""
    raise NotImplementedError("To be implemented by the subclass.")