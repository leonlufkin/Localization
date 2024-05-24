"""Weight initializers for neural networks."""
import numpy as np
from math import sqrt

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import equinox.nn as enn

from jax import Array
from collections.abc import Callable
import ipdb

def trunc_normal_init(
  weight: Array, key: Array, stddev: float | None = None
) -> Array:
  """Truncated normal distribution initialization."""
  _, in_ = weight.shape
  stddev = stddev or sqrt(1.0 / max(1.0, in_))
  return stddev * jax.random.truncated_normal(
    key=key,
    shape=weight.shape,
    lower=-2,
    upper=2,
  )


# Adapted from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py.
def lecun_normal_init(
  weight: Array,
  key: Array,
  scale: float = 1.0,
) -> Array:
  """LeCun (variance-scaling) normal distribution initialization."""
  _, in_ = weight.shape
  scale /= max(1.0, in_)

  stddev = np.sqrt(scale)
  # Adjust stddev for truncation.
  # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
  distribution_stddev = jnp.asarray(0.87962566103423978, dtype=float)
  stddev = stddev / distribution_stddev

  return trunc_normal_init(weight, key, stddev=stddev)

def xavier_normal_init(
  weight: Array,
  key: Array,
  scale: float = 1.0,
):
  xavier = jax.nn.initializers.glorot_normal()
  stddev = np.sqrt(scale)
  return stddev * xavier(key, weight.shape)

def torch_init(
  weight: Array,
  key: Array,
  scale: float = 1.0,
):
  K, L = weight.shape
  assert K == 100 and L == 40
  torch_init = jnp.load('weights/torch_weights.npz', allow_pickle=True)
  weight, bias = torch_init['weight'], torch_init['bias']
  print("Loaded torch weights.")
  return weight

def pretrained_init(
  weight: Array,
  key: Array,
  scale: float = 1.0,
):
  K, L = weight.shape
  weight = jnp.load(f'weights/pretrained_weights_L={L}_K={K}.npy')
  print("Loaded pretrained weights for L={L}, K={K}.")
  return jnp.sqrt(scale) * weight

def pruned_init(
  weight: Array,
  key: Array,
  scale: float = 1.0,
):
  K, L = weight.shape
  weight = jnp.load(f'weights/pruned_weights_L={L}_K={K}.npy')
  print("Loaded pruned weights for L={L}, K={K}.")
  return jnp.sqrt(scale) * weight

def small_bump_init(
  weight: Array,
  key: Array,
  scale: float = 1.0,
):
  K, L = weight.shape
  assert K == 2 and L == 40
  scale = jnp.sqrt(scale)
  weight = jnp.zeros((K, L))
  weight = weight.at[0, 2].set(1. * scale)
  weight = weight.at[0, 3].set(2. * scale)
  weight = weight.at[0, 4].set(1. * scale)
  weight = weight.at[1, 2].set(-1. * scale)
  weight = weight.at[1, 3].set(-2. * scale)
  weight = weight.at[1, 4].set(-1. * scale)
  return weight