"""Simple feedforward neural networks."""
import numpy as np
from math import sqrt

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import equinox.nn as enn

from jax import Array
from collections.abc import Callable


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

class StopGradient(eqx.Module):
  """Stop gradient wrapper."""

  array: jnp.ndarray

  def __jax_array__(self):
    """Return the array wrapped with a stop gradient op."""
    return jax.lax.stop_gradient(self.array)


class Linear(enn.Linear):
  """Linear layer."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    trainable: bool = True,
    *,
    key: Array,
    init_scale: float = 1.0,
    init_fn: Callable = xavier_normal_init,
  ):
    """Initialize a linear layer."""
    super().__init__(
      in_features=in_features,
      out_features=out_features,
      use_bias=use_bias,
      key=key,
    )

    # Reinitialize weight from variance scaling distribution, reusing `key`.
    self.weight: Array = init_fn(self.weight, key=key, scale=init_scale) # xavier_normal_init
    if not trainable:
      self.weight = StopGradient(self.weight)

    # Reinitialize bias to zeros.
    if use_bias:
      self.bias: Array = jnp.zeros_like(self.bias)

      if not trainable:
        self.bias = StopGradient(self.bias)


class MLP(eqx.Module):
  """Multi-layer perceptron."""

  fc1: eqx.Module
  act: Callable
  fc2: eqx.Module
  tanh: Callable

  def __init__(
    self,
    in_features: int,
    hidden_features: int | None = None,
    out_features: int | None = 1,
    act: Callable = lambda x: x,
    *,
    key: Array = None,
    init_scale: float = 1.0,
    **linear_kwargs,
  ):
    """Initialize an MLP.

    Args:
       in_features: The expected dimension of the input.
       hidden_features: Dimensionality of the hidden layer.
       out_features: The dimension of the output feature.
       act: Activation function to be applied to the intermediate layers.
       drop: The probability associated with `Dropout`.
       key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation.
       init_scale: The scale of the variance of the initial weights.
    """
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    keys = jrandom.split(key, 2)

    self.fc1 = Linear(
      in_features=in_features,
      out_features=hidden_features,
      key=keys[0],
      init_scale=init_scale,
      **linear_kwargs,
    )
    self.act = act
    self.fc2 = Linear(
      in_features=hidden_features,
      out_features=out_features,
      key=keys[1],
      init_scale=init_scale,
      **linear_kwargs,
    )
    self.tanh = jax.nn.tanh

  def __call__(self, x: Array, *, key: Array) -> Array:
    """Apply the MLP block to the input."""
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    x = self.tanh(x)
    return x

class SimpleNet(eqx.Module):
  """
  2-layer MLP, but only one layer is learnable.
  No dropout.
  """

  fc1: eqx.Module
  act: Callable

  def __init__(
    self,
    in_features: int,
    hidden_features: int | None = None,
    act: Callable = lambda x: x,
    *,
    key: Array = None,
    init_scale: float = 1.0,
    **linear_kwargs
  ):
    """Initialize an MLP.

    Args:
       in_features: The expected dimension of the input.
       hidden_features: Dimensionality of the hidden layer.
       out_features: The dimension of the output feature.
       act: Activation function to be applied to the intermediate layers.
       drop: The probability associated with `Dropout`.
       key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation.
       init_scale: The scale of the variance of the initial weights.
    """
    super().__init__()
    # out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    self.fc1 = Linear(
      in_features=in_features,
      out_features=hidden_features,
      key=key,
      init_scale=init_scale,
      **linear_kwargs # TODO: try use_bias = False
    ) 
    self.act = act

    # TODO(leonl): Add an static layer with input dimension `hidden_features`.
    del hidden_features


  def __call__(self, x: Array, *, key: Array) -> Array:
    """Apply the MLP block to the input."""
    x = self.fc1(x)
    x = self.act(x)

    # TODO(leonl): Call the static layer with input dimension `hidden_features`.
    #x = self.fc2(x)
    #x = self.act(x)

    x = jnp.mean(x)#, axis=0)#.reshape(-1)

    return x
