"""Independent component analysis."""
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

class StopGradient(eqx.Module):
  """Stop gradient wrapper."""

  array: jnp.ndarray

  def __jax_array__(self):
    """Return the array wrapped with a stop gradient op."""
    return jax.lax.stop_gradient(self.array)

class ICA()



class Linear(enn.Linear):
  """Linear layer."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    weight_trainable: bool = True,
    bias_value: float = 0.0,
    bias_trainable: bool = False,
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
    if not weight_trainable:
      self.weight = StopGradient(self.weight)

    # Reinitialize bias to zeros.
    if use_bias:
      self.bias: Array = bias_value * jnp.ones_like(self.bias)

      if not bias_trainable:
        self.bias = StopGradient(self.bias)


class MLP(eqx.Module):
  """Multi-layer perceptron."""

  fc1: eqx.Module
  act: Callable
  fc2: eqx.Module
  # tanh: Callable

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
    # ipdb.set_trace()
    # self.tanh = jax.nn.tanh

  def __call__(self, x: Array, *, key: Array) -> Array:
    """
    Apply the MLP block to the input.
    Unlike previous MLP, return hidden layer as well (for sparse autoencoder).
    """
    preact = self.fc1(x)
    x = self.act(preact)
    x = self.fc2(x)
    return x, preact

class MLP_hidden(MLP):
  def __call__(self, x: Array, *, key: Array) -> Array:
    """
    Apply the MLP block to the input.
    Unlike previous MLP, return hidden layer as well (for sparse autoencoder).
    """
    preact = self.fc1(x)
    x = self.act(preact)
    x = self.fc2(x)
    return x, preact

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
    out_features: int | None = 1,
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

class GatedNet(eqx.Module):
  """
  2-layer MLP, but only one layer is learnable.
  Rather than an activation, we apply a gating function.
  """

  fc1: eqx.Module
  gate: Callable

  def __init__(
    self,
    in_features: int,
    hidden_features: int | None = None,
    act: Callable = lambda x: 1.,
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
       act: Gating function to be applied to the intermediate layer.
            Given input x, returns a vector of gates for all the intermediate neurons.
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
    self.gate = act

    # TODO(leonl): Add an static layer with input dimension `hidden_features`.
    del hidden_features


  def __call__(self, x: Array, *, key: Array) -> Array:
    """Apply the MLP block to the input."""
    p = self.fc1(x)
    g = self.gate(x)
    p = g * p
    y = jnp.mean(p)

    return y