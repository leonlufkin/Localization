"""Simulate the expected value of some function over samples from a dataset."""

import pandas as pd
from pandas.api.types import CategoricalDtype

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Generator
from jax import Array

import os
from socket import gethostname
import itertools
from functools import partial
import pprint
import time

import numpy as np
import wandb

import jax
import jax.numpy as jnp

import equinox as eqx
import optax

from localization import datasets
from localization import samplers
from localization.utils import build_non_gaussian_covariance

# @partial(jax.jit, static_argnums=(0, 1))
def simulate_exp(key: jax.Array, func: Callable, sampler: samplers.Sampler, n: int = 1000) -> Array:
  """Simulate the expected value of `func` over `n` samples from `dataset`."""

  vmap = jax.vmap(func, in_axes=(0, 0))
  i = jax.random.randint(key, minval=0, maxval=len(sampler)-n, shape=())
  x, y = sampler[jnp.arange(i, i + n)]
  return vmap(x, y).mean(axis=0)

def f_relu_inner(x, y, w):
  return y * jnp.where(w @ x >= 0, x, 0)

def f_relu_approx(w, xi1=2, g=100):
  num_dimensions = len(w)
  Sigma1 = build_non_gaussian_covariance(num_dimensions, xi1, g)
  sw1 = Sigma1 @ w
  D = 1 / np.sqrt(w @ sw1 - (sw1 ** 2))
  f = 0.25 * jax.scipy.special.erf(D * sw1 / jnp.sqrt(2))
  return f

def gauss(x):
  return jnp.exp(-0.5 * x ** 2)
  
def dgauss(x):
  return -x * jnp.exp(-0.5 * x ** 2)

def f_sigmoid_inner(x, y, w):
  return y * jnp.exp(-0.5 * ((w @ x) - 1) ** 2) * x

def f_sigmoid_inner_rewrite(x, y, w):
  c = w @ x
  return y * ( jnp.exp(-0.5 * (c-1) ** 2) - jnp.exp(-0.5 * (c+1) ** 2) ) * jnp.where(c >= 0, x, 0)

if __name__ == '__main__':

  # define config
  config = dict(
    seed=0,
    num_dimensions=100,
    batch_size=100000,
    dataset_cls=datasets.NonlinearGPDataset,
    # dataset_cls=datasets.IsingDataset,
    xi1=5,#0.7,
    xi2=2,#0.3,
    gain=100,
    adjust=(-1.0, 1.0),
  )
  
  key = jax.random.PRNGKey(0)
  dataset = datasets.NonlinearGPDataset(key=key, xi=(config['xi1'], config['xi2'],), gain=config['gain'], num_dimensions=config['num_dimensions'], num_exemplars=config['batch_size'])
  sampler = samplers.EpochSampler(key, dataset, num_epochs=1)
  
  # plot
  def draw_samples(key: jax.Array, func: Callable, sampler: samplers.Sampler, n: int = 1000) -> Array:
    vmap = jax.vmap(func, in_axes=(0, 0))
    i = jax.random.randint(key, minval=0, maxval=len(sampler)-n, shape=())
    x, y = sampler[jnp.arange(i, i + n)]
    return vmap(x, y)
  
  @jax.vmap
  def adjust_filter(x):
    return gauss(x-1) - gauss(x+1)
  @jax.vmap
  def stability_filter(x):
    return dgauss(x-1) - dgauss(x+1)
  print(adjust_filter(jnp.array([1e-3, 0.5])))
  print(stability_filter(jnp.array([1e-3, 0.5])))
  
  import matplotlib.pyplot as plt
  fig, axs = plt.subplots(5, 6, figsize=(20, 10), sharex=False, sharey=False)
  for i in range(5):
    if i == 0:
      w = 0.001 * jax.random.normal(jax.random.PRNGKey(42), shape=(config['num_dimensions'],))
    elif i == 1:
      w = 0.1 * jax.random.normal(jax.random.PRNGKey(42), shape=(config['num_dimensions'],))
    elif i == 2:
      w = jax.random.normal(jax.random.PRNGKey(42), shape=(config['num_dimensions'],))
    elif i == 3:
      w = jnp.zeros(config['num_dimensions']); w = w.at[20].set(1); w = w.at[25].set(1)
    else:
      w = jnp.exp( -( jnp.arange(config['num_dimensions']) - 20 ) ** 2 / 10 )

    # std dev
    def preactivation(x, y, w):
      return y * (w @ x)
    preact_samples = draw_samples(key, partial(preactivation, w=w), sampler, n=config['batch_size'])
    print(i, jnp.std(preact_samples, axis=0))
    adjust_samples = adjust_filter(preact_samples)
    dg_samples = stability_filter(preact_samples)
      
    # plot
    w_relu_approx = partial(f_relu_approx, xi1=config['xi1'], g=100)(w)
    f_relu_inner_ = partial(f_relu_inner, w=w)
    w_relu = simulate_exp(key, f_relu_inner_, sampler, n=config['batch_size'])
    f_sigmoid_inner_ = partial(f_sigmoid_inner, w=w)
    w_sigmoid = simulate_exp(key, f_sigmoid_inner_, sampler, n=config['batch_size'])
    f_sigmoid_inner_rewrite_ = partial(f_sigmoid_inner_rewrite, w=w)
    w_sigmoid_rewrite = simulate_exp(key, f_sigmoid_inner_rewrite_, sampler, n=config['batch_size'])
    
    axs[i,0].plot(w)
    axs[i,0].set_title('w')
    axs[i,0].set_xlabel('dimension')
    #
    axs[i,1].plot(w_relu_approx, label='approx', color='tab:blue', alpha=0.5, linestyle='--')
    axs[i,1].plot(w_relu, label='true', color='tab:orange', alpha=0.5, linestyle='--')
    axs[i,1].legend()
    axs[i,1].set_title('f(w) - ReLU')
    axs[i,1].set_xlabel('dimension')
    #
    axs[i,2].plot(w_sigmoid, label='sigmoid', color='tab:blue', alpha=0.5, linestyle='--')
    axs[i,2].plot(w_sigmoid_rewrite / jnp.abs(w_sigmoid_rewrite).max() * jnp.abs(w_sigmoid).max(), label='sigmoid (rewrite)', color='tab:green', alpha=0.5, linestyle='-', linewidth=0.5)
    axs[i,2].plot(w_relu / jnp.abs(w_relu).max() * jnp.abs(w_sigmoid).max(), label='ReLU', color='tab:orange', alpha=0.5, linestyle='--')
    axs[i,2].legend()
    axs[i,2].set_title('f(w) - sigmoid')
    axs[i,2].set_xlabel('dimension')
    axs[i,2].set_ylabel('value')
    #
    axs[i,3].hist(preact_samples, bins=50, density=True)
    axs[i,3].set_title('preactivation')
    #
    axs[i,4].hist(adjust_samples, bins=50, density=True)
    axs[i,4].set_title('adjustment')
    # axs[i,4].plot(preact_samples, adjust_samples, '.', alpha=0.005)
    # axs[i,4].set_title('preactivation v adjustment')
    #
    axs[i,5].hist(dg_samples, bins=50, density=True)
    axs[i,5].set_title('stability')
    # axs[i,5].plot(preact_samples, dg_samples, '.', alpha=0.005)
    # axs[i,5].set_title('preactivation v stability')
  
  fig.suptitle('w and f(w)')
  fig.tight_layout()
  fig.savefig('w_and_f_w.png', dpi=300)
  