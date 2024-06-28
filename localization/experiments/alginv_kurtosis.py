"""Simulate online stochastic gradient descent learning of a simple task."""

# Pandas before JAX or JAXtyping. # TODO:(leonl) Why?
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

import jax
import jax.numpy as jnp

import optax

from localization import datasets
# from nets import samplers
from localization import samplers
from localization import models
from localization.utils import make_key
from localization.experiments import supervise

import ipdb

if __name__ == '__main__':
  
  # define config
  config = dict(
    seed=0,
    num_dimensions=40,
    dim=1,
    gain=None, # gain=100,
    init_scale=0.1,
    activation='relu',
    model_cls=models.SimpleNet,
    use_bias=False,
    optimizer_fn=optax.sgd,
    learning_rate=0.01,
    batch_size=1000,
    num_epochs=1000,
    dataset_cls=datasets.NortaDataset,
    # marginal_qdf=datasets.AlgQDF(k=5),
    xi=(3, 0.1),
    num_steps=1000,
    adjust=(-1.0, 1.0),
    class_proportion=0.5,
    sampler_cls=samplers.EpochSampler,
    init_fn=models.xavier_normal_init,
    loss_fn='mse',
    save_=True,
    evaluation_interval=100,
  )
  
  ks = list(np.arange(2, 7, 0.5))
  hiddens = [1, 2, 40, 100]
  
  # run it
  from tqdm import tqdm
  print(len(list(itertools.product(ks, hiddens))))
  for (k, hidden) in tqdm(itertools.product(ks, hiddens)):
    config.update(dict(marginal_qdf=datasets.AlgQDF(k=k), num_hiddens=hidden))
    supervise(wandb_=False, **config)

