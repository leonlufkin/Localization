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
import ipdb

from jaxnets.models import StopGradient, Linear, MLP, SCM, GatedNet
from localization.models.initializers import trunc_normal_init, lecun_normal_init, xavier_normal_init, torch_init, pretrained_init, pruned_init, small_bump_init
