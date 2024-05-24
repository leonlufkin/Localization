"""
Simulate online stochastic gradient descent learning of a simple task.
Help from: https://github.com/tuananhle7/ica/blob/main/ica.py
"""


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
import math
import wandb

import jax
import jax.numpy as jnp

import equinox as eqx
import optax

from localization import datasets
# from nets import samplers
from localization import samplers
from localization import models
from localization.utils import make_ica_key
from localization.models.feedforward import StopGradient

from tqdm import tqdm
import ipdb


# These functions copied from the GitHub repo cited above, with minor modifications
def get_signal(mixing_matrix, source):
    """Compute single signal from a single source
    Args
        mixing_matrix [signal_dim, source_dim]
        source [source_dim]
    
    Returns
        signal [signal_dim]
    """
    return jnp.dot(mixing_matrix, source)


def get_subgaussian_log_prob(source):
    """Subgaussian log probability of a single source.

    Args
        source [source_dim]

    Returns []
    """
    return jnp.sum(jnp.sqrt(jnp.abs(source)))


def get_supergaussian_log_prob(source):
    """Supergaussian log probability of a single source.
    log cosh(x) = log ( (exp(x) + exp(-x)) / 2 )
                = log (exp(x) + exp(-x)) - log(2)
                = logaddexp(x, -x) - log(2)
                   
    https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions
    https://en.wikipedia.org/wiki/FastICA#Single_component_extraction

    Args
        source [source_dim]

    Returns []
    """
    return jnp.sum(jnp.logaddexp(source, -source) - jnp.log(2))


def get_antisymmetric_matrix(raw_antisymmetric_matrix):
    """Returns an antisymmetric matrix
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix

    Args
        raw_antisymmetric_matrix [dim * (dim - 1) / 2]: elements in the upper triangular
            (excluding the diagonal)

    Returns [dim, dim]
    """
    dim = math.ceil(math.sqrt(raw_antisymmetric_matrix.shape[0] * 2))
    # ipdb.set_trace()
    zeros = jnp.zeros((dim, dim))
    indices = jnp.triu_indices(dim, k=1)
    upper_triangular = zeros.at[indices].set(raw_antisymmetric_matrix)
    return upper_triangular - upper_triangular.T


def get_orthonormal_matrix(raw_orthonormal_matrix):
    """Returns an orthonormal matrix
    https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map

    Args
        raw_orthonormal_matrix [dim * (dim - 1) / 2]

    Returns [dim, dim]
    """
    antisymmetric_matrix = get_antisymmetric_matrix(raw_orthonormal_matrix)
    dim = antisymmetric_matrix.shape[0]
    eye = jnp.eye(dim)
    return jnp.matmul(eye - antisymmetric_matrix, jnp.linalg.inv(eye + antisymmetric_matrix))


def get_source(signal, raw_mixing_matrix, num_components):
    """Get source from signal
    
    Args
        signal [signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
    
    Returns []
    """
    return jnp.matmul(get_mixing_matrix(raw_mixing_matrix)[:,:num_components].T, signal)


def get_log_likelihood(signal, raw_mixing_matrix, get_source_log_prob, num_components):
    """Log likelihood of a single signal log p(x_n)
    
    Args
        signal [signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
        get_source_log_prob [source_dim] -> []
    
    Returns []
    """
    return get_source_log_prob(get_source(signal, raw_mixing_matrix, num_components))


def get_mixing_matrix(raw_mixing_matrix):
    """Get mixing matrix from a vector of raw values (to be optimized)

    Args
        raw_orthonormal_matrix [dim * (dim - 1) / 2]

    Returns [dim, dim]
    """
    return get_orthonormal_matrix(raw_mixing_matrix)


def get_total_log_likelihood(signals, raw_mixing_matrix, get_source_log_prob, num_components):
    """Log likelihood of all signals ∑_n log p(x_n)
    
    Args
        signals [num_samples, signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
        get_source_log_prob [source_dim] -> []
    
    Returns []
    """
    log_likelihoods = jax.vmap(partial(get_log_likelihood, num_components=num_components), (0, None, None), 0)(
        signals, raw_mixing_matrix, get_source_log_prob
    )
    return jnp.sum(log_likelihoods)


def update_raw_mixing_matrix(raw_mixing_matrix, signals, get_source_log_prob, num_components, lr=1e-3):
    """Update raw mixing matrix by stepping the gradient

    Args:
        raw_mixing_matrix [signal_dim, source_dim]
        signals [num_samples, signal_dim]
        get_source_log_prob [source_dim] -> []
        lr (float)

    Returns
        total_log_likelihood []
        updated_raw_mixing_matrix [signal_dim, source_dim]
    """
    total_log_likelihood, g = jax.value_and_grad(partial(get_total_log_likelihood, num_components=num_components), 1)(
        signals, raw_mixing_matrix, get_source_log_prob
    )
    return total_log_likelihood, raw_mixing_matrix + lr * g


def preprocess_signal(signal):
    """Center and whiten the signal
    x_preprocessed = A @ (x - mean)

    Args
        signal [num_samples, signal_dim]
    
    Returns
        signal_preprocessed [num_samples, signal_dim]
        preprocessing_params
            A [signal_dim, signal_dim]
            mean [signal_dim]
    """
    mean = jnp.mean(signal, axis=0)
    signal_centered = signal - jnp.mean(signal, axis=0)

    signal_cov = jnp.mean(jax.vmap(jnp.outer, (0, 0), 0)(signal_centered, signal_centered), axis=0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(signal_cov)
    A = jnp.diag(eigenvalues ** (-1 / 2)) @ eigenvectors.T

    return jax.vmap(jnp.matmul, (None, 0), 0)(A, signal_centered), (A, mean)


def batcher(sampler: Sequence, batch_size: int) -> Generator[Sequence, None, None]:
  """Batch a sequence of examples."""
  n = len(sampler)
  # print("batcher: n=", n)
  for i in range(0, n, batch_size):
    yield sampler[i : min(i + batch_size, n)]




def summarize_metrics(metrics):
  """Summarize metrics output from `eval_step` for printing."""
  loss = metrics['loss'].mean(0)
  acc = metrics['accuracy'].mean(0)
  # bias = metrics['bias'].mean(0)
  # mean_y = metrics['mean y'].mean(0)
  # mean_pred_y = metrics['mean pred_y'].mean(0)
  with np.printoptions(precision=2):
    return (
      "\tloss:"
      f"\t\t{loss}"
      "\n\taccuracy:"
      f"\t{acc}"
      # "\n\tmean y:"
      # f"\t\t{mean_y}"
      # "\n\tmean pred_y:"
      # f"\t{mean_pred_y}"
    )

def metrics_to_dict(metrics: Mapping[str, Array]) -> dict:
  """dict-ify metrics from `eval_step` for later analysis."""
  iteration = metrics["training iteration"]
  loss = metrics["loss"].mean(0)
  acc = metrics["accuracy"].mean(0)
  # mean_pred_y, mean_y = metrics["mean pred_y"].mean(0), metrics["mean y"].mean(0)
  d = {"iteration": iteration, "loss": loss, "accuracy": acc}#, "mean pred_y": mean_pred_y, "mean y": mean_y}
  if "bias" in metrics.keys():
    d["bias"] = metrics["bias"].mean(0)
  return d

def metrics_to_df(metrics: Mapping[str, Array]) -> pd.DataFrame:
  """Pandas-ify metrics from `eval_step` for later analysis."""
  metrics_ = metrics_to_dict(metrics)
  if "bias" in metrics_.keys():
    metrics_.pop("bias")
  return pd.DataFrame(metrics_to_dict(metrics_), index=[0])

def log_to_wandb(metrics: Mapping[str, Array]) -> None:
  wandb.log(metrics)

def ica(
  # Model params.
  num_dimensions: int,
  get_source_log_prob: Callable,
  # Training and evaluation params.
  learning_rate: float | Callable,  # TODO(eringrant): Define interface.
  batch_size: int,
  # Dataset params.
  xi: tuple[float] = (3, 0.1),
  # Default params.
  seed: int = 0,
  num_components: int = None,
  num_steps: int = 1000,
  dataset_cls: type[datasets.Dataset] = datasets.NonlinearGPDataset,
  nonlinearity: str | Callable | None = None,
  class_proportion: float = 0.5,
  # Dataset args
  base_dataset: type[datasets.Dataset] = datasets.NonlinearGPDataset,
  dim: int = 1,
  gain: float | None = None,
  adjust=(-1.0, 1.0),
  marginal_adjust: Callable = lambda key, n: jax.random.normal(key, (n,)),
  # Model class and args
  # init_fn: Callable = models.xavier_normal_init,
  # init_scale: float = 0.1,
  # Sampler args
  sampler_cls: type[samplers.Sampler] = samplers.OnlineSampler,
  # Extra args
  wandb_: bool = False,
  save_: bool = True,
  evaluation_interval: int | None = None,
  **kwargs, # extra kwargs
) -> tuple[pd.DataFrame, ...]:
  """Simulate in-context learning of classification tasks."""
  print(f"Using JAX backend: {jax.default_backend()}\n")

  print("Using configuration:")
  pprint.pprint(locals())
  print()
  
  num_components = num_components or num_dimensions

  config = dict(
    seed=seed,
    num_dimensions=num_dimensions,
    num_components=num_components,
    num_steps=num_steps,
    # nonlinearity=nonlinearity,
    get_source_log_prob=get_source_log_prob,
    learning_rate=learning_rate,
    batch_size=batch_size,
    dataset_cls=dataset_cls,
    adjust=adjust,
    xi=xi,
    gain=gain,
    dim=dim,
    class_proportion=class_proportion,
    sampler_cls=sampler_cls,
  )
  
  if dataset_cls == datasets.AdjustMarginalDataset or dataset_cls == datasets.SymmBreakDataset:
    config.update(dict(base_dataset=base_dataset, marginal_adjust=marginal_adjust))
    if dataset_cls == datasets.SymmBreakDataset:
      batch_size *= 100
  
  path_key = make_ica_key(**config)

  #########
  # wandb setup.
  if wandb_:
    try:
      wandb.init(
        project="gimme-intuition",
        group="debug",
        name=path_key,
        notes=
        """
        Exploring variations in emergence of convolutional structure in fully-connected networks using the nonlinear Gaussian process and single pulse datasets.
        """,
        config=config
      )
    except Exception as e:
      print("wandb.init() failed with exception: ", e)
      print("Continuing by running locally.")
      wandb_ = False

  # Single source of randomness.
  data_key, model_key, train_key, eval_key = jax.random.split(
    jax.random.PRNGKey(seed), 4
  )

  # Fixing annoying activation function bug
  # NOTE: nonlinearity not currently being implemented; just using linear ICA
  if nonlinearity == 'sigmoid':
    nonlinearity = jax.nn.sigmoid
  elif nonlinearity == 'relu':
    nonlinearity = jax.nn.relu
  elif nonlinearity == 'identity':
    nonlinearity = lambda x: x

  #########
  # Data setup.
  dataset_key, sampler_key = jax.random.split(data_key)

  train_dataset_key, eval_dataset_key = jax.random.split(dataset_key)
  
  train_dataset = dataset_cls(
    key=train_dataset_key,
    num_exemplars=batch_size,
    **config,
    **kwargs,
  )
  
  # import ipdb; ipdb.set_trace()

  print(f"Length of train dataset: {len(train_dataset)}")

  # `None` batch size implies full-batch optimization.
  if batch_size is None:
    batch_size = len(train_dataset)

  if len(train_dataset) % batch_size != 0:
    raise ValueError("Batch size must evenly divide the number of training examples.")

  train_sampler_key, eval_sampler_key = jax.random.split(sampler_key)

  train_sampler = sampler_cls(
    key=train_sampler_key,
    dataset=train_dataset,
    num_epochs=1,
  )
  print(f"Length of train sampler: {len(train_sampler)}")

  #########
  # Training loop.

  start_time = time.time()
  
  # Sample data.
  x, y = train_sampler[:batch_size]

  # Preprocess data (i.e. whiten signal).
  signal_preprocessed, (Wz, mean) = preprocess_signal(x)
  
  # Initialize mixing matrix.
  raw_mixing_matrix = jax.random.normal(train_key, (int(num_dimensions * (num_dimensions - 1) / 2),))
  mixing_matrix = get_mixing_matrix(raw_mixing_matrix) @ Wz
  
  # Evaluate before starting training.
  metrics = []
  total_log_likelihood = get_total_log_likelihood(signal_preprocessed, raw_mixing_matrix, get_source_log_prob, num_components)
  tll = total_log_likelihood.item()
  log_to_wandb({'total log likelihood': tll}) if wandb_ else metrics.append(tll)
  
  # Start training.
  raw_mixing_matrices = [raw_mixing_matrix]
  mixing_matrices = [mixing_matrix]
  print("\nStarting training...")
  for train_step_num in tqdm(range(num_steps)):
    total_log_likelihood, raw_mixing_matrix = update_raw_mixing_matrix(
      raw_mixing_matrix, signal_preprocessed, get_source_log_prob, num_components, learning_rate,
    )

    if train_step_num % evaluation_interval == 0:
      epoch_time = time.time() - start_time
      # if epoch_time > 1.:
      print(f"Epoch {train_step_num // evaluation_interval} in {epoch_time:0.2f} seconds.")
      
      tll = total_log_likelihood.item()
      log_to_wandb({'total log likelihood': tll}) if wandb_ else metrics.append(tll)
      print(f"total log likelihood: {tll}")
      
      raw_mixing_matrices.append(raw_mixing_matrix)
      mixing_matrix = get_mixing_matrix(raw_mixing_matrix) @ Wz
      mixing_matrices.append(mixing_matrix)
      
      start_time = time.time()

  # wrapping up
  print("Training finished.")
  if wandb_:
    try:
      wandb.finish(quiet=True)
    except:
      print("wandb.finish() failed")

  # compiling weights and metrics
  raw_mixing_matrices = np.stack(raw_mixing_matrices, axis=0)
  mixing_matrices = np.stack(mixing_matrices, axis=0)
  metrics = pd.DataFrame(metrics)
  metrics['iteration'] = np.minimum(metrics.index * evaluation_interval, num_steps)
  metrics = metrics.to_numpy()[:,:-1]
  if save_:
    weightwd = '/Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights' if gethostname() == 'Leons-MBP' else '/ceph/scratch/leonl/results/weights'
    np.savez(f"{weightwd}/{path_key}.npz", raw_mixing_matrices=raw_mixing_matrices, mixing_matrices=mixing_matrices, metrics=metrics, Wz=Wz, mean=mean)
    print("Saved " + path_key + ".npz")

  return raw_mixing_matrices, mixing_matrices, metrics, (Wz, mean)

if __name__ == '__main__':

  # define config
  config = dict(
    # Training and evaluation params.
    num_dimensions=256,
    get_source_log_prob=get_subgaussian_log_prob,
    num_steps=2500,
    learning_rate=0.025,
    batch_size=1000,
    # Dataset params.
    xi=(3, 3,),
    # Default params.
    seed=0,
    # Dataset args
    dataset_cls=datasets.NonlinearGPDataset,
    dim=1,
    gain=3,
    adjust=(-1.0, 1.0),
    sampler_cls=samplers.EpochSampler,
    # Extra args
    save_=True, # FXIME: Make True!
    evaluation_interval=20,
  )
  
  # define config
  config = dict(
    # Training and evaluation params.
    num_components=10,
    num_dimensions=144,
    get_source_log_prob=get_supergaussian_log_prob,
    num_steps=250,
    learning_rate=5,
    batch_size=1000,
    # Dataset params.
    side_length=12,
    # Default params.
    seed=0,
    # Dataset args
    dataset_cls=datasets.ScenesDataset,
    sampler_cls=samplers.EpochSampler,
    # Extra args
    save_=True, # FXIME: Make True!
    evaluation_interval=20,
  )
  
  # log config to wandb
  wandb_ = False

  raw_mixing_matrices, mixing_matrices, metrics, preprocessing_params = ica(wandb_=wandb_, **config)

