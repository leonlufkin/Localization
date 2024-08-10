"""Simulate online stochastic gradient descent learning of a simple task."""

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
import jax.random as jr

import equinox as eqx
import optax

from localization import datasets, samplers, models
from localization.utils import make_key, localization_make_key
from jaxnets.utils import get_weights, repack_weights, batcher
from jaxnets.utils import accuracy, mse, ce

import ipdb


@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array, key: Array, loss_fn: Callable) -> Array:
  """Compute cross-entropy loss on a single example."""
  keys = jax.random.split(key, x.shape[0])
  pred_y = jax.vmap(model)(x, key=keys)
  losses = loss_fn(pred_y, y)
  loss = jnp.mean(losses)
  return loss


@eqx.filter_jit
def train_step(
  model: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
  x: Array,
  y: Array,
  key: Array,
  loss_fn: Callable,
) -> tuple[Array, eqx.Module, Array]:
  """Train the model on a single example."""
  loss, grads = compute_loss(model, x, y, key, loss_fn)
  updates, opt_state = optimizer.update(grads, opt_state)
  model = eqx.apply_updates(model, updates)
  return loss, model, opt_state


@eqx.filter_jit
def eval_step(
  x: Array,
  y: Array,
  key: Array,
  model: eqx.Module,
  loss_fn: Callable,
) -> Mapping[str, Array]:
  """Evaluate the model on a single example-label pair."""
  pred_y = model(x, key=key)

  # Standard metrics.
  elementwise_acc = accuracy(pred_y, y)
  elementwise_loss = loss_fn(pred_y, y)

  d = {
    "accuracy": elementwise_acc.mean(),
    "loss": elementwise_loss.mean(),
  }

  return d


def summarize_metrics(metrics):
  """Summarize metrics output from `eval_step` for printing."""
  acc = metrics['accuracy'].mean(0)
  loss = metrics['loss'].mean(0)
  with np.printoptions(precision=2):
    return (
      "\taccuracy:"
      f"\t{acc}"
      "\n\tloss:"
      f"\t\t{loss}"
    )

def metrics_to_dict(metrics: Mapping[str, Array]) -> dict:
  """dict-ify metrics from `eval_step` for later analysis."""
  iteration = metrics["training iteration"]
  loss = metrics["loss"].mean(0)
  acc = metrics["accuracy"].mean(0)
  d = {"iteration": iteration, "accuracy": acc, "loss": loss}
  return d

def metrics_to_df(metrics: Mapping[str, Array]) -> pd.DataFrame:
  """Pandas-ify metrics from `eval_step` for later analysis."""
  metrics_ = metrics_to_dict(metrics)
  return pd.DataFrame(metrics_to_dict(metrics_), index=[0])
  
def evaluate(
  iteration: int,
  dataset_split: str,
  key: Array,
  model: eqx.Module,
  sampler: Sequence,
  batch_size: int,
  loss_fn: Callable,
) -> pd.DataFrame:
  """Convenience function to evaluate `model` on batches from `sampler`."""
  metrics = {}

  ### Metrics metadata.
  metrics["training iteration"] = iteration

  ### Behavioral metrics.
  # TODO: Figure out the right syntax for `eqx.filter_vmap`.
  _eval_step = jax.vmap(partial(eval_step, model=model, loss_fn=loss_fn), (0, 0, 0))

  # Probing metric shapes.
  num_examples = len(sampler)
  incremental_metrics = dict(
    (
      metric_name,
      np.repeat(np.empty_like(metric_value), repeats=num_examples, axis=0),
    )
    for metric_name, metric_value in _eval_step(
      sampler[:1][0], sampler[:1][1], key[jnp.newaxis]
    ).items()
  )

  print("Starting evaluation...")
  start = time.time()

  for i, (x, y) in enumerate(batcher(sampler, batch_size)):
    (key,) = jax.random.split(key, 1)
    print(x.shape, y.shape)
    batch_metrics = _eval_step(x, y, jax.random.split(key, x.shape[0]))
    for metric_name in incremental_metrics.keys():
      incremental_metrics[metric_name][
        i * batch_size : min((i + 1) * batch_size, num_examples)
      ] = batch_metrics[metric_name]

  metrics.update(incremental_metrics)

  ### Model / parameter metrics.
  # metrics["last layer weight norm"] = float(jnp.linalg.norm(model.unembed.weight))
  # metrics["last layer bias norm"] = float(jnp.linalg.norm(model.unembed.bias))

  end = time.time()
  print(f"Completed evaluation over {num_examples} examples in {end - start:.2f} secs.")

  print("####")
  print(f"ITERATION {iteration}")
  print(f"{dataset_split} set:")
  print(summarize_metrics(metrics))
  
  return metrics_to_dict(metrics)




def update_config(
  config,
) -> dict:
  num_dimensions = config['num_dimensions']
  dim = config['dim']
  if 'hidden_size' in config.keys():
    num_hiddens = hidden_size = config.pop('hidden_size')
  if 'num_hiddens' in config.keys():
    num_hiddens = hidden_size = config.pop('num_hiddens')
  
  config.update(dict(
    in_size=num_dimensions ** dim,
    out_size=1,
    num_hiddens=num_hiddens,
    hidden_size=num_hiddens,
  ))
  
  return config

def save(
  metrics, 
  weights, 
  model, 
  path_key, 
  save_weights, 
  save_model,
):
  # save directory
  savedir = f"../results" if gethostname() == "Leons-MBP" else "/ceph/scratch/leonl/results"
  os.makedirs(f"{savedir}/metrics", exist_ok=True)
  np.savez(f"{savedir}/metrics/{path_key}.npz", metrics=metrics)
  print(f"Metrics saved at {savedir}/metrics/{path_key}.npz")
  # compiling and saving weights (if requested)
  if save_weights:
    weights = repack_weights(weights)
    os.makedirs(f"{savedir}/weights", exist_ok=True)
    np.savez(f"{savedir}/weights/{path_key}.npz", *weights)
    print(f"Weights saved at {savedir}/weights/{path_key}.npz")
  # save model (if requested)
  if save_model:
    os.makedirs(f"{savedir}/models", exist_ok=True)
    eqx.tree_serialise_leaves(f"{savedir}/models/{path_key}.eqx", model)
    print(f"Model saved at {savedir}/models/{path_key}.eqx")
  # return
  if save_weights and save_model:
    return metrics, weights, model
  if save_weights:
    return metrics, weights
  if save_model:
    return metrics, model
  return metrics
  

def simulate(
  seed: int = 0,
  # Dataset
  dataset_cls: type[datasets.Dataset] = datasets.NonlinearGPDataset,
  num_dimensions: int = 100,
  dim: int = 1,
  # class_proportion: float = 0.5,
  # xi: float | tuple[float],
  # marginal_qdf = None,
  # Model
  model_cls: type[eqx.Module] = models.MLP,
  hidden_size: int = 64,
  activation: str | Callable = jax.nn.relu,
  init_fn: Callable = models.xavier_normal_init,
  # init_scale: float,
  # use_bias = True,
  # bias_trainable = True,
  # bias_value = 0.0,
  # Training and evaluation
  learning_rate: float = 0.1,
  batch_size: int = 100,
  num_epochs: int = 1000,
  optimizer_fn: Callable = optax.sgd,
  loss_fn: Callable = mse,
  sampler_cls: type[samplers.Sampler] = samplers.DirectSampler,
  # Logging and saving
  # wandb_: bool = False,
  save_weights: bool = True,
  save_model: bool = True,
  evaluation_interval: int | None = None,
  **kwargs, # extra kwargs
) -> tuple[pd.DataFrame, ...]:
  """Simulate in-context learning of classification tasks."""
  print(f"Using JAX backend: {jax.default_backend()}\n")

  config = locals()
  config.pop("kwargs") # unnecessary, but for clarity's sake
  config.update(kwargs)
  print("Using configuration:")
  pprint.pprint(config)
  print()
  
  path_key = make_key(**config)
  # path_key = localization_make_key(**config)
  print(f"Path key: {path_key}")

  # Update configuration.
  config = update_config(config)
  num_epochs = config['num_epochs']
  batch_size = config['batch_size']

  # Single source of randomness.
  data_key, model_key, train_key, eval_key = jax.random.split(
    jax.random.PRNGKey(seed), 4
  )

  #########
  # Data setup.
  dataset_key, sampler_key = jax.random.split(data_key)
  
  train_dataset_key, eval_dataset_key = jax.random.split(dataset_key)
  
  train_dataset = dataset_cls(
    key=train_dataset_key,
    # num_exemplars=num_epochs*batch_size, # FIXME: How should I set this in general?
    **config,
  )
  
  eval_dataset = dataset_cls(
    key=eval_dataset_key,
    num_exemplars=1000, # FIXME: This is a hack.
    **config,
  )
  # ipdb.set_trace()

  print(f"Length of train dataset: {len(train_dataset)}")
  print(f"Length of eval dataset: {len(eval_dataset)}")

  if len(train_dataset) % batch_size != 0:
    raise ValueError("Batch size must evenly divide the number of training examples.")

  train_sampler_key, eval_sampler_key = jax.random.split(sampler_key)

  train_sampler = sampler_cls(
    key=train_sampler_key,
    dataset=train_dataset,
    **config,
  )
  
  eval_sampler = sampler_cls(
    key=eval_sampler_key,
    dataset=eval_dataset,
    **config,
  )
  
  print(f"Length of train sampler: {len(train_sampler)}")
  print(f"Length of eval sampler: {len(eval_sampler)}")

  #########
  # Model setup.
  model = model_cls(
    key=model_key,
    **config,
  )
  print(f"Model:\n{model}\n")

  #########
  # Training loop.
  optimizer = optimizer_fn(learning_rate=learning_rate)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  # Bookkeeping.
  metrics = []
  itercount = itertools.count()
  
  # Save weights before starting training.
  weights = []
  if save_weights:
    weights = [get_weights(model)]

  # Evaluate before starting training.
  metrics_ = evaluate(
      iteration=0,
      dataset_split="eval",
      sampler=eval_sampler,
      model=eqx.tree_inference(model, True),
      key=eval_key,
      batch_size=len(eval_sampler),
      loss_fn=loss_fn,
    )
  metrics.append(metrics_)

  # Training starts at iteration 1.
  next(itercount)
  if evaluation_interval is None:
    evaluation_interval = min(500, max(num_epochs // 10, 1))
  print(f"Evaluating every {evaluation_interval} training steps.")
  if evaluation_interval == 0:
    raise ValueError("Too many `evaluations_per_epoch`.")

  print("\nStarting training...")
  
  start_time = time.time()
  
  for epoch, (x, y) in enumerate(batcher(train_sampler, batch_size, num_epochs*batch_size)):
    (train_key,) = jax.random.split(train_key, 1)
    train_step_num = int(next(itercount))
    train_loss, model, opt_state = train_step(
      model, optimizer, opt_state, x, y, train_key, loss_fn
    )

    if train_step_num % evaluation_interval == 0:
      epoch_time = time.time() - start_time
      print(f"Epoch {train_step_num // evaluation_interval} in {epoch_time:0.2f} seconds.")
      
      metrics_ = evaluate(
        iteration=train_step_num,
        dataset_split="eval",
        sampler=eval_sampler,
        model=eqx.tree_inference(model, True),
        key=eval_key,
        batch_size=len(eval_sampler),
        loss_fn=loss_fn,
      )
      metrics.append(metrics_)
      
      if save_weights:
        weights.append(get_weights(model))
      
      start_time = time.time()

  # wrapping up
  print("Training finished.")
  
  # compiling metrics
  metrics = pd.DataFrame(metrics)
  metrics['epoch'] = np.minimum(metrics.index * evaluation_interval, num_epochs)

  # save and return, as desired
  return save(metrics, weights, model, path_key, save_weights, save_model)


if __name__ == '__main__':
  
  # define config
  config = dict(
    seed=0,
    num_dimensions=40,
    dim=1,
    hidden_size=1,
    gain=100,
    init_scale=0.001,
    activation=jax.nn.relu, #'relu',
    # model_cls=models.SCM,
    model_cls=models.MLP,
    use_bias=False,
    optimizer_fn=optax.sgd,
    learning_rate=0.3,
    batch_size=1000,
    num_epochs=1000,
    datset_cls=datasets.NonlinearGPDataset,
    xi=(0.3, 0.7),
    sampler_cls=samplers.DirectSampler,
    init_fn=models.xavier_normal_init,
    loss_fn=mse, #'mse',
    save_weights=True,
    save_model=True,
    evaluation_interval=100,
  )

  # run it
  metrics, weights, model = simulate(**config)
  ipdb.set_trace()

