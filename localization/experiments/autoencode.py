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
import wandb

import jax
import jax.numpy as jnp

import equinox as eqx
import optax

from localization import datasets
# from nets import samplers
from localization import samplers
from localization import models
from localization.utils import make_key
from localization.models.feedforward import StopGradient

# import ipdb

def mse(pred_x: Array, x: Array) -> Array:
  """Compute elementwise mean squared error."""
  # ipdb.set_trace()
  return jnp.square(pred_x - x)

def kl_sparsity(hidden: Array, rho: float) -> Array:
  """Compute KL divergence for sparsity regularization."""
  hidden = jax.nn.sigmoid(hidden)
  rho_hat = jnp.mean(hidden, axis=0)
  kl = rho * jnp.log(rho / rho_hat) + (1 - rho) * jnp.log((1 - rho) / (1 - rho_hat))
  return kl.mean()

def l1_sparsity(hidden: Array) -> Array:
  """Compute L1 sparsity regularization."""
  return jnp.abs(hidden).mean()

def batcher(sampler: Sequence, batch_size: int) -> Generator[Sequence, None, None]:
  """Batch a sequence of examples."""
  n = len(sampler)
  # print("batcher: n=", n)
  for i in range(0, n, batch_size):
    yield sampler[i : min(i + batch_size, n)]

@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, key: Array, loss_fn: Callable, beta: float = 0., rho: float = 0.05) -> Array:
  """Compute cross-entropy loss on a single example."""
  keys = jax.random.split(key, x.shape[0])
  pred_x, hidden_act = jax.vmap(model)(x, key=keys)
  if rho is not None:
    loss = loss_fn(pred_x, x) + beta * kl_sparsity(hidden_act, rho)
  else:
    loss = loss_fn(pred_x, x) + beta * l1_sparsity(hidden_act)
  return loss.mean()

@eqx.filter_jit
def train_step(
  model: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
  x: Array,
  key: Array,
  loss_fn: Callable,
  beta: float = 0.,
  rho: float = 0.05,
) -> tuple[Array, eqx.Module, Array]:
  """Train the model on a single example."""
  loss, grads = compute_loss(model, x, key, loss_fn, beta, rho)
  # print("grads l2 norm: ", jnp.linalg.norm(grads(x, key=key)))
  # print(grads)
  updates, opt_state = optimizer.update(grads, opt_state)
  # print("updates", type(updates))
  # print("opt_state", opt_state)
  model = eqx.apply_updates(model, updates)
  return loss, model, opt_state


@eqx.filter_jit
def eval_step(
  x: Array,
  key: Array,
  model: eqx.Module,
  loss_fn: Callable,
) -> Mapping[str, Array]:
  """Evaluate the model on a single example-label pair."""
  # print(f"eval_step: x.shape={x.shape}")
  pred_x, hidden = model(x, key=key)
  # ipdb.set_trace()
  # x = (40,), pred_x = ()

  # Standard metrics.
  elementwise_loss = loss_fn(pred_x, x)
  elementwise_sparsity = jnp.abs(hidden).mean(axis=0)
  
  # Track model internals.
  bias = model.fc1.bias
  if isinstance(bias, StopGradient):
    bias = bias.array

  d = { "loss": elementwise_loss.mean(), "sparsity": elementwise_sparsity.mean() }
  if bias is not None:
    d.update({ "bias": bias })

  return d


def summarize_metrics(metrics):
  """Summarize metrics output from `eval_step` for printing."""
  loss = metrics['loss'].mean(0)
  sparsity = metrics['sparsity'].mean(0)
  with np.printoptions(precision=2):
    return (
      "\tloss:"
      f"\t\t{loss}\n"
      "\tsparsity:"
      f"\t{sparsity}"
    )

def metrics_to_dict(metrics: Mapping[str, Array]) -> dict:
  """dict-ify metrics from `eval_step` for later analysis."""
  iteration = metrics["training iteration"]
  loss = metrics["loss"].mean(0)
  # mean_pred_y, mean_y = metrics["mean pred_y"].mean(0), metrics["mean y"].mean(0)
  d = {"iteration": iteration, "loss": loss}#, "mean pred_y": mean_pred_y, "mean y": mean_y}
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
  # TODO(eringrant): Figure out the right syntax for `eqx.filter_vmap`.
  _eval_step = jax.vmap(partial(eval_step, model=model, loss_fn=loss_fn), (0, 0))

  # Probing metric shapes.
  num_examples = len(sampler)
  print(sampler[:1][0].shape)
  incremental_metrics = dict(
    (
      metric_name,
      np.repeat(np.empty_like(metric_value), repeats=num_examples, axis=0),
    )
    for metric_name, metric_value in _eval_step(
      sampler[:1][0], key[jnp.newaxis]
    ).items()
  )

  print("Starting evaluation...")
  start = time.time()

  for i, (x, _) in enumerate(batcher(sampler, batch_size)):
    batch_metrics = _eval_step(x, jax.random.split(key, x.shape[0]))
    for metric_name in incremental_metrics.keys():
      incremental_metrics[metric_name][
        i * batch_size : min((i + 1) * batch_size, num_examples)
      ] = batch_metrics[metric_name]

  metrics.update(incremental_metrics)

  end = time.time()
  print(f"Completed evaluation over {num_examples} examples in {end - start:.2f} secs.")

  print("####")
  print(f"ITERATION {iteration}")
  print(f"{dataset_split} set:")
  print(summarize_metrics(metrics))
  
  return metrics_to_dict(metrics)


def autoencode(
  # Model params.
  # num_ins: int,
  num_hiddens: int,
  init_scale: float,
  activation: str | Callable,
  # Training and evaluation params.
  learning_rate: float | Callable,  # TODO(eringrant): Define interface.
  batch_size: int,
  num_epochs: int,
  # Dataset params.
  xi: tuple[float],
  num_dimensions: int,
  class_proportion: float,
  # Default params.
  seed: int = 0,
  dataset_cls: type[datasets.Dataset] = datasets.NonlinearGPDataset,
  base_dataset: type[datasets.Dataset] = datasets.NonlinearGPDataset,
  marginal_adjust: Callable = lambda key, n: jax.random.normal(key, (n,)),
  dim: int = 1,
  adjust: tuple[float, float] | Callable | None = None,
  gain: float | None = None,
  num_steps : int = 1000,
  df: float | None = None,
  model_cls: type[eqx.Module] = models.MLP,
  use_bias = True,
  bias_value = 0.0,
  bias_trainable = True,
  init_fn: Callable = models.xavier_normal_init,
  optimizer_fn: Callable = optax.sgd,
  loss_fn: str | Callable = 'mse',
  sampler_cls: type[samplers.Sampler] = samplers.OnlineSampler,
  wandb_: bool = False,
  save_: bool = True,
  evaluation_interval: int | None = None,
  beta: float = 0.,
  rho: float = 0.05,
  **kwargs, # extra kwargs
) -> tuple[pd.DataFrame, ...]:
  """Simulate in-context learning of classification tasks."""
  print(f"Using JAX backend: {jax.default_backend()}\n")

  print("Using configuration:")
  pprint.pprint(locals())
  print()
  
  if not model_cls == models.MLP:
    raise ValueError("Only MLP models are supported at this time.")

  config = dict(
    seed=seed,
    num_dimensions=num_dimensions,
    num_hiddens=num_hiddens,
    init_scale=init_scale,
    activation=activation,
    model_cls=model_cls,
    use_bias=use_bias,
    bias_value=bias_value,
    bias_trainable=bias_trainable,
    optimizer_fn=optimizer_fn,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=num_epochs,
    dataset_cls=dataset_cls,
    adjust=adjust,
    xi=xi,
    gain=gain,
    dim=dim,
    num_steps=num_steps,
    df=df,
    class_proportion=class_proportion,
    sampler_cls=sampler_cls,
    init_fn=init_fn,
    loss_fn=loss_fn,
    beta=beta,
    rho=rho,
    supervise=False,
  )
  
  # Determine size of output
  loss_fn = mse
  
  if dataset_cls == datasets.AdjustMarginalDataset or dataset_cls == datasets.SymmBreakDataset:
    config.update(dict(base_dataset=base_dataset, marginal_adjust=marginal_adjust))
    if dataset_cls == datasets.SymmBreakDataset:
      batch_size *= 100
  
  path_key = make_key(**config)

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
  if activation == 'sigmoid':
    activation = jax.nn.sigmoid
  elif activation == 'relu':
    activation = jax.nn.relu
  elif activation == 'identity':
    activation = lambda x: x

  #########
  # Data setup.
  dataset_key, sampler_key = jax.random.split(data_key)

  train_dataset_key, eval_dataset_key = jax.random.split(dataset_key)
  
  train_dataset = dataset_cls(
    key=train_dataset_key,
    num_exemplars=num_epochs*batch_size,
    **config,
    **kwargs,
  )
  
  eval_dataset = dataset_cls(
    key=eval_dataset_key,
    num_exemplars=1000,
    **config,
    **kwargs,
  )
  
  # import ipdb; ipdb.set_trace()

  print(f"Length of train dataset: {len(train_dataset)}")
  print(f"Length of eval dataset: {len(eval_dataset)}")

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
  
  eval_sampler = sampler_cls(
    key=eval_sampler_key,
    dataset=eval_dataset,
    num_epochs=1,
  )
  
  print(f"Length of train sampler: {len(train_sampler)}")
  print(f"Length of eval sampler: {len(eval_sampler)}")
  
  # import ipdb; ipdb.set_trace()

  #########
  # Model setup.
  model = model_cls(
      in_features=num_dimensions ** dim,
      hidden_features=num_hiddens,
      out_features=num_dimensions ** dim,
      act=activation,
      key=model_key,
      init_scale=init_scale,
      use_bias=use_bias,
      bias_value=bias_value,
      bias_trainable=bias_trainable,
      init_fn=init_fn,
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
  weights.append(model.fc1.weight)

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
  log_to_wandb(metrics_) if wandb_ else metrics.append(metrics_)

  # Training starts at iteration 1.
  next(itercount)
  if evaluation_interval is None:
    evaluation_interval = min(500, max(num_epochs // 10, 1))
  print(f"Evaluating every {evaluation_interval} training steps.")
  if evaluation_interval == 0:
    raise ValueError("Too many `evaluations_per_epoch`.")

  print("\nStarting training...")
  
  start_time = time.time()
  
  for epoch, (x, _) in enumerate(batcher(train_sampler, batch_size)):
    (train_key,) = jax.random.split(train_key, 1)
    train_step_num = int(next(itercount))
    train_loss, model, opt_state = train_step(
      model, optimizer, opt_state, x, train_key, loss_fn, beta, rho
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
      
      log_to_wandb(metrics_) if wandb_ else metrics.append(metrics_)
      
      weights.append(model.fc1.weight)
      
      start_time = time.time()

  # wrapping up
  print("Training finished.")
  if wandb_:
    try:
      wandb.finish(quiet=True)
    except:
      print("wandb.finish() failed")

  # compiling weights and metrics
  weights = np.stack(weights, axis=0)
  metrics = pd.DataFrame(metrics)
  metrics['epoch'] = np.minimum(metrics.index * evaluation_interval, num_epochs)
  metrics = metrics.to_numpy()[:,:-1]
  if save_:
    weightwd = '/Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights' if gethostname() == 'Leons-MBP' else '/ceph/scratch/leonl/results/weights'
    np.savez(f"{weightwd}/{path_key}.npz", weights=weights, metrics=metrics)
    print("Saved " + path_key + ".npz")

  return weights, metrics

if __name__ == '__main__':

  # define config
  config = dict(
    seed=0,
    num_dimensions=40,
    num_hiddens=2,
    gain=100,#0.01,#3,
    dim=2,
    init_scale=0.01,
    activation='relu',
    # activation='sigmoid',
    model_cls=models.MLP,
    use_bias=False,
    optimizer_fn=optax.sgd,
    # learning_rate=0.3,
    learning_rate=10.0,
    batch_size=1000,#1000,
    num_epochs=1000,#500,
    # dataset_cls=datasets.SinglePulseDataset,
    dataset_cls=datasets.NonlinearGPDataset,
    xi=(3., 0.1,),
    num_steps=1000,
    adjust=(-1.0, 1.0),
    class_proportion=0.5,
    sampler_cls=samplers.EpochSampler,
    init_fn=models.xavier_normal_init,
    loss_fn='mse',
    save_=True, # FIXME: Reset to False!
    evaluation_interval=10,
    beta=0.5,
    # rho=0.05,
    # rho=None,
  )
  
  autoencode(wandb_=False, **config)
  
  # gains = [0.01, 100]
  # hiddens = [1, 2, 40, 100]
  # xis = [(3., 0.1,), (0.1, 3,), (3, 3), (0.1, 0.1), (5, 1, 0.1)]
  
  # # log config to wandb
  # wandb_ = False

  # # run it
  # from tqdm import tqdm
  # for (gain, hidden, xi) in tqdm(itertools.product(gains, hiddens, xis)):
  #   config.update(dict(gain=gain, num_hiddens=hidden, xi=xi))
  #   autoencode(wandb_=wandb_, **config)
