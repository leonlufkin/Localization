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
import wandb

import jax
import jax.numpy as jnp

import equinox as eqx
import optax

# from nets import datasets
# export PYTHONPATH="${PYTHONPATH}:./"
from localization.datasets import Dataset
from localization.datasets import NonlinearGPDataset, SinglePulseDataset
from nets import samplers
# from nets import models
from localization import models
# from models import SimpleNet
# from nets import models

# from tqdm import tqdm


def accuracy(pred_y: Array, y: Array) -> Array:
  """Compute elementwise accuracy."""
  # print("accuracy: pred_y.shape=", pred_y.shape, "y.shape=", y.shape)
  predicted_class = jnp.where(pred_y > 0, 1., -1) # jnp.argmax(pred_y, axis=-1)
  return predicted_class == y


def mse(pred_y: Array, y: Array) -> Array:
  """Compute elementwise mean squared error."""
  return jnp.square(pred_y - y)


def batcher(sampler: Sequence, batch_size: int) -> Generator[Sequence, None, None]:
  """Batch a sequence of examples."""
  n = len(sampler)
  # print("batcher: n=", n)
  for i in range(0, n, batch_size):
    yield sampler[i : min(i + batch_size, n)]


@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array, key: Array) -> Array:
  """Compute cross-entropy loss on a single example."""
  keys = jax.random.split(key, x.shape[0])
  pred_y = jax.vmap(model)(x, key=keys)
  # import ipdb; ipdb.set_trace()
  loss = mse(pred_y, y)
  # print(jnp.mean(jnp.abs(pred_y)).item())
  return loss.mean()


@eqx.filter_jit
def train_step(
  model: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
  x: Array,
  y: Array,
  key: Array,
) -> tuple[Array, eqx.Module, Array]:
  """Train the model on a single example."""
  loss, grads = compute_loss(model, x, y, key)
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
  y: Array,
  key: Array,
  model: eqx.Module,
) -> Mapping[str, Array]:
  """Evaluate the model on a single example-label pair."""
  # print(f"eval_step: x.shape={x.shape}, y.shape={y.shape}")
  pred_y = model(x, key=key)

  # Standard metrics.
  elementwise_acc = accuracy(pred_y, y)
  elementwise_loss = mse(pred_y, y)

  return {
    "loss": elementwise_loss.mean(),
    "accuracy": elementwise_acc.mean(),
  }


def summarize_metrics(metrics):
  """Summarize metrics output from `eval_step` for printing."""
  loss = metrics['loss'].mean(0)
  acc = metrics['accuracy'].mean(0)
  with np.printoptions(precision=2):
    return (
      "\tloss:"
      f"\t\t{loss}"
      "\n\taccuracy:"
      f"\t{acc}"
    )

def metrics_to_dict(metrics: Mapping[str, Array]) -> dict:
  """dict-ify metrics from `eval_step` for later analysis."""
  iteration = metrics["training iteration"]
  loss = metrics["loss"].mean(0)
  acc = metrics["accuracy"].mean(0)
  d = {"iteration": iteration, "loss": loss, "accuracy": acc,}
  return d

def metrics_to_df(metrics: Mapping[str, Array]) -> pd.DataFrame:
  """Pandas-ify metrics from `eval_step` for later analysis."""
  return pd.DataFrame(metrics_to_dict(metrics), index=[0])

def log_to_wandb(metrics: Mapping[str, Array]) -> None:
  wandb.log(metrics)

def xi_to_str(xi):
  if isinstance(xi, int):
    xi = (xi,)
    start, end = "", ""
  else:
    start, end = "[", "]"
  out = ",".join([ f"{x:05.2f}" for x in xi ])
  return f"{start}{out}{end}"

def make_key(dataset_cls, xi1, xi2, class_proportion, batch_size, num_epochs, learning_rate, model_cls, num_dimensions, num_hiddens, activation, init_scale, init_fn: Callable, *extra_args, **extra_kwargs):
  dataset_name = dataset_cls.__name__
  model_name = model_cls.__name__
  return f'{dataset_name}_xi1={xi_to_str(xi1)}_xi2={xi_to_str(xi2)}_p={class_proportion:.2f}'\
    f'_batch_size={batch_size}_num_epochs={num_epochs}'\
    f'_loss=mse_lr={learning_rate:.3f}'\
    f'_{model_name}_L={num_dimensions:03d}_K={num_hiddens:03d}_activation={activation.__name__ if isinstance(activation, Callable) else activation}'\
    f'_init_scale={init_scale:.3f}_{init_fn.__name__ if isinstance(init_fn, Callable) else init_fn}'
  
def evaluate(
  iteration: int,
  dataset_split: str,
  key: Array,
  model: eqx.Module,
  sampler: Sequence,
  batch_size: int,
) -> pd.DataFrame:
  """Convenience function to evaluate `model` on batches from `sampler`."""
  metrics = {}

  ### Metrics metadata.
  metrics["training iteration"] = iteration

  ### Behavioral metrics.
  # TODO: Figure out the right syntax for `eqx.filter_vmap`.
  _eval_step = jax.vmap(partial(eval_step, model=model), (0, 0, 0))

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
    # (key,) = jax.random.split(key, 1)
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


def simulate(
  # Model params.
  # num_ins: int,
  num_hiddens: int,
  init_scale: float,
  activation: Callable,
  # Training and evaluation params.
  learning_rate: float | Callable,  # TODO: Define interface.
  batch_size: int,
  num_epochs: int,
  # Dataset params.
  xi1: tuple[float, float],
  xi2: tuple[float, float],
  num_dimensions: int,
  class_proportion: float,
  # Default params.
  seed: int = 0,
  dataset_cls: type[Dataset] = SinglePulseDataset,
  model_cls: type[eqx.Module] = models.MLP,
  init_fn: Callable = models.xavier_normal_init,
  optimizer_fn: Callable = optax.sgd,
  sampler_cls: type[samplers.EpochSampler] = samplers.EpochSampler,
  wandb_: bool = False,
) -> tuple[pd.DataFrame, ...]:
  """Simulate in-context learning of classification tasks."""
  print(f"Using JAX backend: {jax.default_backend()}\n")

  print("Using configuration:")
  pprint.pprint(locals())
  print()

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
    xi1=xi1, xi2=xi2,
    class_proportion=class_proportion,
    num_dimensions=num_dimensions,
    num_exemplars=num_epochs*batch_size,
  )
  
  eval_dataset = dataset_cls(
    key=eval_dataset_key,
    xi1=xi1, xi2=xi2,
    class_proportion=class_proportion,
    num_dimensions=num_dimensions,
    num_exemplars=20*batch_size,
  )

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


  #########
  # Model setup.
  model = model_cls(
      in_features=num_dimensions,
      hidden_features=num_hiddens,
      act=activation,
      key=model_key,
      init_scale=init_scale,
      use_bias=False,
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

  # Evaluate before starting training.
  metrics_ = evaluate(
      iteration=0,
      dataset_split="eval",
      sampler=eval_sampler,
      model=eqx.tree_inference(model, True),
      key=eval_key,
      batch_size=batch_size,
    )
  log_to_wandb(metrics_) if wandb_ else metrics.append(metrics_)

  # Training starts at iteration 1.
  next(itercount)
  evaluation_interval = min(500, max(num_epochs // 10, 1))
  print(f"Evaluating every {evaluation_interval} training steps.")
  if evaluation_interval == 0:
    raise ValueError("Too many `evaluations_per_epoch`.")

  print("\nStarting training...")
  print(f"Length of train sampler: {len(train_sampler)}")
  
  start_time = time.time()

  path_key = make_key(dataset_cls, xi1, xi2, class_proportion=class_proportion,
                      batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, 
                      model_cls=model_cls, num_dimensions=num_dimensions, num_hiddens=num_hiddens, 
                      activation=activation, init_scale=init_scale, init_fn=init_fn)
      
  def save_model_weights(train_step_num):
    # save model weights
    jnp.save(f"results/weights/{path_key}/fc1_{train_step_num}.npy", model.fc1.weight)
    print(f"Saved model weights at iteration {train_step_num}.")
  
  weights = [] # just for local runs
  weights.append(model.fc1.weight)
  
  for epoch, (x, y) in enumerate(batcher(train_sampler, batch_size)):
    (train_key,) = jax.random.split(train_key, 1)
    train_step_num = int(next(itercount))
    train_loss, model, opt_state = train_step(
      model, optimizer, opt_state, x, y, train_key
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
        batch_size=batch_size,
      )
      
      log_to_wandb(metrics_) if wandb_ else metrics.append(metrics_)
      
      weights.append(model.fc1.weight)
      
      start_time = time.time()

  print("Training finished.")
  try:
    wandb.finish(quiet=True)
  except:
    print("wandb.finish() failed")

  # combine all weights in /tmp/weights/
  weights = [ np.load(f"'/tmp/weights/fc1_{train_step_num}.npy") for train_step_num in range(0, num_epochs+1, evaluation_interval) ]
  weights = np.stack(weights, axis=0)
  os.makedirs(f"results/weights/{path_key}", exist_ok=True)
  np.save(f"results/weights/{path_key}/fc1.npy", weights)

  if not wandb_:
    df = pd.DataFrame(metrics)
    df['epoch'] = np.minimum(df.index * evaluation_interval, num_epochs)
    df.to_csv(f"results/weights/{path_key}/metrics.csv")
    return df

if __name__ == '__main__':

  # define config
  config = dict(
    seed=0,
    num_dimensions=70,
    num_hiddens=100,
    init_scale=1.0,
    activation=jax.nn.hard_tanh,
    model_cls=models.SimpleNet,
    optimizer_fn=optax.sgd,
    learning_rate=1.0,
    batch_size=100,
    num_epochs=1000,
    dataset_cls=SinglePulseDataset,
    xi1=(20, 25), # (20, 25),
    xi2=(5, 10), # (5, 10),
    class_proportion=0.5,
    sampler_cls=samplers.EpochSampler,
    init_fn=models.lecun_normal_init
  )
  
  # log config to wandb
  wandb_ = True # False
  if wandb_:
    try:
      wandb.init(
        project="single-pulse",
        group="debug",
        name=make_key(**config),
        notes=
        """
        Exploring variations in emergence of convolutional structure in fully-connected networks using the single-pulse dataset.
        """,
        config=config
      )
    except Exception as e:
      print("wandb.init() failed with exception: ", e)
      print("Continuing by running locally.")
      wandb_ = False

  # run it
  simulate(wandb_=wandb_, **config)
