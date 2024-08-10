import numpy as np
from jax import Array
import jax.numpy as jnp

from functools import partial

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Generator

from localization import datasets
from jaxnets.utils import make_key as jaxnets_make_key

def interval_to_str(interval):
  if isinstance(interval, tuple):
    start, end = "[", "]"
  else:
    interval = (interval,)
    start, end = "", ""
  out = ",".join([ f"{x:05.2f}" for x in interval ])
  return f"{start}{out}{end}"

def adjust_to_str(adjust):
  if adjust is None:
    return ""
  elif isinstance(adjust, tuple):
    return interval_to_str(adjust)
  elif isinstance(adjust, Callable):
    if isinstance(adjust, partial):
      return adjust.func.__name__
    return adjust.__name__
  else:
    raise ValueError(f"Invalid type for `adjust`: {type(adjust)}")
  
def xi_to_str(xi):
  if len(xi) == 2:
      xi2, xi1 = xi # NOTE: flipped order!
      xi_str = f'_xi1={interval_to_str(xi1)}_xi2={interval_to_str(xi2)}'
  else:
      xi_str = '_xi=' + ','.join([ f"{xi_:05.2f}" for xi_ in xi ])
  return xi_str 
  
def make_key(**config):
  return jaxnets_make_key(
    remove_keys=['wandb_', 'save_weights', 'save_model', 'task', 'config_modifier'],
    config=config
  )
  
def localization_make_key(dataset_cls, adjust, xi, class_proportion, batch_size, num_epochs, learning_rate, model_cls, use_bias, num_dimensions, num_hiddens, activation, init_scale, init_fn: Callable, loss_fn: str | Callable = 'mse', bias_value=None, gain=None, marginal_qdf=None, dim=1, df=None, seed=None, base_dataset=None, marginal_adjust=None, supervise=True, beta=None, rho=None, **extra_kwargs):
  dataset_name = dataset_cls.__name__
  model_name = model_cls.__name__
  base_dataset_name = base_dataset.__name__ if base_dataset is not None else None
  marginal_adjust_name = marginal_adjust.__name__ if marginal_adjust is not None else None
  # import ipdb; ipdb.set_trace()
  # construct xis name
  if isinstance(xi, int):
    xi = (xi,)
  xi_str = xi_to_str(xi)
  
  if df is not None and gain is not None:
    raise ValueError("Cannot specify both `df` and `gain`.")
  if gain is not None and marginal_qdf is not None:
    raise ValueError("Cannot specify both `gain` and `marginal_qdf`.")
  loss = loss_fn.__name__ if isinstance(loss_fn, Callable) else loss_fn
  return f'{dataset_name}{"_" + base_dataset_name if base_dataset is not None else ""}{"_" + marginal_adjust_name if marginal_adjust_name is not None else ""}{adjust_to_str(adjust)}{xi_str}'\
    f'{f"_gain={gain:.3f}" if gain is not None else ""}{marginal_qdf.__name__ if marginal_qdf is not None else ""}{f"_df={df}" if df is not None else ""}{f"_dim={dim:d}" if dim>1 else ""}_p={class_proportion:.2f}'\
    f'_batch_size={batch_size}_num_epochs={num_epochs}'\
    f'_loss={loss}_lr={learning_rate:.3f}'\
    f'_{model_name}{("" if bias_value is None else bias_value) if use_bias else "nobias"}_L={num_dimensions:03d}_K={num_hiddens:03d}_activation={activation.__name__ if isinstance(activation, Callable) else activation}'\
    f'_init_scale={init_scale:.3f}_{init_fn.__name__ if isinstance(init_fn, Callable) else init_fn}'\
    f'{f"_seed={seed:d}" if seed is not None else ""}' + ('_autoencode' if not supervise else '')\
    + (f'_beta={beta:.3f}' if (beta is not None) and (beta != 0.) else '') + (f'_rho={rho:.3f}' if (rho is not None) and ((beta is not None) and (beta != 0.)) else '')
        
def make_ica_key(num_dimensions, dataset_cls, xi, **extra_kwargs):
  dataset_name = dataset_cls.__name__
  if dataset_cls == datasets.NonlinearGPDataset:
    xi_str = xi_to_str(xi)
    return f"ica_{dataset_name}_L={num_dimensions}_xi={xi_str}"
  else:
    return 'ica_test'