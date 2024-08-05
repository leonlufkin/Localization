
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from localization.experiments import load, simulate_or_load
from localization.utils import sweep_func, product_kwargs

from jax import Array
from typing import Tuple
from functools import partial
import ipdb
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import pandas as pd
    from localization.utils import tupify, ipr, plot_rf_evolution
    from localization.experiments.model_sweep import config    
    from localization import datasets, models
    
    # config['batch_size'] = 5000
    # # sweep params
    # # seed = tuple(np.arange(30)),
    # # num_dimensions = (40, 100, 400,)
    # # dataset_cls = (datasets.NonlinearGPDataset, datasets.NortaDataset,)
    gains = jnp.logspace(-2, 2, 10)
    ks = jnp.array([4.1, 4.3, 4.5, 4.74, 5.0, 5.4, 6.1, 7.7, 10., 50.]) 
    ks = jnp.concatenate([ks, jnp.linspace(4, 10, 7)])
    num_dimensions = 40
    seed = 0 # 12
    
    # Update config: more gradient steps & more hidden neurons
    config.update(dict(
        num_epochs=10000, evaluation_interval=100,
        num_hiddens=10,
        batch_size=5000,
    ))
    # 
    weights = simulate_or_load(
        **config, 
        seed=seed,
        num_dimensions=40,
        dataset_cls=datasets.NortaDataset,
        marginal_qdf=datasets.AlgQDF(ks[-3]),
    )[0]
    fig, axs = plot_rf_evolution(weights, figsize=(15,5), cmap='gray')
    fig.savefig(f'results/figures/twolayer_ipr/scm_{seed}.png')
    