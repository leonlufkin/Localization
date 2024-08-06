
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
    
    # config.update(dict(
    #     batch_size=5000,
    #     num_epochs=10000,
    #     learning_rate= 0.1,
    #     num_hiddens=1,
    # ))
    
    config.update(dict(
        batch_size=5000,
        num_epochs=10000,#100000,#
        evaluation_interval=100,
        learning_rate=10.,#5.,
        num_hiddens=10,
    ))
    
    # sweep params
    # seed = tuple(np.arange(30)),
    # num_dimensions = (40, 100, 400,)
    # dataset_cls = (datasets.NonlinearGPDataset, datasets.NortaDataset,)
    gains = jnp.logspace(-2, 2, 10)
    ks = jnp.array([4.1, 4.3, 4.5, 4.74, 5.0, 5.4, 6.1, 7.7, 10., 50.]) 
    ks = jnp.concatenate([ks, jnp.linspace(4, 10, 7)])
    num_dimensions = 40
    problem_seed = 11 # 20
    
    # ORIGINAL, from cluster
    weights = simulate_or_load(
        **config, 
        seed=problem_seed,
        num_dimensions=40,
        dataset_cls=datasets.NortaDataset,
        marginal_qdf=datasets.AlgQDF(ks[0]), # (ks[8]),
    )[0]
    fig, ax = plot_rf_evolution(weights, num_rows=2, num_cols=5, figsize=(15,5), cmap='cb.pregunta')
    fig.savefig(f'results/figures/high_ipr/original_{problem_seed}.png')
    ipdb.set_trace()
    