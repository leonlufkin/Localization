
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from localization.experiments import load
from localization.utils import sweep_func, product_kwargs

from jax import Array
from typing import Tuple
from functools import partial
import ipdb
import matplotlib.pyplot as plt

def load_or_na(**kwargs):
    try:
        return load(**kwargs)
    except ValueError as e:
        print(e)
        # ipdb.set_trace()
        # return jnp.nan * jnp.zeros((kwargs['num_epochs'] // kwargs['evaluation_interval'] + 1, 1, kwargs['num_dimensions']))
        return None, None # np.nan * np.zeros((1, 1, kwargs['num_dimensions']))

def compute_kurtosis(dataset_cls, freeze, kurtosis_param_name, kurtosis_param_value):
    # get samples
    dataset = dataset_cls(**{**freeze, kurtosis_param_name: kurtosis_param_value})
    exemplars = dataset[:10000][0]
    x = exemplars[:,0]
    # compute kurtosis
    kurtosis = jnp.mean(jnp.power(x, 4)) / jnp.power(jnp.mean(jnp.power(x, 2)), 2)
    return kurtosis

def load_nlgp(gains, seeds, num_dimensions, config):
    # Load results
    sweep, configs = sweep_func(
        load_or_na,
        kwargs_array=product_kwargs(
            **tupify(config),
            seed=tuple(seeds),
            num_dimensions=(num_dimensions,),
            dataset_cls=(datasets.NonlinearGPDataset,), 
            gain=tuple(gains),
        ),
    )
    
    # Extract final weights
    final_weights = np.stack([ s[0][-1,0].__array__() if s[1] is not None else np.nan * np.zeros(num_dimensions) for s in sweep ])
    
    # Compute kurtoses
    kurts = jax.vmap(compute_kurtosis, in_axes=(None, None, None, 0))(
        datasets.NonlinearGPDataset, 
        dict(**config, key=jr.PRNGKey(0), num_dimensions=num_dimensions,),
        'gain', gains,
    )
    
    # Compute IPR summaries
    x = ipr(final_weights).reshape(-1, len(gains))
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    return (m, s, kurts), configs, x

def load_norta(ks, seeds, num_dimensions, config):
    # Load results
    sweep, configs = sweep_func(
        load_or_na,
        kwargs_array=product_kwargs(
            **tupify(config),
            seed=tuple(seeds),
            num_dimensions=(num_dimensions,),
            dataset_cls=(datasets.NortaDataset,),
            marginal_qdf=tuple(datasets.AlgQDF(k) for k in ks)
        ),
    )
    
    # Extract final weights
    final_weights = np.stack([ s[0][-1,0].__array__() if s[1] is not None else np.nan * np.zeros(num_dimensions) for s in sweep ])
    
    # Compute kurtoses
    kurts = jnp.stack(list(map(lambda x: compute_kurtosis(
                datasets.NortaDataset,
                dict(**config, key=jr.PRNGKey(0), num_dimensions=num_dimensions,),
                'marginal_qdf', x
            ),
            [datasets.AlgQDF(k) for k in ks]
        )))
    
    # Compute IPR summaries
    x = ipr(final_weights).reshape(-1, len(ks))
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    return (m, s, kurts), configs, x

if __name__ == '__main__':

    import pandas as pd
    from localization.utils import tupify, ipr
    from localization.experiments.model_sweep import config    
    from localization import datasets, models
    
    # config['batch_size'] = 5000
    # # sweep params
    # # seed = tuple(np.arange(30)),
    # # num_dimensions = (40, 100, 400,)
    # # dataset_cls = (datasets.NonlinearGPDataset, datasets.NortaDataset,)
    gains = jnp.logspace(-2, 2, 10)
    ks = jnp.array([4.1, 4.3, 4.5, 4.74, 5.0, 5.4, 6.1, 7.7, 10., 50.]) # 
    # ks = jnp.linspace(1, 10, 10)
    ks = jnp.concatenate([ks, jnp.linspace(4, 10, 7)])
    num_dimensions = 40
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # NLGP
    config_ = config.copy()
    config_['batch_size'] = 5000
    # (m, s, kurts), configs, iprs = load_nlgp(gains, jnp.arange(30), num_dimensions, config_)
    # np.savez(f'results/figures/ipr_kurtosis/nlgp_kurtosis_vs_ipr_{num_dimensions}.npz', kurts=kurts, m=m, s=s, configs=configs, iprs=iprs)
    data = np.load(f'results/figures/ipr_kurtosis/nlgp_kurtosis_vs_ipr_{num_dimensions}.npz')
    m, s, kurts = data['m'], data['s'], data['kurts']
    # m = np.median(data['iprs'], axis=0)
    ax.scatter(kurts-3, m, c='k', s=10, label='NLGP') # means
    ax.errorbar(kurts-3, m, yerr=s, fmt='o', c='k', alpha=0.5) # stds
    # NORTA
    # (m, s, kurts), configs, iprs = load_norta(ks, jnp.arange(100), num_dimensions, config)
    # np.savez(f'results/figures/ipr_kurtosis/algk_kurtosis_vs_ipr_{num_dimensions}.npz', kurts=kurts, m=m, s=s, configs=configs, iprs=iprs)
    data = np.load(f'results/figures/ipr_kurtosis/algk_kurtosis_vs_ipr_{num_dimensions}.npz')
    m, s, kurts = data['m'], data['s'], data['kurts']
    # m = np.median(data['iprs'], axis=0)
    ipdb.set_trace()
    # m, s, kurts = m[3:], s[3:], kurts[3:]
    ax.scatter(kurts-3, m, c='r', s=10, label='AlgK') # means
    ax.errorbar(kurts-3, m, yerr=s, fmt='o', c='r', alpha=0.5) # stds
    # Labels
    ax.set_xlabel('Excess kurtosis')
    ax.set_ylabel('IPR')
    ax.legend()
    fig.savefig(f'results/figures/ipr_kurtosis/kurtosis_vs_ipr_{num_dimensions}.png', dpi=300, bbox_inches='tight')
    ipdb.set_trace()
    
    
    