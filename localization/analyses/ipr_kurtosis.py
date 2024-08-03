
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
        return None, jnp.nan * jnp.zeros((1, 1, kwargs['num_dimensions']))

def compute_kurtosis(dataset_cls, freeze, kurtosis_param_name, kurtosis_param_value):
    # get samples
    dataset = dataset_cls(**{**freeze, kurtosis_param_name: kurtosis_param_value})
    exemplars = dataset[:10000][0]
    x = exemplars[:,0]
    # compute kurtosis
    kurtosis = jnp.mean(jnp.power(x, 4)) / jnp.power(jnp.mean(jnp.power(x, 2)), 2)
    return kurtosis

if __name__ == '__main__':

    import pandas as pd
    from localization.utils import tupify, ipr
    from localization.experiments.model_sweep import config    
    from localization import datasets, models
    
    config['batch_size'] = 5000
    # sweep params
    # seed = tuple(np.arange(30)),
    # num_dimensions = (40, 100, 400,)
    # dataset_cls = (datasets.NonlinearGPDataset, datasets.NortaDataset,)
    gains = jnp.logspace(-2, 2, 10)
    # ks = jnp.array([4.1, 4.3, 4.5, 4.74, 5.0, 5.4, 6.1, 7.7, 10., 50.]) # 
    ks = jnp.linspace(1, 10, 10)
    num_dimensions = 40
    
    sweep, configs = sweep_func(
        load_or_na,
        # lambda **kwargs: load(**kwargs)[1], # just weights
        kwargs_array=product_kwargs(
            **tupify(config),
            seed=tuple(np.arange(30,)),
            num_dimensions=(num_dimensions,),
            dataset_cls=(datasets.NonlinearGPDataset,), gain=tuple(gains),
            # dataset_cls=(datasets.NortaDataset,), marginal_qdf=tuple(datasets.AlgQDF(k) for k in ks)
        ),
    )
    configs = pd.DataFrame(configs)
    # configs['gain'] = configs['gain'].astype(float)
    # configs['k'] = jnp.stack([ c.k for c in configs['marginal_qdf'] ])
    # metrics = jnp.stack([ s[0] for s in sweep ])
    # weights = jnp.stack([ s[1] for s in sweep ])
    # final_weights = weights[:, -1, 0, :]
    final_weights = jnp.stack([ s[1][-1,0] for s in sweep ])
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0,0].plot(final_weights[19])
    axs[0,1].plot(final_weights[39])
    axs[1,0].plot(final_weights[1])
    axs[1,1].plot(final_weights[31])
    # fig.savefig(f'results/figures/ipr_kurtosis/nlgp_debug_{num_dimensions}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'results/figures/ipr_kurtosis/algk_debug_{num_dimensions}.png', dpi=300, bbox_inches='tight')
    
    # plot gain vs ipr
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # kurts = jax.vmap(compute_kurtosis, in_axes=(None, None, None, 0))(
    #     datasets.NonlinearGPDataset, 
    #     dict(**config, key=jr.PRNGKey(0), num_dimensions=num_dimensions,),
    #     'gain', gains,
    # )
    kurts = jnp.stack(list(map(lambda x: compute_kurtosis(
                datasets.NortaDataset,
                dict(**config, key=jr.PRNGKey(0), num_dimensions=num_dimensions,),
                'marginal_qdf', x
            ),
            [datasets.AlgQDF(k) for k in ks]
        )))
    print(kurts)
    # iprs = configs[['gain']].assign(ipr=ipr(final_weights)).groupby('gain').agg(['mean', 'std']).to_numpy()
    iprs = configs[['k']].assign(ipr=ipr(final_weights)).groupby('k').agg(['mean', 'std']).to_numpy()
    print(iprs)
    kurts = kurts[3:]
    iprs = iprs[3:]
    ax.scatter(kurts-3, iprs[:,0], c='k', s=10) # means
    ax.errorbar(kurts-3, iprs[:,0], yerr=iprs[:,1], fmt='o', c='k', alpha=0.5) # stds
    ax.set_xlabel('Kurtosis')
    ax.set_ylabel('IPR')
    fig.savefig(f'results/figures/ipr_kurtosis/nlgp_kurtosis_vs_ipr_{num_dimensions}.png', dpi=300, bbox_inches='tight')
    # fig.savefig(f'results/figures/ipr_kurtosis/algk_kurtosis_vs_ipr_{num_dimensions}.png', dpi=300, bbox_inches='tight')
    ipdb.set_trace()
    
    
    