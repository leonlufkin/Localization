
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from localization import datasets, models, samplers
from localization.experiments import load, simulate_or_load
from localization.utils import sweep_func, product_kwargs, tupify, ipr, plot_rf_evolution

from jax import Array
from typing import Tuple
from functools import partial
import ipdb
import matplotlib.pyplot as plt

def fit_sinusoid(
    w: Array, 
    init: Tuple[float, float, float, float],
    learning_rate: float = 1e-2,
    max_iterations: int = 10000,
    min_iterations: int = 1000,
    convergence_threshold: float = 1e-8,
    optimizer_fn = optax.adam,
):
    
    f_init, amp_init, x_shift_init, y_shift_init = init
    x = jnp.linspace(0, 1, len(w))
    
    # Initialize parameters and optimizer
    init_params = jnp.array([f_init, amp_init, x_shift_init, y_shift_init])
    optimizer = optimizer_fn(learning_rate)
    opt_state = optimizer.init(init_params)
    
    def predict(f, amp, x_shift, y_shift):
        return amp * jnp.sin( 2 * jnp.pi * f * (x - x_shift) ) + y_shift
    
    def loss_fn(params):
        f, amp, x_shift, y_shift = params
        return jnp.mean((w - predict(f, amp, x_shift, y_shift)) ** 2)
        
    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    # Optimization loop
    params = init_params
    prev_loss = float('inf')
    losses = jnp.zeros(max_iterations)
    for i in range(max_iterations):
        params, opt_state, current_loss = update(params, opt_state)
        losses = losses.at[i].set(current_loss)
        
        if i > min_iterations:
            if jnp.sqrt(current_loss) < convergence_threshold:
                break
        # prev_loss = current_loss

    fitted_values = predict(*params)
    return params, fitted_values, losses[:i+1]
    

if __name__ == '__main__':


    from localization.experiments.model_sweep import config    
    
    data_config = dict(
        # Data
        key=jax.random.PRNGKey(0),
        num_dimensions=40,#100, 
        dim=1,
        xi=(1, 3,),
        adjust=(-1.0, 1.0),
        class_proportion=0.5,
    )

    config = dict(
        # Model
        num_hiddens=1,
        init_scale=0.001,
        activation='relu',
        model_cls=models.SimpleNet,
        use_bias=False, bias_trainable=False, bias_value=0.0,
        optimizer_fn=optax.sgd, 
        learning_rate=0.05,
        num_steps=5000, num_epochs=5000,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        loss_fn='mse',
        save_=True,
        evaluation_interval=100,
        # Misc
        supervise=True,
        wandb_=False, 
    )
    config.update(data_config)
    config.pop('key')
    
    n = data_config['num_dimensions']
    
    
    # # Elliptical Extreme
    # weights_weird, metrics = simulate_or_load(seed=42, dataset_cls=datasets.EllipticalDataset, inverse_cdf=lambda x: -0.5 * jnp.log(2/(x+1) - 1) + 2, batch_size=10000, **config)
    
    # fitparams, wfit, losses = fit_sinusoid(
    #     weights_weird[-1,0],
    #     init = (1., 0.005, 0.4, 0.04),
    #     optimizer_fn = optax.sgd,
    #     min_iterations = 1000,
    # )
    # print(losses[-1]) # 2.71e-06
    # print(jnp.sqrt(losses[-1] * n) / jnp.sqrt(jnp.sum(weights_weird[-1,0] ** 2)))
    
    # fig, ax = plot_rf_evolution(weights_weird[:,[0],:], figsize=(4, 2), cmap='cb.solstice')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.plot(wfit, c='r', linestyle='--')
    # fig.savefig(f'results/figures/ellipticals/extreme.png', bbox_inches='tight', dpi=300)
    # # ipdb.set_trace()
    
    
    # Student-t
    config_ = config.copy(); config_.pop('learning_rate'); config_.pop('num_steps'); config_.pop('num_epochs'); config_.pop('xi'); config_.pop('num_dimensions'); config_.pop('init_scale')
    weights_t, metrics = simulate_or_load(seed=0, dataset_cls=datasets.TDataset, df=3, 
                                          batch_size=50000,#1000, 
                                          xi=(1, 3,), num_dimensions=40, learning_rate=0.01, 
                                          num_epochs=10000, #evaluation_interval=100, 
                                          num_steps=10000, 
                                          init_scale=0.001,#1, 
                                          **config_)
    
    fitparams, wfit, losses = fit_sinusoid(
        weights_t[-1,0],
        init = (1., 0.05, 0., 0.),
        optimizer_fn = optax.sgd,
        min_iterations = 1000,
    )
    print(losses[-1]) # 0.000289
    print(jnp.sqrt(losses[-1] * n) / jnp.sqrt(jnp.sum(weights_t[-1,0] ** 2)))
    
    fig, ax = plot_rf_evolution(weights_t[:,[0],:], figsize=(4, 2), cmap='cb.solstice')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(wfit, c='r', linestyle='--')
    fig.savefig(f'results/figures/ellipticals/t3.png', bbox_inches='tight', dpi=300)
    # ipdb.set_trace()
    
    
    # # Elliptical Shell
    # weights_shell, metrics = simulate_or_load(seed=0, dataset_cls=datasets.EllipticalDataset, inverse_cdf=lambda x: 1., batch_size=10000, **config)
    # fig, ax = plot_rf_evolution(weights_shell[:,[0],:], figsize=(4, 2), cmap='cb.solstice')
    
    # fitparams, wfit, losses = fit_sinusoid(
    #     weights_shell[-1,0],
    #     init = (1., 0.005, 0.25, -0.04),
    #     optimizer_fn = optax.sgd,
    #     min_iterations = 1000,
    # )
    # print(losses[-1]) # 2.43e-06
    # print(jnp.sqrt(losses[-1] * n) / jnp.sqrt(jnp.sum(weights_shell[-1,0] ** 2)))
    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.plot(wfit, c='r', linestyle='--')
    # fig.savefig('results/figures/ellipticals/shell.png', bbox_inches='tight', dpi=300)
    # # ipdb.set_trace()
    