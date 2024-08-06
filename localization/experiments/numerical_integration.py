# PDE SIMULATION
# This script numerically integrates the gradient flow differential equation governing our single-ReLU-neuron neural net's dynamics.
# It lets us easily:
#   1. manipulate starting weights,
#   2. manipulate the covariance function,
#   3. manipulate other aspects of the model (like number of units), and
#   4. maniuplate some aspects of the data (like relative size of classes) 

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from typing import Callable

from localization import datasets, models, samplers
from localization.utils import plot_rf_evolution

from functools import partial
from jax.scipy.special import erf
from tqdm import tqdm

import ipdb

# gaussian_cdf = lambda x: 0.5 * (erf(x/np.sqrt(2)) + 1)

def alginv(x):
    return x / jnp.sqrt(1 - x ** 2)

def phi(x):
    return erf(alginv(x) / jnp.sqrt(2))

def integrate(
    w_init: Array,
    cov1: Array,
    cov0: Array,
    tau: float,
    num_steps: int,
    phi: Callable = phi,
    return_all: bool = False
):
    phi_vmap = jax.vmap(phi)
    
    def step(w):
        x = jnp.dot(cov1, w) / jnp.sqrt(jnp.dot(w, jnp.dot(cov1, w)))
        io = phi_vmap(x)
        oo = jnp.dot(cov0 + cov1, w)
        w_new = w + (tau/2) * (io - oo)
        return w_new
    
    if return_all:
        # use jax.lax.scan to return all intermediate weights
        _, w = jax.lax.scan(lambda w, _: 2 * (step(w),), w_init, jnp.arange(num_steps))
    else:
        # use jax.lax.fori_loop to avoid python loop overhead
        w = jax.lax.fori_loop(0, num_steps, lambda i, w: step(w), w_init)
        
    # append init weight, to be consistent with the empirical weights
    w = jnp.concatenate([w_init[None], w], axis=0) 
        
    return w

def mse_aligned(w1, w2):
    """
    Accounting for possible slight differences in centering of receptive fields.
    There are two ways to do this:
    1. 
    """
    return jnp.square(w1 - w2).mean()

if __name__ == '__main__':
    
    from localization.utils import build_gaussian_covariance, build_non_gaussian_covariance, plot_rf_evolution, plot_receptive_fields
    from localization.experiments import simulate, load, simulate_or_load
    import optax
    import os
    
    # Check dtype 
    x = jax.random.uniform(jax.random.key(0), (1000,), dtype=jnp.float64)
    print(x.dtype) # --> dtype('float64')
    
    # seed=0_L=100_g=100.0_is=0.001_lr=0.01_b=50000_xi=0.3,0.7_T=5000
    
    # Config
    c = dict(
        seed=1,#0, # 0
        num_dimensions=100, # 100
        num_hiddens=1,
        dim=1,
        gain=100.,#0.01,#100,#0.01,
        init_scale=0.001, # 0.001
        activation='relu',
        model_cls=models.SimpleNet,
        use_bias=False,
        optimizer_fn=optax.sgd,
        learning_rate=0.01,
        batch_size=50000,#10000,
        num_epochs=5000,
        dataset_cls=datasets.NonlinearGPDataset,
        xi=(0.3, 0.7), #(0.7, 0.3,),
        # num_steps=10000,
        adjust=(-1.0, 1.0),
        class_proportion=0.5,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        loss_fn='mse',
        save_=True,
        evaluation_interval=50,
    )
    w_model = simulate_or_load(**c)[0][:,0]
    mini_key = f'seed={c["seed"]}_L={c["num_dimensions"]}_g={c["gain"]}_is={c["init_scale"]}_lr={c["learning_rate"]}_b={c["batch_size"]}_xi={c["xi"][0]},{c["xi"][1]}_T={c["num_epochs"]}'
    # ipdb.set_trace()
    dir = f'results/figures/numerical_integration/{mini_key}/'
    os.makedirs(dir, exist_ok=True)
    fig, axs = plot_rf_evolution(w_model, figsize=(15, 5), cmap='gray')
    fig.savefig(f'{dir}/empirical.png')
    
    w_init = w_model[0]
    xi = c['xi']
    cov1 = build_non_gaussian_covariance(c['num_dimensions'], xi=xi[1], g=c['gain'])
    cov0 = build_non_gaussian_covariance(c['num_dimensions'], xi=xi[0], g=c['gain'])
    
    w_sim = integrate(
        w_init=w_init,
        cov1=cov1,
        cov0=cov0,
        tau=c['learning_rate'],
        num_steps=c['num_epochs'],
        # phi=phi, # this is the correct phi for large g
        # phi=lambda x: jnp.sqrt(2/jnp.pi) * x,
        phi=phi if c['gain'] > 1 else lambda x: jnp.sqrt(2/jnp.pi) * x, # high/low gain
        return_all=True
    )
    eval_int = c['evaluation_interval']
    w_sim = w_sim[::eval_int]
    
    fig, axs = plot_rf_evolution(w_sim, figsize=(15, 5), cmap='gray')
    fig.savefig(f'{dir}/analytical.png')
    
    # Compare IPRs and difference across time
    from localization.utils import ipr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ipr(w_model), label='Empirical')
    ax1.plot(ipr(w_sim), label='Analytical')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('IPR')
    ax1.set_title('IPR')
    ax1.legend()
    diff = jnp.sqrt(jnp.square(w_model - w_sim).mean(axis=1))
    ax2.plot(diff)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('RMSE')
    ax2.set_title('l2(model - analytical)')
    fig.savefig(f'{dir}/ipr_mse.png')
    
    # Plot timeshot after 40, 60, 100 steps
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # 40 steps
    ax1.plot(w_model[40], label='Empirical')
    ax1.plot(w_sim[40], label='Analytical')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Weight magnitude')
    ax1.set_title('Weights at t=40')
    # 60 steps
    ax2.plot(w_model[60], label='Empirical')
    ax2.plot(w_sim[60], label='Analytical')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Weight magnitude')
    ax2.set_title('Weights at t=60')
    # 100 steps
    ax3.plot(w_model[100], label='Empirical')
    ax3.plot(w_sim[100], label='Analytical')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Weight magnitude')
    ax3.set_title('Weights at t=100')
    ax3.legend()
    fig.savefig(f'{dir}/timeshot.png')
    
    # Compare IPRs and difference across time
    from localization.utils import ipr
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    ax1.plot(ipr(w_model), label='Empirical')
    ax1.plot(ipr(w_sim), label='Analytical')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('IPR')
    ax1.set_title('IPR')
    ax1.legend()
    diff = jnp.sqrt(jnp.square(w_model - w_sim).mean(axis=1))
    ax2.plot(diff)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('RMSE')
    ax2.set_title('l2(model - analytical)')
    # Zooming in for ax3
    for line in ax1.get_lines():
        ax3.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_ylim((0,0.1))
    ax3.axhline(y=0.03, color='r', linestyle='--')
    ax3.legend()
    fig.savefig(f'{dir}/ipr_mse_zoomed_in.png')
        
    
    
    