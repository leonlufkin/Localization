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
    
    # seed=0_L=100_g=100.0_is=0.001_lr=0.01_b=50000_xi=0.3,0.7_T=5000
    # seed=1_L=100_g=100.0_is=0.001_lr=0.01_b=50000_xi=0.3,0.7_T=5000
    # seed=0_L=100_g=0.01_is=0.001_lr=0.05_b=50000_xi=0.3,0.7_T=5000
    
    # Config
    c = dict(
        seed=0,#2,#3,#1,#0, # 0
        num_dimensions=100, # 100
        num_hiddens=1,
        dim=1,
        gain=0.01,#100.,#0.01,#100,#0.01,
        init_scale=0.001, # 0.001
        activation='relu',
        model_cls=models.SimpleNet,
        use_bias=False,
        optimizer_fn=optax.sgd,
        learning_rate=0.05,#0.01,
        batch_size=25000,#50000,#10000,
        num_epochs=20000,#10000,#5000,
        dataset_cls=datasets.NonlinearGPDataset,
        xi=(0.3, 0.7), #(0.7, 0.3,),
        # num_steps=10000,
        adjust=(-1.0, 1.0),
        class_proportion=0.5,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        loss_fn='mse',
        save_=True,
        evaluation_interval=200,#100,#50,
    )
    w_model = simulate_or_load(**c)[0][:,0]
    mini_key = f'seed={c["seed"]}_L={c["num_dimensions"]}_g={c["gain"]}_is={c["init_scale"]}_lr={c["learning_rate"]}_b={c["batch_size"]}_xi={c["xi"][0]},{c["xi"][1]}_T={c["num_epochs"]}'
    # ipdb.set_trace()
    dir = f'results/figures/numerical_integration/{mini_key}/'
    os.makedirs(dir, exist_ok=True)
    fig, axs = plot_rf_evolution(w_model, figsize=(4, 2), cmap='cb.solstice')
    # fig.savefig(f'{dir}/empirical.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir}/empirical.pdf', bbox_inches='tight')
    
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
    
    fig, axs = plot_rf_evolution(w_sim, figsize=(4, 2), cmap='cb.solstice')
    # fig.savefig(f'{dir}/analytical.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir}/analytical.pdf', bbox_inches='tight')
    
    # Prep time axis
    time_axis = jnp.arange(c['num_epochs'] // c['evaluation_interval'] + 1) * c['evaluation_interval'] * c['learning_rate']
    
    # Compare IPRs and difference across time
    from localization.utils import ipr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.67, 2))
    fig.subplots_adjust(wspace=0.375)
    ax1.plot(time_axis, ipr(w_model), label='Empirical')
    ax1.plot(time_axis, ipr(w_sim), label='Analytical')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('IPR')
    ax1.set_title('IPR')
    ax1.legend()
    diff = jnp.sqrt(jnp.square(w_model - w_sim).mean(axis=1))
    ax2.plot(time_axis, diff)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('$\ell_2$(model - analytical)')
    ax2.set_title('Error')
    # fig.savefig(f'{dir}/ipr_mse.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir}/ipr_mse.pdf', bbox_inches='tight')
    
    # Plot timeshot after t1 (close), t2 (diverging), and -1 steps
    t1, t2 = 40, 60 # seed = 0
    # t1, t2 = 24, 40 # seed = 1
    # t1, t2 = 25, 35 # seed = 2
    # t1, t2 = 30, 50 # seed = 3
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 2), sharey=True)
    # t1 steps
    ax1.plot(w_model[t1], label='Empirical')
    ax1.plot(w_sim[t1], label='Analytical')
    ax1.set_xlabel(r'dimension $i$ of weight $\mathbf{w}$')
    ax1.set_ylabel('Weight magnitude')
    ax1.set_title(f'Weights at t={int(time_axis[t1]):d}')
    # t2 steps
    ax2.plot(w_model[t2], label='Empirical')
    ax2.plot(w_sim[t2], label='Analytical')
    ax2.set_xlabel(r'dimension $i$ of weight $\mathbf{w}$')
    # ax2.set_ylabel('Weight magnitude')
    ax2.set_title(f'Weights at t={int(time_axis[t2]):d}')
    # -1 steps
    ax3.plot(w_model[-1], label='Empirical')
    ax3.plot(w_sim[-1], label='Analytical')
    ax3.set_xlabel(r'dimension $i$ of weight $\mathbf{w}$')
    # ax3.set_ylabel('Weight magnitude')
    ax3.set_title(f'Weights at t={int(time_axis[-1]):d}')
    ax3.legend()
    # fig.savefig(f'{dir}/timeshot.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir}/timeshot.pdf', bbox_inches='tight')
    
    # Compare IPRs and difference across time
    from localization.utils import ipr
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 2))
    # for ax in (ax1, ax2, ax3):
    #     ax.yaxis.labelpad = 15
    fig.subplots_adjust(wspace=0.275)
    ax1.plot(time_axis, ipr(w_model), label='Empirical')
    ax1.plot(time_axis, ipr(w_sim), label='Analytical')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('IPR')
    ax1.set_title('IPR')
    ax1.legend()
    diff = jnp.sqrt(jnp.square(w_model - w_sim).mean(axis=1))
    ax2.plot(time_axis, diff)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'$\ell_2$(model - analytical)')
    if c['seed'] == 1:
        ax2.set_yticks([0., 0.01, 0.02, 0.03])
    ax2.set_title('Error')
    # Zooming in for ax3
    for line in ax1.get_lines():
        ax3.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
    ax3.set_xlabel('Time')
    ax3.set_ylabel('IPR')
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_ylim((0,0.1))
    ax3.axhline(y=0.03, color='r', linestyle='--')
    ax3.set_title('Zoomed in')
    # ax3.legend()
    # fig.savefig(f'{dir}/ipr_mse_zoomed_in.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir}/ipr_mse_zoomed_in.pdf', bbox_inches='tight')
        
    
    
    