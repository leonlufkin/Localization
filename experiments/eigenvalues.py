import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import os

from localization import datasets
from localization import models
from localization import samplers
from localization.experiments import simulate, make_key

from scipy.stats import kurtosis

BUMP_INDEX = 3

nlgp_config = dict(
    key=jax.random.PRNGKey(0),
    dataset_cls=datasets.NonlinearGPDataset,
    xi1=5,
    xi2=1,
    gain=3,#0.05,
    num_dimensions=40,
    num_exemplars=10000,
    support=(-1.0, 1.0), # not implemented rn for nonlinear
)

sp_config = dict(
    key=jax.random.PRNGKey(0),
    dataset_cls=datasets.SinglePulseDataset,
    xi1=(0.25, 0.275),
    xi2=(0.05, 0.075),
    num_dimensions=40,
    num_exemplars=10000,
    support=(0.0, 1.0), # not implemented rn for nonlinear
)

bump = jnp.array([1. if i == BUMP_INDEX else 0. for i in range(40)])

def bump_gate(x):
    """
    The code below is a differentiable version of:
        pos_pre = jnp.dot(x, pos_bump)
        if pos_pre >= 0:
            return jnp.array([1., 0.])
        return jnp.array([0., 1.])
    """
    first_pre = jnp.dot(x, bump)
    w_ = first_pre / (jnp.abs(first_pre) + 1e-8)
    w = (w_ + 1) / 2
    return jnp.array([w, 1 - w])

model_config = dict(
    num_hiddens=12, 
    init_scale=1e-3,#1e-1,
    model_cls=models.SimpleNet,
    activation='sigmoid',
    # model_cls=models.GatedNet,
    # activation=bump_gate,
    use_bias=True,
    optimizer_fn=optax.sgd,
    learning_rate=10.0,#0.25, #0.5,
    batch_size=1000,
    num_epochs=2500,
    sampler_cls=samplers.EpochSampler,
    init_fn=models.xavier_normal_init,
)

experiment_config = dict(
    seed=0,
    sampler_cls=samplers.EpochSampler,
    class_proportion=0.5,
    save_=False,
    wandb_=False,
    evaluation_interval=20,
)

def simulate_or_load(**kwargs):
    path_key = make_key(**kwargs)
    if path_key + '.npz' in os.listdir('../localization/results/weights'):
        print('Already simulated')
        data = np.load('../localization/results/weights/' + path_key + '.npz', allow_pickle=True)
        weights_, metrics_ = data['weights'], data['metrics']
    else:
        print('Simulating')
        weights_, metrics_ = simulate(**kwargs)
    return weights_, metrics_
    

if __name__ == '__main__':
    config = nlgp_config
    # config = sp_config
    
    # simulation
    config_ = {**config, **model_config, **experiment_config}
    weights, metrics = simulate_or_load(**config_)
    
    dataset = config['dataset_cls'](**config)
    x, y = dataset[:config['num_exemplars']]
    
    preacts = jnp.tensordot(x, weights, axes=[(1,), (2,)])
    postacts = jax.nn.relu(preacts)
    
    sign = jnp.sign(postacts)
    # check if sign at time t is the same as sign at time t+1
    flipped_ = jnp.abs(sign[:, 1:] - sign[:, :-1])
    flipped = 0.5 * jnp.mean(flipped_, axis=0)
    
    # compute the loss, not just the accuracy
    flipped_loss_ = flipped_ * preacts[:, 1:]
    flipped_loss = (flipped > 0).astype(int) * jnp.mean(jnp.square(flipped_loss_), axis=0) / (flipped + 1e-12)
    
    # compute std & kurtosis of pre- and post-activations
    preact_std, postact_std = jnp.std(preacts, axis=0), jnp.std(postacts, axis=0)
    preact_kurt, postact_std = kurtosis(preacts, axis=0), kurtosis(postacts, axis=0)
    kurt_min, kurt_max = np.min(preact_kurt), np.max(preact_kurt)
    
    # plot it
    num_rows = np.sqrt(flipped.shape[-1]).astype(int) #min(2 * , flipped.shape[-1])
    num_cols = np.ceil(flipped.shape[-1] / num_rows).astype(int)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10), sharex=True, sharey=True)
    x_ = np.arange(flipped.shape[0] + 1) * experiment_config['evaluation_interval']
    for i, ax in enumerate(axs.flatten()):
        ax.plot(x_[:-1], flipped[:, i], color='#1f77b4', label='Flips' if i == 0 else None)
        # ax_.plot(x_, flipped_loss[:, i], color='#ff7f0e')
        ax_ = ax.twinx()
        # ax_.plot(x_, preact_std[:, i], color='#ff7f0e', label='Std' if i == 0 else None)
        preact_kurt_ = preact_kurt[:, i]
        ax_.plot(x_, preact_kurt_, color='#2ca02c', label='Kurt' if i == 0 else None)
        ax_.set_ylim(kurt_min, kurt_max)
        if np.any(np.abs(preact_kurt_) > 0.6):
            transition_x = np.nonzero(np.abs(preact_kurt_) > 0.6)[0][0]
            # ax_.axhline(0.5, color='k', linestyle='--')
            ax_.axvline(transition_x * experiment_config['evaluation_interval'], color='k', linestyle='--')
            
        ax.set_title(f"Neuron {i}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction of sign flips")
        ax_.set_ylabel("Kurtosis of preactivations")
        ax.set_xscale('log')
        
        ax__ = ax.twinx().twiny()
        ax__.plot(weights[-1,i], color='#d62728', alpha=0.25, label='Final weights' if i == 0 else None)
        ax__.set_xticks([])
        ax__.set_yticks([])
        
    fig.legend(loc='upper right')
        
    # remove unused axes
    for i in range(flipped.shape[-1], axs.size):
        fig.delaxes(axs.flatten()[i])
    
    fig.tight_layout()
    fig.savefig(f"../thoughts/towards_gdln/figs/sign_flipping/{config['dataset_cls'].__name__}_{config['num_dimensions']}_{model_config['num_hiddens']}_{config['xi1']}_{config['xi2']}{'_' + str(config['gain']) if 'gain' in config.keys() else ''}_{model_config['init_scale']}.png")
    
    
    
    
    