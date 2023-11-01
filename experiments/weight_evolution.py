import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

from localization import datasets
from localization import models
from localization import samplers
from localization.experiments import simulate

BUMP_INDEX = 3

nlgp_config = dict(
    key=jax.random.PRNGKey(0),
    dataset_cls=datasets.NonlinearGPDataset,
    xi1=5,
    xi2=1,
    gain=0.05,
    num_dimensions=40,
    num_exemplars=10000,
    support=(-1.0, 1.0), # not implemented rn for nonlinear
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
    num_dimensions=40,
    num_hiddens=2,
    init_scale=1e-3,
    # model_cls=models.SimpleNet,
    # activation='relu',
    model_cls=models.GatedNet,
    activation=bump_gate,
    use_bias=False,
    optimizer_fn=optax.sgd,
    learning_rate=0.5,
    batch_size=1000,
    num_epochs=1000,
    sampler_cls=samplers.EpochSampler,
    init_fn=models.xavier_normal_init,
)

experiment_config = dict(
    seed=0,
    sampler_cls=samplers.EpochSampler,
    class_proportion=0.5,
    save_=False,
    wandb_=False,
)
    

if __name__ == '__main__':
    config = nlgp_config
    
    # theory
    dataset = config['dataset_cls'](**config)
    x_, y_ = dataset[:config['num_exemplars']]
    print(x_.shape, y_.shape)
    g = x_[:, BUMP_INDEX] > 0
    x, y = x_[g], y_[g]
    yx = (y.T @ x) / len(x)
    xx = (x.T @ x) / len(x)
    xx_ = (x_.T @ x_) / len(x_)
    
    xx_inv = jnp.linalg.inv(xx)
    w = xx_inv @ yx
    print(xx.shape, yx.shape, w.shape)
    
    # simulation
    config_ = {**config, **model_config, **experiment_config}
    weights, metrics = simulate(**config_)
    
    # plot it
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    
    im = axs[0,0].imshow(xx_, cmap='gray')
    cbar = fig.colorbar(im, ax=axs[0,0])
    axs[0,0].set_title('xx (full dataset)')
    
    im = axs[0,1].imshow(xx, cmap='gray')
    cbar = fig.colorbar(im, ax=axs[0,1])
    axs[0,1].set_title('xx (gated dataset)')
    
    im = axs[0,2].imshow(xx_inv, cmap='gray')
    cbar = fig.colorbar(im, ax=axs[0,2])
    axs[0,2].set_title('xx_inv (gated dataset)')
    
    axs[1,0].plot(yx)
    axs[1,0].axvline(BUMP_INDEX, color='r')
    axs[1,0].set_title('yx (gated)')
    
    axs[1,1].plot(w)
    axs[1,1].axvline(BUMP_INDEX, color='r')
    axs[1,1].set_title('w (theory)')
    
    print(weights.shape)
    axs[1,2].plot(weights[-1][0])
    axs[1,2].axvline(BUMP_INDEX, color='r')
    axs[1,2].set_title('w (simulated GDLN)')
    
    fig.tight_layout()
    
    # save it
    fig.savefig(f"../thoughts/towards_gdln/figs/weight_evolution/{config['dataset_cls'].__name__}_{config['xi1']}_{config['xi2']}{'_' + str(config['gain']) if 'gain' in config.keys() else ''}.png")
    
    
    
    