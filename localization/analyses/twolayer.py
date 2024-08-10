import numpy as np
import jax
import jax.numpy as jnp
import optax
from localization import datasets, samplers, models, experiments
from jaxnets.utils import make_key, simulate_or_load, mse
import ipdb

if __name__ == '__main__':
  
  import matplotlib.pyplot as plt
  from localization.utils import plot_rf_evolution
  from localization.utils import build_non_gaussian_covariance
  
  # define config
  config = dict(
    task=experiments.supervise,
    config_modifier=experiments.supervise_update_config,
    seed=0,
    num_dimensions=40,
    dim=1,
    hidden_size=1,
    # gain=100,
    gain=0.01,
    init_scale=0.001,
    # init_scale=1.0,
    # init_scale=10.0,
    activation=jax.nn.relu, #'relu',
    # model_cls=models.SCM,
    model_cls=models.MLP,
    use_bias=False,
    optimizer_fn=optax.sgd,
    learning_rate=0.1,
    batch_size=1000,
    num_epochs=10000,
    datset_cls=datasets.NonlinearGPDataset,
    xi=(0.3, 0.7),
    sampler_cls=samplers.DirectSampler,
    init_fn=models.xavier_normal_init,
    loss_fn=mse, #'mse',
    save_weights=True,
    save_model=True,
    evaluation_interval=100,
  )
  
  for gain in np.logspace(-1, 1, 10):
    config['gain'] = gain
    metrics, weights, model = simulate_or_load(**config)
      
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # First layer
    fig, ax1 = plot_rf_evolution(weights[0], fig=fig, axs=ax1)
    ax1.set_title('First layer')
    
    # Second layer
    w2s = weights[1].flatten()
    ax2.plot(w2s)
    ax2.set_title('Second layer')
    # Get theoretical w2 value from w1
    w1 = weights[0][-1,0]
    Sigma0 = build_non_gaussian_covariance(config['num_dimensions'], xi=config['xi'][0], g=config['gain'])
    Sigma1 = build_non_gaussian_covariance(config['num_dimensions'], xi=config['xi'][1], g=config['gain'])
    top = jnp.sqrt(w1 @ Sigma1 @ w1)
    bottom = w1 @ (Sigma0 + Sigma1) @ w1
    pred = jnp.sqrt(2/jnp.pi) * top / bottom
    ax2.axhline(pred, color='r', linestyle='--')
    # ax2.set_ylim(min(w2s.min()), max(w2s.max()) + 0.1)
    
    # Save
    fig.savefig(f"results/figures/twolayer/g={config['gain']}_is={config['init_scale']}.png")
  ipdb.set_trace()