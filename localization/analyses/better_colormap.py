import jax
import jax.numpy as jnp
from localization import datasets, models, samplers
import ipdb

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
        seed=0, # 0
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
    cmap = 'cb.pregunta' # 'cb.solstice'
    fig, axs = plot_rf_evolution(w_model, figsize=(5, 3), cmap=cmap)
    fig.savefig(f'results/figures/better_colormap/{cmap}.png', dpi=300)
    # ipdb.set_trace()
    