import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from localization import datasets
from localization import models
from localization import samplers
from localization.experiments import simulate, simulate_or_load, make_key
from localization.utils import plot_receptive_fields, plot_rf_evolution, build_gaussian_covariance, build_non_gaussian_covariance, entropy_sort, build_DRT
from scipy.special import erf

gaussian_cdf = lambda x: 0.5 * (erf(x/np.sqrt(2)) + 1)

if __name__ == '__main__':
    # define config
    config = dict(
    # data config
    num_dimensions=40,
    xi=(0.1, 3),
    adjust=(-1, 1),
    class_proportion=0.5,
    # model config
    model_cls=models.SimpleNet,
    # activation='relu', use_bias=False, batch_size=1000, init_scale=0.01, learning_rate=0.1, evaluation_interval=10,
    activation='sigmoid', use_bias=True, bias_value=-1, bias_trainable=False, batch_size=1000, init_scale=0.1, learning_rate=0.1, evaluation_interval=10,
    num_hiddens=1,
    sampler_cls=samplers.EpochSampler,
    init_fn=models.xavier_normal_init,
    optimizer_fn=optax.sgd,
    num_epochs=2000,
    # experiment config
    seed=42,#0,
    save_=True,
    )

    # simulate
    weights_nlgp, metrics_nlgp = simulate_or_load(**config, gain=100, dataset_cls=datasets.NonlinearGPDataset)
    print("Metrics")
    print(metrics_nlgp)
    fig, axs = plot_rf_evolution(weights_nlgp[:,:1], figsize=(8, 4))
    fig.savefig(f'/Users/leonlufkin/Documents/GitHub/Localization/localization/results/figures/simplenet_rf.png')