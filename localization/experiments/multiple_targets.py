# MULTIPLE TARGETS
# This script runs 3 kinds of experiments:
# 1. It shows how moving from MSE to CE affects RFs (in 1D) with N targets
# 2. It shows how moving from 2 -> 3 -> 4 -> 10 -> 30 targets with CE changes RFs
# 3. The same as 2., but for MSE

import numpy as np
import optax
from localization import datasets, models, samplers
from localization.experiments import simulate, simulate_or_load, make_key
from localization.utils import ipr, entropy, plot_receptive_fields, plot_rf_evolution, build_gaussian_covariance, build_non_gaussian_covariance, entropy_sort, build_DRT

from argparse import ArgumentParser
parser = ArgumentParser(prog='MULTIPLE TARGETS')
parser.add_argument('experiment', default=0, type=int)

# TODO: need to update batched_online to let me pick the loss function
# DONE!

# TODO: define MSE_to_CE
# DONE, but tune params when landed
def MSE_to_CE(config):
    # Adjust config
    config['xi'] = (0.1, 1.,)
    config.pop('model_cls')
    
    # Get weights for each loss_fn
    weights_mse, _ = simulate_or_load(**config, loss_fn='mse', gain=3., model_cls=models.SimpleNet, dataset_cls=datasets.NonlinearGPDataset)
    weights_ce, _ = simulate_or_load(**config, loss_fn='ce', gain=3., model_cls=models.MLP, dataset_cls=datasets.NonlinearGPDataset)

    # Split into localized & HF noise
    # Not necessary for MSE; doing it for CE
    ce_loc_mask = ipr(weights_ce[-1]) > 0.15

    # Plot weights
    fig_mse, axs_mse = plot_receptive_fields(weights_mse[[-1]], num_cols=1, figsize=(5,5), sort_fn=entropy_sort, ind=-1)
    fig_ce, axs_ce = plot_receptive_fields(weights_ce[[-1]], num_cols=1, figsize=(5,5), sort_fn=entropy_sort, ind=-1)
    
    fig2_mse, axs2_mse = plt.subplots(1, 1, figsize=(10,5))
    axs2_mse.plot(weights_mse[-1].T)
    
    fig2_ce, axs2_ce = plt.subplots(2, 1, figsize=(10,5))
    axs2_ce[0].plot(weights_ce[-1, ce_loc_mask].T)
    axs2_ce[1].plot(weights_ce[-1, ~ce_loc_mask].T)
    
    # Save figs
    fig_mse.savefig('results/figures/multiple_targets/experiment_0/mse.png', dpi=300)
    fig_ce.savefig('results/figures/multiple_targets/experiment_0/ce.png', dpi=300)
    fig2_mse.savefig('results/figures/multiple_targets/experiment_0/mse_profile.png', dpi=300)
    fig2_ce.savefig('results/figures/multiple_targets/experiment_0/ce_profile.png', dpi=300)

def progression(config, loss_fn='ce'):
    xis = tuple(list(np.linspace(0.1, 1, 10)) + list(np.linspace(1, 5, 20)))
    num_targets = [2, 3, 5, 10, 30]
    all_weights = []
    
    # Get weights
    for i, n in enumerate(num_targets):
        xi = xis[::(30//n)]
        weights, _ = simulate_or_load(**config, xi=xi, loss_fn=loss_fn, gain=3., dataset_cls=datasets.NonlinearGPDataset)
        all_weights.append(weights[-1])
    
    # return all_weights
    
    # Plot weights
    fig, axs = plt.subplots(2, 5, figsize=(20,5), sharey=True)
    DRT = build_DRT(40)
    for i, weight in enumerate(all_weights):
        loc_mask = ipr(weight) > 0.1
        #np.linalg.norm(DRT @ weight.T, axis=0) < 1.2
        # ipr(weight) > 0.1
        # np.argmax(np.absolute(DRT.T @ weight), axis=0) < 10
        # print(np.mean(loc_mask))
        # ipr(weight) > 0.1
        # entropy(weight) > 2.
        # ipdb.set_trace()
        axs[0,i].plot(weight[loc_mask].T)
        axs[1,i].plot(weight[~loc_mask].T)
        axs[0,i].set_title(f'{num_targets[i]} targets')
    axs[0,0].set_ylabel('Localized')
    axs[1,0].set_ylabel('Oscillatory')
      
    return fig, axs
    

if __name__ == '__main__':
    
    import ipdb
    import matplotlib.pyplot as plt
    
    # Define config
    config = dict(
        # data config
        num_dimensions=40,
        adjust=(-1, 1),
        class_proportion=0.5,
        # model config
        # model_cls=models.SimpleNet,
        model_cls=models.MLP,
        activation='relu', use_bias=False, batch_size=1000, init_scale=0.01, learning_rate=0.1, evaluation_interval=100,
        # activation='sigmoid', use_bias=True, bias_value=-1, bias_trainable=False, batch_size=1000, init_scale=0.1, learning_rate=0.1, evaluation_interval=10,
        num_hiddens=10,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        optimizer_fn=optax.sgd,
        num_epochs=1000,
        # experiment config
        seed=42,#0,
        save_=True,
    )
    
    # ipdb.set_trace()
    experiment = int(vars(parser.parse_args())['experiment'])
    print(f'Running experiment {experiment:d}')
    
    if experiment == 0:
        MSE_to_CE(config)

    elif experiment == 1:
        fig, _ = progression(config, loss_fn='ce')
        fig.savefig('results/figures/multiple_targets/experiment_1/experiment_1.png', dpi=300)
        
    else:
        fig, _ = progression(config, loss_fn='mse')
        fig.savefig('results/figures/multiple_targets/experiment_2/experiment_2.png', dpi=300)

