# import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import entropy as scipy_entropy
# from localization.utils.measurement import 

#########################
# VISUALIZATION FUNCTIONS

def plot_receptive_fields(weights: list, num_cols=None, evaluation_interval=500, figsize=(10, 5), reordering_fn=None, **reordering_kwargs):
    num_cols = num_cols or len(weights)
    num_rows = int(np.ceil(len(weights) / num_cols))
    
    # determine ordering
    reordering = reordering_fn(weights=weights, **reordering_kwargs) if reordering_fn is not None else np.arange(weights.shape[1])
    
    # plot each set of weights
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs_ = axs.flatten()
    min_, max_ = np.min([np.min(weight_) for weight_ in weights]), np.max([np.max(weight_) for weight_ in weights])
    for i, (weight, ax) in enumerate(zip(weights, axs_)):
        ax.imshow(weight[reordering], cmap='gray', vmin=min_, vmax=max_)
        ax.set_xlabel(evaluation_interval * i)
        ax.set_xticks([])
        if i == 0:
            # ax.set_xlabel('Input neurons')
            ax.set_ylabel('Hidden neurons')
        else:
            ax.set_yticks([])
            
    # remove unused axes
    for ax in axs_.flatten()[len(weights):]:
        ax.remove()
    
    return fig, axs

def plot_rf_evolution(weights, num_rows=1, num_cols=1, figsize=(15, 5), cmap='gray'):
    # num_rows * num_cols is the number of receptive fields to plot
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
    cmap_ = plt.get_cmap(cmap)
    color = cmap_(np.linspace(0.2, 0.8, weights.shape[0]))
    for i, ax in enumerate(axs.flatten() if isinstance(axs, np.ndarray) else [axs]):
        for t in range(weights.shape[0]):
            ax.plot(weights[t,i,:], color=color[t], alpha=0.5)
    return fig, axs