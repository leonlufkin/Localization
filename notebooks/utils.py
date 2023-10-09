import os
import numpy as np
import matplotlib.pyplot as plt
# import jax
# import jax.numpy as jnp
# import optax
# import datasets
# import models
# import samplers
from scipy.stats import entropy as scipy_entropy


######################
# LOCALIZATION METRICS

def ipr(weights):
    return np.sum(np.power(weights, 4), axis=1) / np.sum(np.square(weights), axis=1) ** 2

def entropy(weights, low=-10, upp=10, delta=0.1, base=2):
    entropies = np.zeros(weights.shape[0])
    for neuron, weight in enumerate(weights):
        xs = np.arange(low, upp, delta)
        count = np.zeros(len(xs)+1)
        count[0] = np.sum(weight < xs[0])
        for i in range(len(xs)-1):
            count[i] = np.sum(weight < xs[i+1]) - np.sum(weight < xs[i])
        count[-1] = np.sum(weight >= xs[-1])
        prob = count / np.sum(count)
        entropies[neuron] = scipy_entropy(prob, base=base)
    return entropies

def position_mean_var(weight):
    # use weights to construct probability distribution
    magnitude = np.square(weight)
    magnitude /= np.sum(magnitude, axis=-1).reshape(-1,1)

    # compute first and second order stats
    L = weight.shape[-1]
    x = np.arange(L)
    first = np.sum(x * magnitude, axis=-1)
    second = np.sum(x * x * magnitude, axis=-1)
    var = (second - first ** 2) 
    return first, var




###################
# SORTING FUNCTIONS

def entropy_sort(weights: list, ind: int = -1, center_sort=False):
    weight_ = weights[ind]
    l = weight_.shape[0]
    entropy_ = entropy(weight_)
    reordering = np.argsort(entropy_)
    if center_sort:
        conv_ = weight_[reordering[:l//2]]
        centers = np.mean(np.abs(conv_) * np.arange(conv_.shape[1]), axis=1)
        reordering[:l//2] = reordering[:l//2][np.argsort(centers)]
    return reordering

def mean_sort(weights: list, ind: int = -1):
    mu, var = position_mean_var(weights[ind])
    return np.argsort(mu)
        
def var_sort(weights: list, ind: int = -1):
    mu, var = position_mean_var(weights[ind])
    return np.argsort(var)




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

def plot_rf_evolution(weights, num_rows=2, num_cols=4, figsize=(15, 5), cmap='gray'):
    # num_rows * num_cols is the number of receptive fields to plot
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
    cmap_ = plt.get_cmap(cmap)
    color = cmap_(np.linspace(0.2, 0.8, weights.shape[0]))
    for i, ax in enumerate(axs.flatten()):
        for t in range(weights.shape[0]):
            ax.plot(weights[t,i,:], color=color[t], alpha=0.5)
    return fig, axs