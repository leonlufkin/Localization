import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import datasets
import models
import samplers
import matplotlib.pyplot as plt
from localization.experiments import make_key, simulate


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
        entropies[neuron] = entropy(prob, base=base)
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

def ipr(weights):
    return np.sum(np.power(weights, 4), axis=1) / np.sum(np.square(weights), axis=1) ** 2

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

# FIXME: why do we still have this function?
def plot_receptive_field(weights, figsize=(10, 5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    weights = weights[np.argsort(entropy(weights))]
    ax.imshow(weights, cmap='gray')#, vmin=-1, vmax=1)
    ax.set_xlabel('Input neurons')
    ax.set_ylabel('Hidden neurons')
    return fig, ax

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
