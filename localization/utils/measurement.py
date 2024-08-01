# import os
import jax.numpy as jnp
# import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy


######################
# LOCALIZATION METRICS

def ipr(weights):
    if weights.ndim == 1:
        jnp.sum(jnp.power(weights, 4)) / jnp.sum(jnp.square(weights)) ** 2
    return jnp.sum(jnp.power(weights, 4), axis=1) / jnp.sum(jnp.square(weights), axis=1) ** 2

def entropy(weights, low=-10, upp=10, delta=0.1, base=2):
    entropies = jnp.zeros(weights.shape[0])
    for neuron, weight in enumerate(weights):
        xs = jnp.arange(low, upp, delta)
        count = jnp.zeros(len(xs)+1)
        count[0] = jnp.sum(weight < xs[0])
        for i in range(len(xs)-1):
            count[i] = jnp.sum(weight < xs[i+1]) - jnp.sum(weight < xs[i])
        count[-1] = jnp.sum(weight >= xs[-1])
        prob = count / jnp.sum(count)
        entropies[neuron] = scipy_entropy(prob, base=base)
    return entropies

def position_mean_var(weight):
    # use weights to construct probability distribution
    magnitude = jnp.square(weight)
    magnitude /= jnp.sum(magnitude, axis=-1).reshape(-1,1)

    # compute first and second order stats
    L = weight.shape[-1]
    x = jnp.arange(L)
    first = jnp.sum(x * magnitude, axis=-1)
    second = jnp.sum(x * x * magnitude, axis=-1)
    var = (second - first ** 2) 
    return first, var




###################
# SORTING FUNCTIONS

def entropy_sort(weights: list, ind: int = -1, center_sort=False):
    weight_ = weights[ind]
    l = weight_.shape[0]
    entropy_ = entropy(weight_)
    reordering = jnp.argsort(entropy_)
    if center_sort:
        conv_ = weight_[reordering[:l//2]]
        centers = jnp.mean(jnp.abs(conv_) * jnp.arange(conv_.shape[1]), axis=1)
        reordering[:l//2] = reordering[:l//2][jnp.argsort(centers)]
    return reordering

def mean_sort(weights: list, ind: int = -1):
    mu, var = position_mean_var(weights[ind])
    return jnp.argsort(mu)
        
def var_sort(weights: list, ind: int = -1):
    mu, var = position_mean_var(weights[ind])
    return jnp.argsort(var)