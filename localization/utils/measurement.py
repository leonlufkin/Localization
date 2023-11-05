# import os
import numpy as np
# import matplotlib.pyplot as plt
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