import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit, vmap

from jax.scipy.special import erf as gain_function
import matplotlib.pyplot as plt

from scipy.stats import entropy

import argparse
import datetime
import timeit

def Z(g):
    return jnp.sqrt( (2/jnp.pi) * jnp.arcsin( (g**2) / (1 + (g**2)) ) )

def generate_gaussian(key, xi, L, num_samples=1):
    C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
    C = jnp.exp(-C ** 2 / (xi ** 2))
    z = jax.random.multivariate_normal(key, np.zeros(L), C, shape=(num_samples,))
    return z

# TODO(leonl): Vectorize this function with `jax.vmap` across `num_samples`!
def generate_non_gaussian_samples(key, xi, L, g, num_samples=1000):
    z = generate_gaussian(key, xi, L, num_samples=num_samples)
    x = gain_function(g * z) / Z(g)
    return x

# FIXME: leon's attempt! (see below)
def generate_non_gaussian(key, xi, L, g):
    C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
    C = jnp.exp(-C ** 2 / (xi ** 2))
    z = jax.random.multivariate_normal(key, np.zeros(L), C)
    x = gain_function(g * z) / Z(g)
    return x

def generate_non_gaussian_chatgpt(keys, xi, L, g):
    z = vmap(lambda key: generate_gaussian(key, xi, L))(keys)
    x = gain_function(g * z) / Z(g)
    return x

def compute_entropy(weights, low=-10, upp=10, delta=0.1, base=2):
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
            
        
def parse_args():
    # read command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--inputs", nargs="+", default=["nlgp"], help="inputs: nlgp | gp")
    parser.add_argument("--xi1", type=float, default=0.1, help="correlation length 1")
    parser.add_argument("--xi2", type=float, default=1.1, help="correlation length 2")
    parser.add_argument("--gain", type=float, default=1, help="gain of the NLGP")
    
    parser.add_argument("--L", type=int, default=400, help="linear input dimension. The input will have D**dim pixels.")
    parser.add_argument("--K", type=int, default=8, help="# of student nodes / channels of the student")
    parser.add_argument("--dim", type=int, default=1, help="input dimension: one for vector, two for images. The input will have D**dim pixels")
        
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=5000, help="number of epochs")
    parser.add_argument("--loss", type=str, default="mse", help="loss: mse | ce")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    
    parser.add_argument("--activation", default="tanh", help="activation: tanh | sigmoid | relu")
    parser.add_argument("--second_layer", default="linear", help="second layer: linear | learnable_bias | float (fixed bias value)")
        
    args = parser.parse_args()
    return vars(args)
        
def main(
    xi1, xi2, gain,
    L, K, dim,
    batch_size, num_epochs, loss='mse', lr=0.01,
    activation='tanh', second_layer='linear',
    path='.', **kwargs
):
    # original
    key = jax.random.PRNGKey(0)
    print("Generating samples the original way...")
    t1 = timeit.timeit(lambda: generate_non_gaussian_samples(key, xi1, L, gain, num_samples=batch_size), number=1000)
    
    # attempt at vectorization
    print("Splitting keys...")
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    lambda_generate_non_gaussian = vmap(lambda key: generate_non_gaussian(key, xi1, L, gain))
    jitted_lambda_generate_non_gaussian = jit(lambda_generate_non_gaussian)

    from functools import partial
    partial_generate_non_gaussian = vmap(partial(generate_non_gaussian, xi= xi1, L= L, g=gain))
    jitted_partial_generate_non_gaussian = jit(partial_generate_non_gaussian)

    # The "canonical" way to use `vmap` and `jit`:
    in_axes_generate_non_gaussian = vmap(generate_non_gaussian, in_axes=(0, None, None, None))
    jitted_in_axes_generate_non_gaussian = jit(in_axes_generate_non_gaussian, static_argnames=('xi', 'L', 'g'))

    # Compile the `jit`ed functions before timing them.
    jitted_lambda_generate_non_gaussian(keys)
    jitted_partial_generate_non_gaussian(keys)
    jitted_in_axes_generate_non_gaussian(keys, xi1, L, gain)

    print("Generating samples the vectorized way...")
    t2 = timeit.timeit(lambda: lambda_generate_non_gaussian(keys), number=1000)
    t3 = timeit.timeit(lambda: partial_generate_non_gaussian(keys), number=1000)
    t4 = timeit.timeit(lambda: in_axes_generate_non_gaussian(keys, xi1, L, gain), number=1000)

    t5 = timeit.timeit(lambda: jitted_lambda_generate_non_gaussian(keys), number=1000)
    t6 = timeit.timeit(lambda: jitted_partial_generate_non_gaussian(keys), number=1000)
    t7 = timeit.timeit(lambda: jitted_in_axes_generate_non_gaussian(keys, xi1, L, gain), number=1000)

    # chatgpt attempt
    print("Generating samples the chatgpt way...")
    t8 = timeit.timeit(lambda: generate_non_gaussian_chatgpt(keys, xi1, L, gain), number=100)
    
    print(f"Original: {t1}")
    print(f"`vmap`-ed lambda: {t2}")
    print(f"`vmap`-ed partial: {t3}")
    print(f"`vmap`-ed in-axes: {t4}")
    print(f"`jit`-ed and `vmap`-ed lambda: {t5}")
    print(f"`jit`-ed and `vmap`-ed partial: {t6}")
    print(f"`jit`-ed and `vmap`-ed in-axes: {t7}")
    print(f"ChatGPT 🤡: {t8}")
        
    
if __name__ == '__main__':
    
    # get arguments
    kwargs = parse_args()
    print("Arguments:")
    for arg, val in kwargs.items():
        print(f"{arg}: {val}")
    
    # main    
    main(**kwargs)
