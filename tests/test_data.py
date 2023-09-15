from functools import partial
import jax
import time
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf as gain_function


from conv_emergence import generate_gaussian
from conv_emergence import generate_non_gaussian

def Z(g):
    return np.sqrt( (2/np.pi) * np.arcsin( (g**2) / (1 + (g**2)) ) )


def generate_gaussian_numpy(L, xi, dim=1, num_samples=1):
    if dim > 2:
        raise NotImplementedError("dim > 2 not implemented")
    
    C = np.abs(np.tile(np.arange(L)[:, np.newaxis], (1, L)) - np.tile(np.arange(L), (L, 1)))
    C = np.exp(-C ** 2 / (xi ** 2))
    
    if dim > 1:
        C = np.kron(C, C)
    
    z = np.random.multivariate_normal(np.zeros(L ** dim), C, size=num_samples)
    if dim > 1:
        z = z.reshape((num_samples, L, L))
        
    return z

def generate_non_gaussian_numpy(L, xi, g, dim=1, num_samples=1000):
    z = generate_gaussian_numpy(L, xi, dim=dim, num_samples=num_samples)
    x = gain_function(g * z) / Z(g)
    return x


def generate_gaussian_to_vmap(key, L, xi, dim=1, num_samples=1):
    # we are fixing dim=1 in this script
    C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
    C = jnp.exp(-C ** 2 / (xi ** 2))
    z = jax.random.multivariate_normal(key, np.zeros(L), C, shape=(num_samples,))
    return z

# TODO(leonl): Vectorize this function with `jax.vmap` across `num_samples`!
def generate_non_gaussian(key, xi, L, g, dim=1, num_samples=1000):
    z = generate_gaussian(key, xi, L, dim=dim, num_samples=num_samples)
    x = gain_function(g * z) / Z(g)
    return x

if __name__ == '__main__':

    key = jax.random.PRNGKey(0)
    L = 100
    xi = 0.1
    g = 1
    num_samples=10000

    z = generate_non_gaussian(key, xi, L, g)
    #import ipdb; ipdb.set_trace()

    compiled_datagen = partial(
      generate_non_gaussian,
      L=L, xi=xi, g=g,
      num_samples=num_samples,
    )
    compiled_datagen_jit = jax.jit(compiled_datagen)
    compiled_datagen(key)

    # Warm-up (to compile the JIT'ed function)
    _ = compiled_datagen_jit(key).block_until_ready()

    # Time the Numpy
    start_time = time.time()
    result_with_numpy = generate_non_gaussian_numpy(
       L, xi, g, dim=1, num_samples=num_samples,
        ).block_until_ready()  # block_until_ready ensures the computation is finished
    end_time = time.time()
    print(f"Time with Numpy: {end_time - start_time} seconds")

    # Time the function without JIT
    start_time = time.time()
    result_no_jit = compiled_datagen(key).block_until_ready()  # block_until_ready ensures the computation is finished
    end_time = time.time()
    print(f"Time without JIT: {end_time - start_time} seconds")

    # Time the function with JIT
    start_time = time.time()
    result_with_jit = compiled_datagen_jit(key).block_until_ready()  # block_until_ready ensures the computation is finished
    end_time = time.time()
    print(f"Time with JIT: {end_time - start_time} seconds")


    N_xi = 500

    # VMAP example.
    generate_non_gaussian_to_vmap = partial(
        generate_non_gaussian, 
        L=L, 
        g=g,
        dim=1,
        num_samples=1,
        )
    generate_non_gaussian_vmapped = jax.vmap(
        generate_non_gaussian_to_vmap, 
        in_axes=(0, 0),
    )

    # TODO(leonl): Remove warm-up (compilation) from this eval.
    start_time = time.time()
    result_with_jit_vmap = generate_non_gaussian_vmapped(
        jax.random.split(key, N_xi),
        xi * jnp.ones(N_xi), 
    ).block_until_ready()  # block_until_ready ensures the computation is finished
    end_time = time.time()
    print(f"Time with VMAP no JIT: {end_time - start_time} seconds")

    start_time = time.time()
    result_with_jit_vmap = jax.jit(generate_non_gaussian_vmapped)(
        jax.random.split(key, N_xi),
        xi * jnp.ones(N_xi), 
    ).block_until_ready()  # block_until_ready ensures the computation is finished
    end_time = time.time()
    print(f"Time with VMAP then JIT: {end_time - start_time} seconds")

    start_time = time.time()
    result_with_vmap_jit = jax.vmap(jax.jit(generate_non_gaussian_to_vmap), in_axes=(0, 0))(
        jax.random.split(key, N_xi),
        xi * jnp.ones(N_xi), 
    ).block_until_ready()  # block_until_ready ensures the computation is finished
    end_time = time.time()
    print(f"Time with JIT then VMAP: {end_time - start_time} seconds")

