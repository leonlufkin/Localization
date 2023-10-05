from functools import partial
import time
import jax
import jax.numpy as jnp
from jax.scipy.special import erf as gain_function


def Z(g):
    return jnp.sqrt((2 / jnp.pi) * jnp.arcsin((g**2) / (1 + (g**2))))


@partial(jax.jit, static_argnums=(2, 3))
def generate_gaussian(key, xi, L, g):
    C = jnp.abs(
        jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L))
        - jnp.tile(jnp.arange(L), (L, 1))
    )
    C = jnp.exp(-(C**2) / (xi**2))
    z = jax.random.multivariate_normal(key, jnp.zeros(L), C, method="svd") # FIXME: using svd for numerical stability, breaks if xi > 2.5 ish

    # The cholesky method is much faster:
    # UNCOMMENT ME:
    #z = jax.random.multivariate_normal(key, jnp.zeros(L), C, method="cholesky")

    x = gain_function(g * z) / Z(g)
    return x


# Generate xi anew!
@partial(jax.jit, static_argnums=(2, 3, 4, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def inverse_branching(key, class_proportion, L, g, xi1, xi2):
    label_key, exemplar_key = jax.random.split(key, 2)

    alpha = 2.0
    beta = 1.0
    xi = jax.random.gamma(label_key, a=alpha) / beta
    label = jnp.float32(xi > (alpha / beta))

    exemplar = generate_gaussian(exemplar_key, xi, L, g)
    label = 2 * label - 1

    return exemplar, label


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def choice_branching(key, class_proportion, L, g, xi1, xi2):
    label_key, exemplar_key = jax.random.split(key, 2)

    xi = jax.random.choice(
        label_key,
        a=jnp.array((xi1, xi2)),
        p=jnp.array((class_proportion, 1 - class_proportion)),
    )
    label = jnp.where(jnp.abs(xi - xi1) < jnp.abs(xi - xi2), 0, 1)

    exemplar = generate_gaussian(exemplar_key, xi, L, g)
    label = 2 * jnp.float32(label) - 1

    return exemplar, label


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def interpolation_branching(key, class_proportion, L, g, xi1, xi2):
    label_key, exemplar_key = jax.random.split(key, 2)
    label = jax.random.bernoulli(label_key, p=class_proportion)
    label = jnp.float32(label)
    xi = label * xi1 + (1 - label) * xi2

    exemplar = generate_gaussian(exemplar_key, xi, L, g)
    label = 2 * label - 1

    return exemplar, label


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def cond_branching(key, class_proportion, L, g, xi1, xi2):
    label_key, exemplar_key = jax.random.split(key, 2)
    label = jax.random.bernoulli(label_key, p=class_proportion)
    xi = jax.lax.cond(label, lambda _: xi1, lambda _: xi2, operand=None)

    exemplar = generate_gaussian(exemplar_key, xi, L, g)
    label = 2 * jnp.float32(label) - 1

    return exemplar, label


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def bernouilli_branching(key, class_proportion, L, g, xi1, xi2):
    label_key, exemplar_key = jax.random.split(key, 2)
    label = jax.random.bernoulli(label_key, p=class_proportion)
    xi = jnp.where(label, xi1, xi2)

    exemplar = generate_gaussian(exemplar_key, xi, L, g)
    label = 2 * jnp.float32(label) - 1

    return exemplar, label


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def constant_branching(key, class_proportion, L, g, xi1, xi2):
    label_key, exemplar_key = jax.random.split(key, 2)
    label = 1.0
    xi = jnp.where(label, xi1, xi2)

    exemplar = generate_gaussian(exemplar_key, xi, L, g)
    label = 2 * jnp.float32(label) - 1

    return exemplar, label


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def non_branching(key, class_proportion, L, g, xi1, xi2):
    label_key, exemplar_key = jax.random.split(key, 2)
    label = 1.0
    xi = xi1

    exemplar = generate_gaussian(exemplar_key, xi, L, g)
    label = 2 * jnp.float32(label) - 1

    return exemplar, label


def benchmark_function_vmap(func, n_samples):
    keys = jax.random.split(jax.random.PRNGKey(0), num=n_samples)
    class_proportion = 0.5
    L = 10
    g = 2.0
    xi1 = 8.0
    xi2 = 2.0

    # Warmup
    results = func(keys, class_proportion, L, g, xi1, xi2)

    start_time = time.time()
    results = func(keys, class_proportion, L, g, xi1, xi2)
    end_time = time.time()

    print(
        f"{str(func)}:\t\t{end_time - start_time:.5f} seconds \tfor {n_samples} samples."
    )
    return results


# Running the benchmark
n_samples = int(1e5)
benchmark_function_vmap(inverse_branching, n_samples)
benchmark_function_vmap(choice_branching, n_samples)
benchmark_function_vmap(cond_branching, n_samples)
benchmark_function_vmap(bernouilli_branching, n_samples)
benchmark_function_vmap(interpolation_branching, n_samples)
benchmark_function_vmap(constant_branching, n_samples)
benchmark_function_vmap(non_branching, n_samples)
