import numpy as np
import jax.numpy as jnp

def build_gaussian_covariance(n, xi):
    C = np.abs(jnp.tile(jnp.arange(n)[:, jnp.newaxis], (1, n)) - jnp.tile(jnp.arange(n), (n, 1)))
    C = jnp.minimum(C, n - C)
    C = np.exp(-C ** 2 / (xi ** 2))
    return C

def gabor_real(c, b, a, x0, k0, x):
    n = len(x)
    d = np.minimum(x-x0, n - (x-x0))
    return c * np.cos(k0 * d) * np.exp(-d ** 2 / a ** 2) + b

def gabor_real(c, b, a, x0, k0, x, n):
    """
    Parameters
    ----------
    x : jnp.ndarray
        The input array.
    n : int
        The length of the input array.
    c : float
        The amplitude.
    b : float
        The bias.
    a : float (positive)
        The width.
    x0 : float
        The center.
    k0 : float
        The frequency.
    """
    d = jnp.minimum(x-x0, n - (x-x0))
    return c * jnp.cos(k0 * d) * jnp.exp(-d ** 2 / a ** 2) + b

def gabor_imag(c, b, a, x0, k0, x, n):
    """
    Parameters
    ----------
    x : jnp.ndarray
        The input array.
    n : int
        The length of the input array.
    c : float
        The amplitude.
    b : float
        The bias.
    a : float (positive)
        The width.
    x0 : float
        The center.
    k0 : float
        The frequency.
    """
    d = jnp.minimum(x-x0, n - (x-x0))
    return -c * jnp.sin(k0 * d) * jnp.exp(-d ** 2 / a ** 2) + b


