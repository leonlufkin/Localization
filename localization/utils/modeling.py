import numpy as np
import jax.numpy as jnp

def build_gaussian_covariance(n, xi):
    C = jnp.abs(jnp.tile(jnp.arange(n)[:, jnp.newaxis], (1, n)) - jnp.tile(jnp.arange(n), (n, 1)))
    C = jnp.minimum(C, n - C)
    C = jnp.exp(-C ** 2 / (xi ** 2))
    return C

def build_gaussian_covariance_numpy(n, xi):
    C = np.abs(np.tile(np.arange(n)[:, np.newaxis], (1, n)) - np.tile(np.arange(n), (n, 1)))
    C = np.minimum(C, n - C)
    C = np.exp(-C ** 2 / (xi ** 2))
    return C

def build_non_gaussian_covariance(n, xi, g):
    C = build_gaussian_covariance(n, xi)
    Z = lambda g: jnp.sqrt( (2/jnp.pi) * jnp.arcsin( (2*g**2) / (1 + (2*g**2)) ) )
    C = 2/jnp.pi/(Z(g)**2) * jnp.arcsin( (2*g**2) / (1 + (2*g**2)) * C )
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

def build_DRT(n):
  DFT = jnp.zeros((n, n), dtype=complex)
  w = jnp.exp(-2 * jnp.pi * 1j / n)
  for i in range(DFT.shape[0]):
    DFT = DFT.at[:,i].set(w ** (i * jnp.arange(n)) / jnp.sqrt(n))

  DCT = DFT.real
  DST = DFT.imag

  DRT_ = jnp.sqrt(2) * jnp.concatenate((DCT[:, :(n//2+1)], DST[:, 1:(n//2)]), axis=1)
  DRT_ = DRT_.at[:,0].set(DRT_[:,0] / jnp.sqrt(2))
  DRT_ = DRT_.at[:,n//2].set(DRT_[:,n//2] / jnp.sqrt(2))
  DRT = jnp.zeros((n, n))
  DRT = DRT.at[:,0].set(DRT_[:,0])
  DRT = DRT.at[:,1::2].set(DRT_[:,1:n//2+1])
  DRT = DRT.at[:,2::2].set(DRT_[:,n//2+1:])
  return DRT
