"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from localization.datasets.base import Dataset, ExemplarType
from localization.utils import build_DRT, build_gaussian_covariance, iterate_kron
from jax.scipy.special import erf as gain_function

from collections.abc import Callable

# python debugger
import ipdb

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))


class EllipticalDataset(Dataset):
  """General elliptical dataset."""

  def __init__(
    self,
    key: Array,
    xi: tuple[float] = (0.1, 1.1,),
    inverse_cdf: Callable = lambda x: -0.5 * jnp.log(2/(x+1) - 1) + 2,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    adjust: tuple[float, float] | None = (-1.0, 1.0),
    scale: float = 1.,
    dim: int = 1,
    **kwargs
  ):
    """Initializes a `NonlinearGPDataset` instance."""
    super().__init__(
      key=key,  # TODO: Use a separate key.
      num_exemplars=num_exemplars,
      num_dimensions=num_dimensions,
      )
    
    self.inverse_cdf = inverse_cdf
    self.scale = scale
    # self.Sigmas = jnp.stack([iterate_kron(build_gaussian_covariance(num_dimensions, xi_), dim) for xi_ in xi], axis=0)
    # self.Lambdas = jnp.stack([ create_outer_matrix(Sigma) for Sigma in self.Sigmas ], axis=0)
    # self.ranks
    self.DRT = build_DRT(num_dimensions, dim)
      
    def create_outer_matrix(Sigma, thresh=1e-6):
      evals = jnp.diag(self.DRT.T @ Sigma @ self.DRT)
      pos_ind = jnp.where(evals > thresh)[0]
      Lambda = self.DRT[:, pos_ind] @ jnp.diag(jnp.sqrt(evals[pos_ind]))
      return Lambda
      
    def generate_elliptical(key, xi, L, d, thresh=1e-6):
      # make scale matrix
      Sigma = iterate_kron(build_gaussian_covariance(L, xi), d)
      # find "square root" of scale matrix (need not be full rank)
      evals = jnp.diag(self.DRT.T @ Sigma @ self.DRT)
      ind = jnp.where(evals > thresh, 1., 0.)
      # Lambda = self.DRT @ jnp.diag(jnp.sqrt(evals * ind))
      Lambda = self.DRT @ jnp.diag(jnp.sqrt(evals))
      # split keys
      z_key, r_key = jax.random.split(key)
      # sample gaussian
      z = jax.random.normal(z_key, (L,))
      # z = z * ind
      # make spherical
      z = z / jnp.linalg.norm(z)
      # sample scalar via inverse CDF
      u = jax.random.uniform(r_key)
      r = self.inverse_cdf(u)
      # make elliptical
      x = r * Lambda @ z
      return x

    self.num_classes = len(xi)
    self.generate_xi = [None for _ in range(self.num_classes)]
    for i, xi in enumerate(xi):
      self.generate_xi[i] = jax.jit(
        jax.vmap(
          partial(generate_elliptical,
                  xi=xi, L=num_dimensions, d=dim,
                  ),
        )
      )
      
    # Computing variance adjustment factors
    self.std_scale = [None for _ in range(self.num_classes)]
    keys = jax.random.split(jax.random.PRNGKey(0), 100000)
    for i in range(self.num_classes):
      x = self.generate_xi[i](keys)
      self.std_scale[i] = 1 / jnp.std(x.flatten())
      
    # Adjust support
    # no adjustment
    if adjust is None:
      self.adjust = lambda x: x
    # adjust to fall within interval
    elif isinstance(adjust, tuple):
      self.adjust = jax.jit(
        jax.vmap(
          partial(lambda x, adjust: (x + 1) * (adjust[1] - adjust[0]) / 2 + adjust[0],
                  adjust=adjust)
          )
      )
    # apply function (should only take in x and return x)
    else:
      self.adjust = jax.jit(jax.vmap(adjust))

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  # @profile
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

    # index, n = self.process_index(index)
    
    if isinstance(index, slice):
      if index.stop is None:
        raise ValueError("Slice `index.stop` must be specified.")
      index = slice_to_array(index, len(self))
      n = index.shape[0]
    elif isinstance(index, int):
      # index = jnp.array([index])
      n = 1
    else:
      index = jnp.array(index)
      n = index.shape[0]
    
    class_keys = jax.random.split(self.key, self.num_classes)
    fold_in = jax.vmap(jax.random.fold_in, in_axes=(None, 0))
    keys = [ fold_in(class_key, index) for class_key in class_keys ]
    if isinstance(index, int):
      keys = [ jnp.expand_dims(keys_, axis=0) for keys_ in keys ]
        
    # generate exemplars and labels
    exemplars = jnp.concatenate([ self.generate_xi[i](key=keys_) * self.std_scale[i] for i, keys_ in enumerate(keys) ], axis=0)
    labels = jnp.concatenate([ i * jnp.ones(n) for i in range(self.num_classes) ], axis=0)
    # subsample
    perm = jax.random.permutation(class_keys[0], exemplars.shape[0])
    exemplars = exemplars[perm[:n]]
    labels = labels[perm[:n]]
    # adjust support
    exemplars = self.scale * self.adjust(exemplars)

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
    config = dict(
      # data config
      num_dimensions=200,#40,
      # xi=(0.1, 0.1),
      # xi=(3, 3),
      xi=(15, 15),
      adjust=(-1, 1),
      gain=0.5, #100,
      key=jax.random.PRNGKey(0),
      scale=2,
    )
    
    dataset = EllipticalDataset(**config)
    x, y = dataset[:100000]
    xx = (x.T @ x) / len(x)
    
    xfilt = x[(0.95<x[:,config['num_dimensions']//2]) & (x[:,config['num_dimensions']//2]<1.05)]
    xxfilt = jnp.cov(xfilt.T)
    
    def create_outer_matrix(Sigma, thresh=1e-6):
      DRT = build_DRT(Sigma.shape[0], 1)
      evals = jnp.diag(DRT.T @ Sigma @ DRT)
      pos_ind = jnp.where(evals > thresh)[0]
      Lambda = DRT[:, pos_ind] @ jnp.diag(jnp.sqrt(evals[pos_ind]))
      return Lambda
    
    def create_outer_matrix_inv(Sigma, thresh=1e-6):
      DRT = build_DRT(Sigma.shape[0], 1)
      evals = jnp.diag(DRT.T @ Sigma @ DRT)
      pos_ind = jnp.where(evals > thresh)[0]
      Lambda_inv = DRT[:, pos_ind] @ jnp.diag(1/jnp.sqrt(evals[pos_ind]))
      return Lambda_inv
    
    Sigma = iterate_kron(build_gaussian_covariance(config['num_dimensions'], config['xi'][0]), 1)
    Lambda = create_outer_matrix(Sigma)
    print(Lambda.shape)
    Lambda_inv = create_outer_matrix_inv(Sigma)
    print(Lambda.T @ Lambda_inv) # good !
    print(jnp.linalg.norm(Lambda @ Lambda.T - Sigma)) # good !
    Sigma_inv = Lambda_inv @ Lambda_inv.T
    g_inner = jnp.sum(x * (x @ Sigma_inv.T), axis=1)
    print(g_inner)
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))
    im = ax1.imshow(xx, cmap='gray'); ax1.set_title("empirical covariance")
    cbar = plt.colorbar(im)
    ax2.plot(x[0]); ax2.set_title("x[0]")
    bins = jnp.linspace(0, g_inner.max(), 40)
    ax3.hist(g_inner, bins=bins); ax3.axvline(x=4 * 2 ** 2, color='r', linestyle='--'); ax3.set_title("g_inner")
    ax4.hist(x[:, 0], bins=40); ax4.set_title("first coordinate of x")
    ax5.imshow(xxfilt, cmap='gray'); ax5.set_title("empirical covariance x conditioned on x[n/2] in [0.95, 1.05]")
    fig.savefig(f"../datasets/elliptical/covariance_{config['xi'][0]}.png")
    plt.close()
  
      
    