"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

from localization.datasets.base import Dataset, ExemplarType
from jax.scipy.special import erf as gain_function

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))

class IsingDataset(Dataset):
  """1D Ising Dataset."""

  def __init__(
    self,
    key: Array,
    xi1: float = 0.7,
    xi2: float = 0.3,
    num_steps: int = 1000,
    class_proportion: float = 0.5,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    support: tuple[float, float] = (-1.0, 1.0),
    **kwargs
  ):
    """Initializes a `NonlinearGPDataset` instance."""
    super().__init__(
      key=key,  # TODO(eringrant): Use a separate key.
      num_exemplars=num_exemplars,
      )
    
    self.num_dimensions = num_dimensions

    # new way, equivalent in distribution but lets us make more direct comparisons to Gaussian clone
    # randomness will be different, so it may yield slightly different results than before
    def calc_energy(x, J):
      xp1 = jnp.roll(x, 1, axis=0)
      return -J * (x * xp1).sum()
    
    def generate_ising(key, L, xi, num_steps):
      """
      L: length of the Ising chain
      xi: interaction strength, equal to beta because we have no external field
      """
      w = 2 * jax.random.randint(key, (L,), 0, 2) - 1
      
      position_key, accept_key = jax.random.split(key)
      positions = jax.random.randint(position_key, (num_steps,), 0, L)
      accepts = jax.random.uniform(accept_key, (num_steps,))
      
      energy = calc_energy(w, xi)
      for _, (pos, acc) in enumerate(zip(positions, accepts)):
        w_ = w.at[pos].set(-w[pos])
        energy_ = calc_energy(w_, xi)
        
        p = jnp.min(jnp.array([jnp.exp(-(energy_-energy)), 1]))
        w = jnp.where(acc < p, w_, w)
        energy = jnp.where(acc < p, energy_, energy)
        
      return w
    
    self.generate_xi1 = jax.vmap(
      partial(generate_ising,
                xi=xi1, L=num_dimensions, num_steps=num_steps,
              )
    )
      
    self.generate_xi2 = jax.vmap(
      partial(generate_ising,
                xi=xi2, L=num_dimensions, num_steps=num_steps,
              )
    )
    
    # Adjust support
    self.adjust_support = jax.jit(
      jax.vmap(
        partial(lambda x, support: (x + 1) * (support[1] - support[0]) / 2 + support[0],
                support=support)
        )
    )
    self.adjust_support = lambda x: x

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  # @profile
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

    if isinstance(index, slice):
      # TODO(leonl): Deal with the case where `index.stop` is `None`.
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

    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, index)
    if isinstance(index, int):
      keys = jnp.expand_dims(keys, axis=0)
        
    # generate xi1 and xi2
    xi1 = self.generate_xi1(key=keys)
    xi2 = self.generate_xi2(key=keys)
    # concatenate xi1 and xi2
    exemplars = jnp.concatenate((xi1, xi2), axis=0)
    labels = jnp.concatenate((jnp.ones(n), jnp.zeros(n)), axis=0)
    # subsample
    perm = jax.random.permutation(keys[0], exemplars.shape[0])
    exemplars = exemplars[perm[:n]]
    labels = labels[perm[:n]]
    # adjust support
    exemplars = self.adjust_support(exemplars)

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
  
    key = jax.random.PRNGKey(0)
    xi1, xi2, num_steps = 2, 0.1, 1000
    print("xi1, xi2, num_steps:", xi1, xi2, num_steps)
    dataset = IsingDataset(key=key, xi1=xi1, xi2=xi2, num_steps=num_steps, num_dimensions=40, num_exemplars=1000)
    
    import matplotlib.pyplot as plt
    x, y = dataset[:10000]
    x1 = x[y==1]
    x0 = x[y==0]
    
    xx = (x.T @ x) / len(x)
    fig, ax = plt.subplots()
    im = ax.imshow(xx)
    cbar = ax.figure.colorbar(im, ax=ax)
    fig.savefig(f'../../thoughts/distributions/figs/ising_{xi1}_{xi2}_{num_steps}.png')
    
    xx1 = (x1.T @ x1) / len(x1)
    fig, ax = plt.subplots()
    im = ax.imshow(xx1)
    cbar = ax.figure.colorbar(im, ax=ax)
    fig.savefig(f'../../thoughts/distributions/figs/ising_{xi1}_{num_steps}.png')

    xx0 = (x0.T @ x0) / len(x0)
    fig, ax = plt.subplots()
    im = ax.imshow(xx0)
    cbar = ax.figure.colorbar(im, ax=ax)
    fig.savefig(f'../../thoughts/distributions/figs/ising_{xi2}_{num_steps}.png')

    

