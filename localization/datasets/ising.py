"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

from localization.datasets.base import Dataset, ExemplarType
from jax.scipy.special import erf as gain_function

def E(
    state_internal: Array,
    J_internal: float,
    field_external: Array,
    J_external: float,
) -> Array:
    return -1 * (
        J_internal * ((state_internal * jnp.roll(state_internal, 1)).sum())
        + J_external * (field_external @ state_internal.T)
    )


def pi(
    state_internal: Array,
    J_internal: float,
    field_external: Array,
    J_external: float,
) -> Array:
    return jnp.exp(-E(state_internal, J_internal, field_external, J_external))


def propose(
    key: Array,
    state: Array,
    J_internal: float,
    field_external: Array,
    J_external: float,
) -> Array:
    flip = jax.random.randint(key, shape=(), minval=0, maxval=state.shape[0])
    updated_state = state.at[flip].set(-1 * state[flip])

    pi_prime = pi(updated_state, J_internal, field_external, J_external)
    pi_ref = pi(state, J_internal, field_external, J_external)

    return jnp.where(1.0 < pi_prime / pi_ref, 1.0, pi_prime / pi_ref), updated_state


def step(
    key: Array,
    J: float,
    state: Array,
):
    propose_key, accept_key = jax.random.split(key)
    accept_prob, next_state = propose(
        propose_key, state, J, state, 0.0
    )  # set external field to null
    return jnp.where(jax.random.uniform(accept_key) < accept_prob, next_state, state)


def initialize_state(key: Array, d: int) -> Array:
    return jax.random.bernoulli(key, p=0.5, shape=(d,)) * 2 - 1


def chain(key: Array, num_steps: int, initial_state: Array, J: float):
    def body(n, args):
        del n
        state_i, key_i = args
        return step(key_i, J, state_i), jax.random.split(key_i)[0]

    final_state, _ = jax.lax.fori_loop(
        0,
        num_steps,
        body,
        (
            initial_state,
            jax.random.split(key)[0],
        ),
    )
    return final_state


def simulate(key: Array, D: int, J: float, num_steps: int = 10000):
    initialize_key, chain_key = jax.random.split(key)
    initial_state = initialize_state(initialize_key, D)
    return chain(chain_key, num_steps, initial_state, J)


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
    # xi1: float = 0.7,
    # xi2: float = 0.3,
    xi: tuple[float] = (0.7, 0.3),
    num_steps: int = 1000,
    class_proportion: float = 0.5,
    num_dimensions: int = 100,
    num_exemplars: int = 1000,
    support: tuple[float, float] = (-1.0, 1.0),
    **kwargs
  ):
    """Initializes a `NonlinearGPDataset` instance."""
    super().__init__(
      key=key,  # TODO: Use a separate key.
      num_exemplars=num_exemplars,
      )
    
    self.num_dimensions = num_dimensions
    
    self.num_classes = len(xi)
    self.generate_xi = [None for _ in range(self.num_classes)]
    for i, xi_ in enumerate(xi):
      self.generate_xi[i] = jax.jit(
        jax.vmap(
          partial(simulate,
                  J=xi_, D=num_dimensions, num_steps=num_steps,
                  ),
        )
      )
    
    # self.generate_xi1 = jax.jit(
    #   jax.vmap(
    #     partial(simulate,
    #               J=xi1, D=num_dimensions, num_steps=num_steps,
    #             )
    #   )
    # )
      
    # self.generate_xi2 = jax.jit(
    #   jax.vmap(
    #     partial(simulate,
    #               J=xi2, D=num_dimensions, num_steps=num_steps,
    #             )
    #   )
    # )
    
    # Adjust support
    self.adjust = jax.jit(
      jax.vmap(
        partial(lambda x, support: (x + 1) * (support[1] - support[0]) / 2 + support[0],
                support=support)
        )
    )
    self.adjust = lambda x: x

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  # @profile
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

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

    # keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, index)
    # if isinstance(index, int):
    #   keys = jnp.expand_dims(keys, axis=0)
        
    # # generate xi1 and xi2
    # xi1 = self.generate_xi1(key=keys)
    # xi2 = self.generate_xi2(key=keys)
    # # concatenate xi1 and xi2
    # exemplars = jnp.concatenate((xi1, xi2), axis=0)
    # labels = jnp.concatenate((jnp.ones(n), jnp.zeros(n)), axis=0)
    # # subsample
    # perm = jax.random.permutation(keys[0], exemplars.shape[0])
    # exemplars = exemplars[perm[:n]]
    # labels = labels[perm[:n]]
    # # adjust support
    # exemplars = self.adjust(exemplars)
    
    class_keys = jax.random.split(self.key, self.num_classes)
    fold_in = jax.vmap(jax.random.fold_in, in_axes=(None, 0))
    keys = [ fold_in(class_key, index) for class_key in class_keys ]
    if isinstance(index, int):
      keys = [ jnp.expand_dims(keys_, axis=0) for keys_ in keys ]
        
    # generate exemplars and labels
    exemplars = jnp.concatenate([ self.generate_xi[i](key=keys_) for i, keys_ in enumerate(keys) ], axis=0)
    labels = jnp.concatenate([ i * jnp.ones(n) for i in range(self.num_classes) ], axis=0)
    # subsample
    perm = jax.random.permutation(class_keys[0], exemplars.shape[0])
    exemplars = exemplars[perm[:n]]
    labels = labels[perm[:n]]
    # adjust support
    exemplars = self.adjust(exemplars)

    if isinstance(index, int):
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels


if __name__ =="__main__":
  
    key = jax.random.PRNGKey(0)
    xi1, xi2, num_steps = 0.7, 0.3, 10000
    print("xi1, xi2, num_steps:", xi1, xi2, num_steps)
    dataset = IsingDataset(key=key, xi1=xi1, xi2=xi2, num_steps=num_steps, num_dimensions=40, num_exemplars=1000)
    
    import matplotlib.pyplot as plt
    from localization.utils import build_ising_covariance, build_pre_gaussian_covariance
    
    # visualizing Ising
    x, y = dataset[:10000]
    x1 = x[y==1]
    x0 = x[y==0]
    
    xx = (x.T @ x) / len(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im = ax1.imshow(xx)
    cbar = ax1.figure.colorbar(im, ax=ax1)
    ax2.plot( xx[20], label='empirical' )
    fig.savefig(f'./thoughts/distributions/figs/experimental/ising_{xi1}_{xi2}_{num_steps}.png')
    
    xx1 = (x1.T @ x1) / len(x1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    im = ax1.imshow(xx1)
    cbar = ax1.figure.colorbar(im, ax=ax1)
    ax2.plot( xx1[20], label='empirical' )
    ax2.plot( build_ising_covariance(40, xi1)[20], label='theoretical' )
    ax2.legend()
    ax3.plot( x1[:3].T )
    fig.savefig(f'./thoughts/distributions/figs/experimental/ising_{xi1}_{num_steps}.png')

    xx0 = (x0.T @ x0) / len(x0)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    im = ax1.imshow(xx0)
    cbar = ax1.figure.colorbar(im, ax=ax1)
    ax2.plot( xx0[20], label='empirical' )
    ax2.plot( build_ising_covariance(40, xi2)[20], label='theoretical' )
    ax2.legend()
    ax3.plot( x0[:3].T )
    fig.savefig(f'./thoughts/distributions/figs/experimental/ising_{xi2}_{num_steps}.png')


    

