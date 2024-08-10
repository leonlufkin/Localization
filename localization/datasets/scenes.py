"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from jaxnets.datasets.base import Dataset, ExemplarType
import scipy.io
import ipdb

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))


class ScenesDataset(Dataset):
  """Natural image patches from http://www.rctn.org/bruno/sparsenet/IMAGES.mat, as used in Olshausen & Field (1997)."""

  def __init__(
    self,
    key: Array,
    side_length: int = 12,
    num_exemplars: int = 1000,
    **kwargs
  ):
    """Initializes a `ScenesDataset` instance."""
    super().__init__(
      key=key,
      num_exemplars=num_exemplars,
      num_dimensions=side_length ** 2,
    )
    
    self.side_length = side_length
    
    # Load images
    self.scenes = scipy.io.loadmat('./localization/datasets/IMAGES.mat')['IMAGES']
    self.scenes = jnp.rollaxis(self.scenes, -1)
    
    # Sampling functions
    self.randint = jax.vmap(jax.random.randint, in_axes=(0, None, None, None))
    
    def get_patch(key, dataset, patch_size):
      # ipdb.set_trace()
      img_key, pos_key = jax.random.split(key, 2)
      img_idx = jax.random.randint(img_key, (), 0, 10)
      x0, y0 = jax.random.randint(pos_key, (2,), 0, 512 - self.side_length)
      patch = jax.lax.dynamic_slice(dataset[img_idx], (x0, y0), (patch_size, patch_size))
      return jnp.ravel(patch)
    
    self.get_patch = jax.jit(
      jax.vmap(
        partial(get_patch,
                  dataset=self.scenes, patch_size=self.side_length
                ),
      ),# static_argnames=['patch_size']
    )
  

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
      
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, index)
    patches = self.get_patch(keys)

    if isinstance(index, int):
      patches = patches[0]

    return patches, None


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    side_length = 16
    print("side_length: ", side_length)
    dataset = ScenesDataset(key=key, side_length=side_length)
    x, y = dataset[:10000]
    xx = (x.T @ x) / len(x)
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    im = ax1.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    ax2.plot(x[0])
    ax3.plot(xx[(side_length**2)//2])
    ax3.axvline((side_length**2)//2, color='r', linestyle='--') 
    ax3.axvline((side_length**2)//2 - side_length, color='r', linestyle='--', linewidth=0.5)
    ax3.axvline((side_length**2)//2 + side_length, color='r', linestyle='--', linewidth=0.5)
    fig.savefig(f'../datasets/scenes/covariance_{side_length}.png')
    plt.close()
    
    # Print mean, variance, kurtosis
    print( f"mean: {x[:,0].mean()}" )
    print( f"variance: {jnp.var(x[:,0])}" )
    kurt = jnp.mean((x[:,0] - x[:,0].mean())**4) / jnp.mean((x[:,0] - x[:,0].mean())**2)**2
    print( f"excess kurtosis: {kurt - 3.}")
    
    