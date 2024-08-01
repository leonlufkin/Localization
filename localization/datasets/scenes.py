"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax import Array

import jax
import jax.numpy as jnp
from functools import partial

# from nets.datasets.base import Dataset
# export PYTHONPATH="${PYTHONPATH}:./"
from localization.datasets.base import Dataset, ExemplarType
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
    subsample: int = 1,
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
    # self.scenes = scipy.io.loadmat('/Users/leonlufkin/Documents/GitHub/Localization/localization/datasets/IMAGES.mat')['IMAGES']
    # self.scenes = jnp.rollaxis(self.scenes, -1)
    self.scenes = jnp.load('/Users/leonlufkin/Documents/GitHub/Localization/localization/datasets/BIPED.npy')
    
    # Sampling functions
    self.randint = jax.vmap(jax.random.randint, in_axes=(0, None, None, None))
    
    # Sampling function for data from Olshausen & Field (1997)
    def get_of_patch(key, dataset, patch_size, subsample=1):
      patch_pre_size = patch_size * subsample
      img_key, pos_key = jax.random.split(key, 2)
      img_idx = jax.random.randint(img_key, (), 0, 10)
      x0, y0 = jax.random.randint(pos_key, (2,), 0, 512 - patch_pre_size)
      pre_patch = jax.lax.dynamic_slice(dataset[img_idx], (x0, y0), (patch_pre_size, patch_pre_size))
      # average and pool
      window_dim = window_stride = (subsample, subsample)
      patch = jax.lax.reduce_window(pre_patch, 0.0, jax.lax.add, window_dim, window_stride, 'VALID') # no padding
      patch = patch / (subsample ** 2)
      return jnp.ravel(patch)
    
    # Sampling function for data from BIPED
    def get_biped_patch(key, dataset, patch_size, subsample=1):
      patch_pre_size = patch_size * subsample
      img_key, pos_key = jax.random.split(key, 2)
      img_idx = jax.random.randint(img_key, (), 0, 250)
      x0 = jax.random.randint(pos_key, (), 0, 720 - patch_pre_size)
      y0 = jax.random.randint(pos_key, (), 0, 1280 - patch_pre_size)
      pre_patch = jax.lax.dynamic_slice(dataset[img_idx], (x0, y0), (patch_pre_size, patch_pre_size))
      # average and pool
      window_dim = window_stride = (subsample, subsample)
      patch = jax.lax.reduce_window(pre_patch, 0.0, jax.lax.add, window_dim, window_stride, 'VALID') # no padding
      patch = patch / (subsample ** 2)
      return jnp.ravel(patch)
    
    self.get_patch = jax.jit(
      jax.vmap(
        partial(get_biped_patch,
                  dataset=self.scenes, patch_size=self.side_length, subsample=subsample
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
    patches = self.get_patch(keys)

    if isinstance(index, int):
      patches = patches[0]

    return patches, None


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    side_length = 16
    sumsample = 3 # 20
    print(f"side_length: {side_length}, sumsample: {sumsample}")
    dataset = ScenesDataset(key=key, side_length=side_length, subsample=sumsample)
    x, y = dataset[:1000]
    # ipdb.set_trace()
    xx = (x.T @ x) / len(x)
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    im = ax1.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    ax2.plot(x[0])
    ax3.plot(xx[(side_length**2)//2])
    ax3.axvline((side_length**2)//2, color='r', linestyle='--') 
    ax3.axvline((side_length**2)//2 - side_length, color='r', linestyle='--', linewidth=0.5)
    ax3.axvline((side_length**2)//2 + side_length, color='r', linestyle='--', linewidth=0.5)
    ax4.imshow(x[0].reshape(side_length, side_length), cmap='gray')
    fig.savefig(f'../datasets/scenes/biped/covariance_{side_length}_{sumsample}.png')
    plt.close()
    
    # Print mean, variance, kurtosis
    print( f"mean: {x[:,0].mean()}" )
    print( f"variance: {jnp.var(x[:,0])}" )
    kurt = jnp.mean((x[:,0] - x[:,0].mean())**4) / jnp.mean((x[:,0] - x[:,0].mean())**2)**2
    print( f"excess kurtosis: {kurt - 3.}")
    
    