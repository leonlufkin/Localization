"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""


from collections.abc import Callable
import jax
from jax import Array
import jax.numpy as jnp
from functools import partial

# export PYTHONPATH="${PYTHONPATH}:./"
from localization.datasets.base import Dataset, ExemplarType
from localization.utils import build_DRT, build_gaussian_covariance, iterate_kron

from jax.scipy.special import erf as gain_function

def slice_to_array(s: slice, array_length: int):
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))

class SymmBreakDataset(Dataset):
  """Given a vector-valued dataset, adjust the marginals."""

  def __init__(
    self,
    key: Array,
    base_dataset: type[Dataset], # dataset_cls to initially draw samples from
    break_fn: Callable = lambda x: False,
    toss_label: int = 0,
    batch_size: int = 100,
    **dataset_kwargs
  ):
    """Initializes an `AdjustMarginalDataset` instance."""
    
    self.dataset = base_dataset(key=key, **dataset_kwargs)
    self.break_fn = jax.vmap(break_fn)
    self.toss_label = toss_label
    self.batch_size = batch_size
    
    super().__init__(
      key=key,  # TODO: Use a separate key.
      num_exemplars=self.dataset.num_exemplars,
      )

    self.num_dimensions = self.dataset.num_dimensions 
    
  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  # @profile
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""

    expanded_index = jnp.zeros(100 * self.batch_size, dtype=jnp.int32)
    index = slice_to_array(index, self.num_exemplars)
    exemplars, labels = self.dataset[index]
    toss_mask = self.break_fn(exemplars)
    # print(toss_mask.sum())
    
    # ind = jnp.where(toss_mask)[0]
    # exemplars = exemplars.at[ind].set(0.) # assumes fixed bias
    # labels = labels.at[ind].set(self.toss_label)
    
    ind = jnp.where(~toss_mask)[0][:self.batch_size]
    exemplars = exemplars[ind]
    labels = labels[ind]
    # pad to batch size
    if len(exemplars) < self.batch_size:
      exemplars = jnp.pad(exemplars, ((0, self.batch_size - len(exemplars)), (0, 0)))
      labels = jnp.pad(labels, (0, self.batch_size - len(labels)))
    
    # if len(labels) > 1:
    #   exemplars = exemplars[~toss_mask]
    #   labels = labels[~toss_mask]
    return exemplars, labels


if __name__ =="__main__":
    config = dict(
      # data config
      num_dimensions=40,#40,
      # xi=(0.1, 0.1),
      xi=(3, 3),
      # xi=(15, 15),
      adjust=(-1, 1),
      gain=0.5, #100,
      key=jax.random.PRNGKey(0),
      scale=2,
    )
    
    from localization.datasets import NonlinearGPDataset, EllipticalDataset
    # from localization.utils import normal_adjust, uniform_adjust, no_adjust
    
    def break_fn(x):
      return jnp.any(jnp.abs(x) < 0.1)
    
    elliptical_break_dataset = SymmBreakDataset(base_dataset=EllipticalDataset, break_fn=break_fn, **config)
    x, y = elliptical_break_dataset[:1000000]
    print(x.shape)
    xfilt = x[~jnp.all(x==0, axis=1)]
    xxfilt = jnp.cov(xfilt.T)
    print(xfilt.shape)
    
    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # im = ax1.imshow(xxfilt, cmap='gray')
    # cbar = plt.colorbar(im)
    # ax2.hist(xfilt[:,0], bins=40, density=True)
    # fig.savefig(f'../datasets/elliptical_break/covariance_{elliptical_break_dataset.break_fn.__name__}.png')
    # plt.close()
    