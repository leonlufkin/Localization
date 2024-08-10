"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""


from collections.abc import Callable
import jax
from jax import Array
import jax.numpy as jnp
from functools import partial

# export PYTHONPATH="${PYTHONPATH}:./"
from jaxnets.datasets.base import Dataset, ExemplarType
from localization.utils import build_DRT, build_gaussian_covariance, iterate_kron

from jax.scipy.special import erf as gain_function

class AdjustMarginalDataset(Dataset):
  """Given a vector-valued dataset, adjust the marginals."""

  def __init__(
    self,
    key: Array,
    base_dataset: type[Dataset], # dataset_cls to initially draw samples from
    marginal_adjust: Callable,
    **dataset_kwargs
  ):
    """Initializes an `AdjustMarginalDataset` instance."""
    
    self.dataset = base_dataset(key=key, **dataset_kwargs)
    self.marginal_adjust = marginal_adjust
    
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

    self.key, adjust_key = jax.random.split(self.key, 2)
    exemplars, labels = self.dataset[index]
    marginal_scales = self.marginal_adjust(adjust_key, len(exemplars))
    exemplars = exemplars * marginal_scales.reshape(-1, 1)

    return exemplars, labels


if __name__ =="__main__":
    key = jax.random.PRNGKey(0)
    xi, gain = (5, 3, 1), 100
    print("xi, gain:", xi, gain)
    
    from localization.datasets import NonlinearGPDataset
    from localization.utils import normal_adjust, uniform_adjust, no_adjust
    
    nlgp_normal_marginal = AdjustMarginalDataset(key=key, base_dataset=NonlinearGPDataset, marginal_adjust=no_adjust, xi=xi, gain=gain, num_dimensions=40, num_exemplars=10000)
    x, y = nlgp_normal_marginal[:10000]
    xx = (x.T @ x) / len(x)
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im = ax1.imshow(xx, cmap='gray')
    cbar = plt.colorbar(im)
    ax2.hist(x[:,0], bins=40, density=True)
    fig.savefig(f'../datasets/nlgp_normal_marginal/covariance_{xi}_{gain}_{nlgp_normal_marginal.marginal_adjust.__name__}.png')
    plt.close()
    
