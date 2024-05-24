import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from functools import partial
from localization.datasets import NonlinearGPDataset, IsingDataset
from localization.utils import build_ising_covariance, build_pre_gaussian_covariance, build_DRT
from jax.scipy.special import erf as gain_function

def Z(g):
    return jnp.sqrt( (2/jnp.pi) * jnp.arcsin( (2*g**2) / (1 + (2*g**2)) ) )

def generate_non_gaussian(key, C, L, g):
    DRT = build_DRT(L)
    evals = jnp.diag(DRT.T @ C @ DRT)
    sqrt_C = DRT @ jnp.diag(jnp.sqrt(jnp.maximum(evals, 0))) @ DRT.T
    
    z_id = jax.random.normal(key, (L,))
    z = sqrt_C @ z_id
    x = gain_function(g * z) / Z(g)
    return x

if __name__ == '__main__':
    
    key = jax.random.PRNGKey(0)
    xi = 0.7
    num_dimensions = 20
    num_exemplars = 10000000
    
    # constructing Ising samples
    # ising_dataset = IsingDataset(key=key, xi1=xi, xi2=xi, num_steps=10000, num_dimensions=num_dimensions, num_exemplars=num_exemplars)
    # x_ising = ising_dataset[:num_exemplars][0]
    # np.save(f'./thoughts/distributions/data/ising_{num_dimensions}_{xi}.npy', x_ising)
    x_ising = np.load(f'./thoughts/distributions/data/ising_{num_dimensions}_{xi}.npy')
    
    # constructing Nonlinear GP samples
    gain = 100
    C = build_pre_gaussian_covariance(build_ising_covariance(num_dimensions, xi), gain)
    generate_nlgp = jax.jit(
      jax.vmap(
        partial(generate_non_gaussian,
                C=C, L=num_dimensions, g=gain,
                ),
      )
    )
    keys = jax.random.split(key, num_exemplars)
    x_nlgp = generate_nlgp(keys)
    np.save(f'./thoughts/distributions/data/nlgp_ising_clone_{num_dimensions}_{xi}.npy', x_nlgp)
    # x_nlgp = np.load(f'./thoughts/distributions/data/nlgp_ising_clone_{num_dimensions}_{xi}.npy')
    
    import matplotlib.pyplot as plt
    # import seaborn as sns
    
    # comparing the two distributions
    # first, check covariances
    xx_ising = (x_ising.T @ x_ising) / len(x_ising)
    xx_nlgp = (x_nlgp.T @ x_nlgp) / len(x_nlgp)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im = ax1.imshow(xx_ising)
    cbar = ax1.figure.colorbar(im, ax=ax1)
    ax1.set_title('Ising')
    im = ax2.imshow(xx_nlgp)
    cbar = ax2.figure.colorbar(im, ax=ax2)
    ax2.set_title('Nonlinear GP')
    fig.savefig(f'./thoughts/distributions/figs/experimental/ising_nlgp_{num_dimensions}_{xi}.png')
    
    # compute distribution of energies
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
    E_vmap = jax.jit(
        jax.vmap(
            partial(E, J_internal=xi, field_external=jnp.zeros(num_dimensions), J_external=0),
        )
    )
        
    E_ising = E_vmap(x_ising)
    x_nlgp = np.sign(x_nlgp)
    E_nlgp = E_vmap(x_nlgp)
    fig, ax = plt.subplots()
    E_min = min(E_ising.min(), E_nlgp.min())
    E_max = max(E_ising.max(), E_nlgp.max())
    bins = np.linspace(E_min, E_max, 20)
    ax.hist(E_ising, bins=bins, density=True, alpha=0.5, label='Ising')
    ax.hist(E_nlgp, bins=bins, density=True, alpha=0.5, label='Nonlinear GP')
    ax.legend()
    fig.savefig(f'./thoughts/distributions/figs/experimental/ising_nlgp_energy_{num_dimensions}_{xi}.png')
        
    
    
    # compute a test statistic
    
    
    
    