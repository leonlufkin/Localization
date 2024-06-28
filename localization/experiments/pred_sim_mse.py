import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from localization import datasets
from localization import models
from localization import samplers
from localization.experiments import simulate, simulate_or_load, make_key
from localization.utils import plot_receptive_fields, plot_rf_evolution, build_gaussian_covariance, build_non_gaussian_covariance, entropy_sort, build_DRT
from scipy.special import erf
from scipy.signal import fftconvolve
from tqdm import tqdm

import ipdb

# gaussian_cdf = lambda x: 0.5 * (erf(x/np.sqrt(2)) + 1)

def alginv(x):
    return x / jnp.sqrt(1 - x ** 2)

def phi(x):
    return erf(alginv(x) / jnp.sqrt(2))

def integrate(w_init, cov1, cov0, lr, num_epochs, cov1_weight=0.5, cov0_weight=0.5):
  n = len(w_init)
  
  w = np.zeros((num_epochs, n))
  w[0] = w_init
  sw1_ = np.zeros((num_epochs, n))
  sw0_ = np.zeros((num_epochs, n))
  f_ = np.zeros((num_epochs, n))
  
#   C = jnp.abs(jnp.tile(jnp.arange(n)[:, jnp.newaxis], (1, n)) - jnp.tile(jnp.arange(n), (n, 1)))
#   C = jnp.minimum(C, n - C)
  C = jnp.arange(n)
  C = jnp.minimum(C, n - C)
  sigma1 = cov1(C / n)
  sigma0 = cov0(C / n)
  
  for i in tqdm(range(1, num_epochs)):
    sw1 = conv_circ(sigma1, w[i-1])
    sw0 = conv_circ(sigma0, w[i-1])
    x = sw1 / jnp.sqrt(w[i-1] @ sw1)
    x = jnp.minimum(jnp.maximum(x, -1), 1)
    f = phi( x )
    w[i] = w[i-1] + 0.5 * lr * (f - (cov1_weight * sw1 + cov0_weight * sw0))
    sw1_[i] = sw1
    sw0_[i] = sw0
    f_[i] = f
    # ipdb.set_trace()
    
    if np.any(np.isnan(w[i])):
        print(f'nan on i={i:d}')
        break

    # if i == 100:
    #     ipdb.set_trace()
    
  return w, sw1_, sw0_, f_


def xavier(n, seed):
    return models.xavier_normal_init(jnp.zeros((n,1)), key=jax.random.PRNGKey(seed)).squeeze()

def conv_circ(s, k):
    return np.real(np.fft.ifft(np.fft.fft(s) * np.fft.fft(k)))

from argparse import ArgumentParser
parser = ArgumentParser(prog='MULTIPLE TARGETS')
parser.add_argument('cov', default='exp', type=str)
parser.add_argument('--psi', default=2., type=float)

def exp(x, xi, **kwargs):
    return np.exp(- x ** 2 / (xi ** 2))
def cos_exp(x, xi, psi, **kwargs):
    return np.exp(- x ** 2 / (xi ** 2)) * np.cos(psi * x)
def sin_exp(x, xi, psi, **kwargs):
    return np.exp(- x ** 2 / (xi ** 2)) * np.absolute(np.sin(psi * x))
def exp_gap(x, xi, **kwargs):
    return np.where(x == 0, 0.025, np.exp(- x ** 2 / (xi ** 2)))

if __name__ == '__main__':
    
    # Covariance is specified as a periodic function k : \T -> \R such < k * x, x >_T >= 0 for all x \in L(\T) 
    # It must satisfy k(0) = 1; is this an issue for sin(x)-like covariance? well, assumption that marginal variance is 1 is not critical, so we could make the smaller, or make off-diagonal covariances > 1.
    
    # Pick covariance
    
    args = parser.parse_args()
    print(vars(args))
    covs = dict(exp=exp, cos_exp=cos_exp, sin_exp=sin_exp, exp_gap=exp_gap)
    cov = covs[args.cov]
    
    # Integrate
    n = 40
    w, sw1, sw0, f = integrate(
        w_init=xavier(n, 0),
        cov1=partial(cov, xi=2/n, psi=args.psi),
        cov0=partial(cov, xi=0.5/n, psi=args.psi),
        lr=0.1,
        num_epochs=10000,
        cov1_weight=0.5,
        cov0_weight=0.5,
    )
    
    print(w.shape)
    print(sw1.shape)
    print(sw0.shape)
    print(f.shape)
    
    # Plot it
    fig, axs = plot_rf_evolution(w[::100].reshape(-1,1,n), figsize=(8, 4))
    fig2, ax = plt.subplots(1, 1, figsize=(8, 4))
    C = jnp.arange(n) - n // 2
    C = jnp.minimum(C, n - C)
    sigma1 = partial(cov, xi=2/n, psi=args.psi)(C / n)
    ax.plot(sigma1)
    
    # Save figure
    fig.savefig(f'results/figures/pde_simulation/{cov.__name__}.png')
    fig2.savefig(f'results/figures/pde_simulation/{cov.__name__}_covariance.png')
    

if __name__ == '__main__':
    # define config
    config = dict(
        # data config
        num_dimensions=40,
        xi=tuple(jnp.arange(0.1, 3, 0.1).tolist()), #(0.1, 3, 10),
        adjust=(-1, 1),
        class_proportion=0.5,
        # model config
        model_cls=models.MLP,
        activation='relu', use_bias=False, batch_size=1000, init_scale=0.01, learning_rate=0.1, evaluation_interval=10,#10,
        # activation='sigmoid', use_bias=True, bias_value=-1, bias_trainable=False, batch_size=1000, init_scale=0.1, learning_rate=0.1, evaluation_interval=10,
        num_hiddens=1,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        optimizer_fn=optax.sgd,
        num_epochs=1000,
        # experiment config
        seed=42,#0,
        save_=True,
    )

    # simulate
    weights_nlgp, metrics_nlgp = simulate(**config, gain=100, dataset_cls=datasets.NonlinearGPDataset)
    print("Metrics")
    print(metrics_nlgp)
    fig, axs = plot_rf_evolution(weights_nlgp[:,:1], figsize=(8, 4))
    fig.savefig(f'/Users/leonlufkin/Documents/GitHub/Localization/localization/results/figures/twolayer_rf.png')