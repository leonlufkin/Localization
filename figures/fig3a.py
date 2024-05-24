"""Code to generate Figure 3."""

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from localization import datasets
from localization import models
from localization import samplers
from localization.experiments import simulate_or_load, make_key
from localization.utils import ipr, plot_receptive_fields, plot_rf_evolution, build_gaussian_covariance, build_non_gaussian_covariance, entropy_sort, build_DRT
from scipy.special import erf
from tqdm import tqdm
import itertools

data_config = dict(
  key=jax.random.PRNGKey(0),
  num_dimensions=40, 
  dim=1,
  num_exemplars=10000,
  adjust=(-1.0, 1.0),
  class_proportion=0.5,
)

config = dict(
  seed=42,#0,
  num_dimensions=40, 
  dim=1,
  adjust=(-1.0, 1.0),
  class_proportion=0.5,
  # Model
  num_hiddens=1,
  init_scale=0.001,
  activation='relu',
  model_cls=models.SimpleNet,
  use_bias=False, bias_trainable=False, bias_value=0.0,
  optimizer_fn=optax.sgd, 
  learning_rate=0.05,
  num_steps=2000, num_epochs=2000,
  sampler_cls=samplers.EpochSampler,
  init_fn=models.xavier_normal_init,
  loss_fn='mse',
  save_=True,
  evaluation_interval=10, # 100
  # Misc
  supervise=True,
  wandb_=False, 
)

alginv = lambda x: x / jnp.sqrt(1 - x**2)

def varphi(a, x):
  a = a.reshape(1,-1)
  x = x.reshape(-1,1)
  return jnp.mean(x * erf( x / jnp.sqrt(2) * alginv (a) ), axis=0)  

def plot_varphi(a, y, kurtosis):
  fig, ax = plt.subplots(figsize=(4,2))

  # Plot the line
  ax.plot(a, y, color='black')

  ax.set_xticks([-1, -0.5, 0.5, 1])
  ax.set_yticks([-1, -0.5, 0.5, 1])

  # Remove the top and right borders
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Set the position of the left and bottom spines to the zero of the data coordinate
  ax.spines['left'].set_position(('data', 0))
  ax.spines['bottom'].set_position(('data', 0))

  # Add ticks on the left and bottom spines only
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')

  # Draw lines at x=0 and y=0
  ax.axhline(0, color='black', linewidth=0.5)
  ax.axvline(0, color='black', linewidth=0.5)

  # Draw sqrt(2/pi) * x
  ax.axline((0,0), slope=jnp.sqrt(2/jnp.pi), linestyle=(0, (5, 5)), color='black', alpha=0.5)

  # Add text
  textstr = r'$\kappa = {:.2f}$'.format(kurtosis)
  ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
          verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
  
  return fig, ax

def simulate_varphi(varphi, w_init, Sigma0, Sigma1, tau, num_steps):

  num_dimensions = len(w_init)
  w = jnp.zeros((num_steps, num_dimensions))
  w = w.at[0].set(w_init)

  for i in tqdm(range(1, num_steps)):
      sw1 = Sigma1 @ w[i-1]
      sw0 = Sigma0 @ w[i-1]
      a = sw1 / jnp.sqrt(sw1 @ w[i-1])
      f = varphi(a)
      w = w.at[i].set( w[i-1] + tau * (f - (sw0 + sw1) / 2) )
      
  return w

def simulate_kurtosis(kurtosis, w_init, Sigma0, Sigma1, tau, num_steps):

  num_dimensions = len(w_init)
  w = jnp.zeros((num_steps, num_dimensions))
  w = w.at[0].set(w_init)
  f = lambda x: jnp.sqrt(2/jnp.pi) * x + ((3 - kurtosis) / 6) * (x**3)

  for i in tqdm(range(1, num_steps)):
      sw1 = Sigma1 @ w[i-1]
      sw0 = Sigma0 @ w[i-1]
      a = sw1 / jnp.sqrt(sw1 @ w[i-1])
      w = w.at[i].set( w[i-1] + tau * (f(a) - (sw0 + sw1) / 2) )
      
  return w

if __name__ =="__main__":
  
  from tqdm import tqdm
  from functools import partial
  
  # Simulate
  
  ## Ising
  weights_ising, metrics = simulate_or_load(dataset_cls=datasets.IsingDataset, xi=(0.3, 0.7,), batch_size=100, **config)
  fig, axs = plot_rf_evolution(weights_ising[:,[0],:], figsize=(4, 2))
  fig.savefig(f'fig3/rf_evol/ising.pdf', bbox_inches='tight')
  
  ## Gaussian
  # weights_gaussian, metrics = simulate_or_load(dataset_cls=datasets.NonlinearGPDataset, gain=0.01, xi=(1, 3,), batch_size=10000, **config)
  # fig, axs = plot_rf_evolution(weights_gaussian[:,[0],:], figsize=(4, 2))
  
  # ## Alg(5)
  # weights_alg, metrics = simulate_or_load(dataset_cls=datasets.NortaDataset, marginal_qdf=datasets.AlgQDF(k=5), xi=(1, 3,), batch_size=10000, **config)
  # fig, axs = plot_rf_evolution(weights_alg[:,[0],:], figsize=(4, 2))


  # Load data
  num_samples = 100000
  
  ## Ising
  dataset = datasets.IsingDataset(xi=(0.3, 0.7), **data_config)
  x_ising, y_ising = dataset[:num_samples]
  x = x_ising.flatten()
  print(jnp.mean(x**4) / (jnp.var(x)**2)) # kurtosis

  ## Gaussian
  dataset = datasets.NonlinearGPDataset(gain=0.01, xi=(1, 3), **data_config)
  x_gaussian, y_gaussian = dataset[:num_samples]
  x = x_gaussian.flatten()
  print(jnp.mean(x**4) / (jnp.var(x)**2)) # kurtosis

  ## Alg(5)
  dataset = datasets.NortaDataset(marginal_qdf=datasets.AlgQDF(k=5), xi=(1, 3), **data_config)
  x_alg, y_alg = dataset[:num_samples]
  x = x_alg.flatten()
  print(jnp.mean(x**4) / (jnp.var(x)**2)) # kurtosis
  
  
  # Computing varphi, kurtosis
  
  a = jnp.linspace(-1, 1, 100)
    
  for model, x in zip(['ising', 'gaussian', 'alg5'], [x_ising, x_gaussian, x_alg]):

    x = x.flatten() # same as taking first entry, but we get more data
    y = varphi(a, x=x)
    kurtosis = jnp.mean(x**4) / (jnp.var(x)**2)
    
    fig, ax = plot_varphi(a, y, kurtosis)
    fig.savefig(f'fig3/varphi/{model}.pdf', bbox_inches='tight')
    plt.close(fig)
   
    
  # Numerically integrate
  
  tau = 0.05
  num_steps = 1000
  w_init = weights_ising[0,0]
  
  for model, (x, y) in zip(['ising', 'gaussian', 'alg5'], 
                                 [(x_ising, y_ising), 
                                  (x_gaussian, y_gaussian), 
                                  (x_alg, y_alg)]):
    
    print(model)
    if model != 'ising':
      continue
    
    # Compute class covariance matrices
    Sigma0 = jnp.cov(x[y==0].T)
    Sigma1 = jnp.cov(x[y==1].T)
    
    # # Integrate with full varphi
    # varphi_ = partial(varphi, x=x)
    # w_varphi = simulate_varphi(varphi_, w_init, Sigma0, Sigma1, tau, num_steps)
    # fig, ax = plot_rf_evolution(w_varphi[::10].reshape(-1,1,len(w_init)), figsize=(4, 2))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # fig.savefig(f'fig3/rf_sim/{model}_varphi.pdf', bbox_inches='tight')
        
    # Integrate with just kurtosis term
    x = x.flatten()
    kurtosis = jnp.mean(x**4) / (jnp.var(x)**2)
    w_kurtosis = simulate_kurtosis(kurtosis, w_init, Sigma0, Sigma1, tau, num_steps)
    fig, ax = plot_rf_evolution(w_kurtosis[::10].reshape(-1,1,len(w_init)), figsize=(4, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(f'fig3/rf_sim/{model}_kurtosis.pdf', bbox_inches='tight')
    