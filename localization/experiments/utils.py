import os
from socket import gethostname
import jax.numpy as jnp
from localization.utils import make_key, localization_make_key
from localization.experiments import supervise as supervise_
from localization.experiments import autoencode

from jaxnets.utils import load as jaxnets_load
from jaxnets.utils import simulate as jaxnets_simulate
from jaxnets.utils import simulate_or_load as jaxnets_simulate_or_load

##########
## LOADING

def localization_load(**kwargs):
  weightwd = '../results/weights'
  path_key = make_key(**kwargs)
  print(path_key)
  if path_key + '.npz' in os.listdir(weightwd):
    print('Already simulated')
    data = jnp.load(weightwd + '/' + path_key + '.npz', allow_pickle=True)
    return data['weights'], data['metrics']
  print('Could not find using make_key, trying localization_make_key')
  path_key = localization_make_key(**kwargs)
  print(path_key)
  if path_key + '.npz' in os.listdir(weightwd):
    print('Already simulated')
    data = jnp.load(weightwd + '/' + path_key + '.npz', allow_pickle=True)
    return data['weights'], data['metrics']
  # try removing the seed, accounts for change in make_key implemented on 12-04-2023 that adds seed to path_key
  if 'seed' in kwargs.keys():
    kwargs.pop('seed')
    return localization_load(**kwargs)
  
  raise ValueError('File ' + path_key + '.npz' + ' not found')

def load(**kwargs):
  try:
    return jaxnets_load(**kwargs)
  except FileNotFoundError as e:
    print(e)
    print('jaxnets_load failed, trying localization_load')
    return localization_load(**kwargs)


#############
## SIMULATING

def localization_simulate(supervise=True, **kwargs):
  if supervise:
    return supervise_(**kwargs)
  else:
    return autoencode(**kwargs)
  
def simulate(**kwargs):
  try:
    return jaxnets_simulate(**kwargs)
  except Exception as e: # catch basically anything
    print(e)
    print('jaxnets_simulate failed, trying localization_simulate')
    return localization_simulate(**kwargs)

# SIMULATE OR LOAD

def localization_simulate_or_load(**kwargs):
  try:
    weights_, metrics_ = load(**kwargs)
  except ValueError as e:
    print(e)
    print('Simulating')
    weights_, metrics_ = simulate(**kwargs)
  return weights_, metrics_

def simulate_or_load(**kwargs):
  try:
    return jaxnets_simulate_or_load(**kwargs)
  except FileNotFoundError as e:
    print(e)
    print('jaxnets_simulate_or_load failed, trying localization_simulate_or_load')
    return localization_simulate_or_load(**kwargs)