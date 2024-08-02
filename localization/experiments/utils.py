import os
from socket import gethostname
import jax.numpy as jnp
from localization.utils import make_key
from localization.experiments import supervise as supervise_
from localization.experiments import autoencode

def simulate(supervise=True, **kwargs):
  if supervise:
    return supervise_(**kwargs)
  else:
    return autoencode(**kwargs)

def load(**kwargs):
  path_key = make_key(**kwargs)
  print(path_key)
  weightwd = '/Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights' if gethostname() == 'Leons-MBP' else '/ceph/scratch/leonl/results/weights'
  if path_key + '.npz' in os.listdir(weightwd):
    print('Already simulated')
    data = jnp.load(weightwd + '/' + path_key + '.npz', allow_pickle=True)
    return data['metrics'], data['weights']
  # print('File ' + path_key + '.npz' + ' not found')
  # try removing the seed, accounts for change in make_key implemented on 12-04-2023 that adds seed to path_key
  if 'seed' in kwargs.keys():
    kwargs.pop('seed')
    return load(**kwargs)
  
  raise ValueError('File ' + path_key + '.npz' + ' not found')

def simulate_or_load(**kwargs):
  try:
    weights_, metrics_ = load(**kwargs)
  except ValueError as e:
    print(e)
    print('Simulating')
    weights_, metrics_ = simulate(**kwargs)
  return weights_, metrics_