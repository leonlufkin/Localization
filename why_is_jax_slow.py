import jax
import optax
import datasets
import models
import samplers
from experiments import simulate as simulate_jax # this function is defined in batched_online.py
from v_torch.conv_emergence import main as simulate_torch
import time

if __name__ == '__main__':

  # define config
  config = dict(
    seed=0,
    num_dimensions=40,
    num_hiddens=100,
    gain=1.1,
    init_scale=1.0,
    activation='tanh',
    model_cls=models.SimpleNet,
    optimizer_fn=optax.sgd,
    learning_rate=1.0,
    batch_size=5000,
    num_epochs=200,#0,
    dataset_cls=datasets.NonlinearGPDataset,
    xi1=4.47,
    xi2=0.1,
    class_proportion=0.5,
    sampler_cls=samplers.OnlineSampler,
    init_fn=models.torch_init,
    save_=False, # if True, will create a bunch of local files
    dim=1, # only the PyTorch version needs this
  )

  # run it with PyTorch (fast)
  config_torch = config.copy()
  config_torch['L'] = config_torch.pop('num_dimensions')
  config_torch['K'] = config_torch.pop('num_hiddens')
  config_torch['lr'] = config_torch.pop('learning_rate')
  config_torch['second_layer'] = 0.0
  start_time = time.time()
  simulate_torch(**config_torch)
  torch_time = time.time() - start_time
  print("\n##############################################")
  print(f"PyTorch took {torch_time:.2f} seconds.")
  print("##############################################\n")

  ## run it with JAX (slow??)
  #start_time = time.time()
  #simulate_jax(wandb_=False, **config) # wandb_=True would log results to Weights & Biases
  #jax_time = time.time() - start_time
  #print("\n##############################################")
  #print(f"JAX took {jax_time:.2f} seconds.")
  #print("##############################################\n")

  import cProfile
  cProfile.runctx('simulate_jax(wandb_=False, **config)', 
                  locals={},
                   globals={'config': config, 'simulate_jax': simulate_jax},
                  filename='jax_stats',
                  )


  import pstats
  p = pstats.Stats('jax_stats')
  from pstats import SortKey
  p.sort_stats(SortKey.CUMULATIVE).print_stats(.01)
  
