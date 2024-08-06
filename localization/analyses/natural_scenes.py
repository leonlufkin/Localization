import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from localization.experiments import load, simulate_or_load
from localization.utils import sweep_func, product_kwargs

from jax import Array
from typing import Tuple
from functools import partial
import ipdb
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import pandas as pd
    from localization.utils import tupify, ipr, plot_rf_evolution
    from localization.experiments.model_sweep import config    
    from localization import datasets, models
    
    