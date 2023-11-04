import os
import datetime
# export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl" # <- this should allow us to import from submit.py
from submit import get_submitit_executor, submit_jobs, product_kwargs
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import optax
from localization import datasets, models, samplers
from localization.experiments.batched_online import simulate, make_key
from itertools import product

from socket import gethostname

def get_timestamp():
    """
    Return a date and time `str` timestamp.
    Format: YYYY-MM-DD_HH-MM-SS
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def gabor_real(c, b, a, x0, k0, x, n):
    """
    Parameters
    ----------
    x : jnp.ndarray
        The input array.
    n : int
        The length of the input array.
    c : float
        The amplitude.
    b : float
        The bias.
    a : float (positive)
        The width.
    x0 : float
        The center.
    k0 : float
        The frequency.
    """
    d = jnp.minimum(x-x0, n - (x-x0))
    return c * jnp.cos(k0 * d) * jnp.exp(-d ** 2 / a ** 2) + b

def gabor_imag(c, b, a, x0, k0, x, n):
    """
    Parameters
    ----------
    x : jnp.ndarray
        The input array.
    n : int
        The length of the input array.
    c : float
        The amplitude.
    b : float
        The bias.
    a : float (positive)
        The width.
    x0 : float
        The center.
    k0 : float
        The frequency.
    """
    d = jnp.minimum(x-x0, n - (x-x0))
    return -c * jnp.sin(k0 * d) * jnp.exp(-d ** 2 / a ** 2) + b

def load(**kwargs):
    path_key = make_key(**kwargs)
    datawd = '../localization/results/weights' if gethostname() == 'Leons-MBP' else '/ceph/scratch/leonl/results/gain_sweep'
    if path_key + '.npz' in os.listdir(datawd):
        data = np.load(datawd + '/' + path_key + '.npz', allow_pickle=True)
        weights_, metrics_ = data['weights'], data['metrics']
    else:
        raise ValueError('No simulation found')
    return weights_, metrics_

def build_sweep(c, b, a, x0, k0, x, n):

    datawd = '../localization/results/weights' if gethostname() == 'Leons-MBP' else '/ceph/scratch/leonl/results/gain_sweep'
    
    if f'sweep_{n}.npy' in os.listdir(datawd):
        sweep = np.load(datawd + f'/sweep_{n}.npy')
        return sweep
        
    evaluate_gabor_real = jax.jit(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(
                        jax.vmap(
                            partial(
                                gabor_real,
                                x=jnp.arange(config_['num_dimensions']),
                                n=config_['num_dimensions']
                            ),
                            in_axes=(None, None, None, None, 0),
                        ),
                        in_axes=(None, None, None, 0, None),
                    ),
                    in_axes=(None, None, 0, None, None),
                ),
                in_axes=(None, 0, None, None, None),
            ),
            in_axes=(0, None, None, None, None),
        )
    )
    
    evaluate_gabor_imag = jax.jit(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(
                        jax.vmap(
                            partial(
                                gabor_imag,
                                x=jnp.arange(config_['num_dimensions']),
                                n=config_['num_dimensions']
                            ),
                            in_axes=(None, None, None, None, 0),
                        ),
                        in_axes=(None, None, None, 0, None),
                    ),
                    in_axes=(None, None, 0, None, None),
                ),
                in_axes=(None, 0, None, None, None),
            ),
            in_axes=(0, None, None, None, None),
        )
    )

    sweep = evaluate_gabor_real(c, b, a, x0, k0)
    
    # save it
    np.save(datawd + f'/sweep_{n}.npy', sweep)
    
    return sweep

def evaluate_sweep_(weight, sweep):
    w = weight.reshape(1, 1, 1, 1, 1, -1)
    # err = jnp.abs(sweep - w).mean(axis=-1) # l-1 norm
    err = ((sweep - w) ** 2).mean(axis=-1) ** (1/2) # l-2 norm
    # err = jnp.abs(sweep - w).max(axis=-1) # l-infinity norm
    argmin = jnp.unravel_index(jnp.argmin(err), err.shape)
    return argmin, err[argmin]

def run(config, sweep_dict, gain):
    
    sweep = build_sweep(**sweep_dict)
    
    config_ = config.copy()
    config_.update(dict(
        gain = gain
    ))
    weights, _ = load(**config_)

    argmin, err = evaluate_sweep_(weights[-1,0], sweep)
    argmin = jnp.stack(argmin).flatten()
    
    params = ['c', 'b', 'a', 'x0', 'k0']
    out = np.zeros(len(params))
    for i, param in enumerate(params):
        out[i] = sweep_dict[param][argmin[i]]
    
    return out, argmin, err
    

if __name__ == '__main__':

    executor = get_submitit_executor(
        timeout_min=60,
        mem_gb=20,
        # export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl"
        # NOTE: `log_dir` should be set to a directory shared across the head
        # (launching) node as well as compute nodes;
        # can set `export RESULTS_HOME="..." external to Python or
        # change the below.
        log_dir=Path(
            os.path.join(os.environ.get("LOGS_HOME"), "gain_analysis")
            if os.environ.get("LOGS_HOME") is not None
            else os.path.join("/tmp", os.environ.get("USER"), "gain_analysis"),
            get_timestamp(),
        ),
        # NOTE: Use `cluster="debug"` to simulate a SLURM launch locally.
        cluster="slurm",
        # NOTE: This may be specific to your cluster configuration.
        # Run `sinfo -s` to get partition information.
        slurm_partition="cpu",
        slurm_parallelism=30,
        gpus_per_node=0
    )

    config_ = dict(
        # data config
        num_dimensions=40,
        xi1=2,
        xi2=1,
        dataset_cls=datasets.NonlinearGPDataset,
        batch_size=1000,
        support=(-1, 1), # defunct
        class_proportion=0.5,
        # model config
        model_cls=models.SimpleNet,
        num_hiddens=1,
        activation='relu',
        use_bias=False,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        init_scale=1.,
        # learning config
        num_epochs=20000,
        evaluation_interval=20,
        optimizer_fn=optax.sgd,
        learning_rate=0.01,
        # experiment config
        seed=0,
        save_=True,
        wandb_=False,
    )
    
    sweep_dict = dict(
        c = jnp.linspace(-0.5, 0.5, 51),
        b = jnp.concatenate([jnp.linspace(-0.5, 0.5, 50), jnp.array([0])]),
        a = jnp.logspace(-0.5, 2, 100),
        x0 = jnp.arange(0, config_['num_dimensions'], 0.5),
        k0 = jnp.linspace(0.05, 0.5, 100),
        x = jnp.arange(config_['num_dimensions']),
        n = config_['num_dimensions'],
    )
    
    GAIN_SWEEP = jnp.array([0.01])# , 10.]) # jnp.logspace(-2, 1, 100)
    
    # all_weights = np.empty((len(GAIN_SWEEP), config_['num_dimensions']))
    # for i, gain in enumerate(GAIN_SWEEP):
    #     config = config_.copy()
    #     config.update(dict(
    #         gain = gain
    #     ))
    #     weights, _ = load(**config)
    #     all_weights[i] = weights[-1,0]
    
    # print( all_weights.shape )
    # argmin, err = evaluate_sweep(all_weights)
    # argmin = jnp.stack(argmin).T
    # print( argmin )
    # print( err )
    
    jobs = submit_jobs(
        executor=executor,
        func=run,
        kwargs_array=product_kwargs(
            config=(config_,),
            sweep_dict=(sweep_dict,),
            gain=GAIN_SWEEP,
        ),
    )
    
    opt_params = np.stack([ j.result()[0] for j in jobs ])
    errs = np.stack([ j.result()[2] for j in jobs ])
    print( opt_params.shape )
    print( errs.shape )
    
    # opt = np.empty((len(GAIN_SWEEP), 5))
    # for r, argmin_ in enumerate(argmin):
    #     print( argmin_ )
    #     for i, arg in enumerate(argmin_):
    #         opt[r,i] = locals()[['c', 'b', 'a', 'x0', 'k0'][i]][arg]
            
    # # print( locals().keys() )
    # # opt = jnp.array([ [ locals()[['c', 'b', 'a', 'x0', 'k0'][i]][arg] for i, arg in enumerate(argmin_) ] for argmin_ in argmin ])
            
    # # opt = jnp.array([[ locals()[['c', 'b', 'a', 'x0', 'k0'][i]][arg] for i, arg in enumerate(argmin_) ] for argmin_ in argmin])
    # # opt = jnp.array([[ locals()[['c', 'b', 'a', 'x0', 'k0'][i]][arg] for arg in param_argmin ] for i, param_argmin in enumerate(argmin) ])
    # print( opt )
    
    # # # double check...
    # # pred = gabor_real(
    # #     c = c[argmin[0]],
    # #     b = b[argmin[1]],
    # #     a = a[argmin[2]],
    # #     x0 = x0[argmin[3]],
    # #     k0 = k0[argmin[4]],
    # #     x = jnp.arange(config_['num_dimensions']),
    # #     n = config_['num_dimensions'],
    # # )
    # # print( pred )