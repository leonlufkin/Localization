import os
# export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl" # <- this should allow us to import from submit.py

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import optax
from localization import datasets, models, samplers
from localization.experiments.batched_online import simulate, load, make_key
from localization.utils.modeling import gabor_real, gabor_imag

from socket import gethostname

def build_sweep(c, b, a, x0, k0, x):
    n = len(x)
    datawd = '../localization/results/gabor_fit' if gethostname() == 'Leons-MBP' else '/ceph/scratch/leonl/results/gabor_fit'
    
    # check if sweep already exists
    if f'sweep_{n}.npy' in os.listdir(datawd):
        sweep = np.load(datawd + f'/sweep_{n}.npy')
        return sweep
        
    # if not, build sweep
    evaluate_gabor_real = jax.jit(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(
                        jax.vmap(
                            partial(
                                gabor_real,
                                x=x,
                                n=n,
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
    
    # save sweep
    np.save(datawd + f'/sweep_{n}.npy', sweep)
    
    return sweep

def find_gabor_fit_(weight, sweep):
    """Given a receptive field and a grid sweep, find the indices of the best fit."""
    w = weight.reshape(1, 1, 1, 1, 1, -1)
    n = w.shape[-1]
    # err = jnp.abs(sweep - w).mean(axis=-1) # l-1 norm
    err = ((sweep - w) ** 2).sum(axis=-1) ** (1/2) / n # l-2 norm
    # err = jnp.abs(sweep - w).max(axis=-1) # l-infinity norm
    argmin = jnp.unravel_index(jnp.argmin(err), err.shape)
    return argmin, err[argmin]

def find_gabor_fit(sweep_dict, **config):
    """Given a model config, fit a Gabor to it using a grid sweep."""
    
    # build grid sweep for this input size
    sweep = build_sweep(**sweep_dict, x=jnp.arange(config['num_dimensions']))
    
    # load the model
    weights, _ = load(**config)

    # find best fit
    argmin, err = find_gabor_fit_(weights[-1,0], sweep)
    argmin = jnp.stack(argmin).flatten()
    params = ['c', 'b', 'a', 'x0', 'k0']
    fit = np.zeros(len(params))
    for i, param in enumerate(params):
        fit[i] = sweep_dict[param][argmin[i]]
    
    # save results
    fitwd = '/ceph/scratch/leonl/results/gabor_fit'
    path_key = make_key(**config)
    np.savez(f"{fitwd}/gabor_fit_{path_key}.npz", fit=fit, err=err)
    
    return config, (fit, argmin, err)
    

if __name__ == '__main__':
    
    from localization.utils.launcher import get_executor, tupify
    from submit import submit_jobs, product_kwargs

    ## Define base config
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
    
    ## Define sweep
    sweep_dict = dict(
        c = jnp.linspace(-0.5, 0.5, 30),
        b = jnp.concatenate([jnp.linspace(-0.15, 0.15, 10), jnp.array([0])]),
        a = jnp.logspace(-0.5, 2, 50),
        x0 = jnp.arange(0, config_['num_dimensions'], 0.5),
        k0 = jnp.linspace(0.05, 0.5, 80),
    )

    ## Define executor
    executor = get_executor(
        job_name="gabor_fit",
        cluster="slurm",
        partition="cpu",
        timeout_min=60,
        mem_gb=40,
        parallelism=200,
        gpus_per_node=0,
    )
    
    ## Compute best fit for each configuration
    jobs = submit_jobs(
        executor=executor,
        func=find_gabor_fit,
        kwargs_array=product_kwargs(
            sweep_dict=(sweep_dict,),
            **tupify(config_),
            gain=jnp.logspace(-2, 1, 100),
        ),
    )
    
    ## Process results
    configs = np.array([ j.result()[0] for j in jobs ])
    fits = np.stack([ j.result()[1][0] for j in jobs ])
    errs = np.stack([ j.result()[1][2] for j in jobs ])
    
    ## Save results for all jobs in one file
    datawd = '/ceph/scratch/leonl/results/gabor_fit'
    np.savez(f"{datawd}/gabor_fit.npz", configs=configs, fits=fits, errs=errs)