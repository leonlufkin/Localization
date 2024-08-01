# export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl" # <- this should allow us to import from submit.py

import numpy as np
import jax
import jax.numpy as jnp
import optax
from localization import datasets, models, samplers
from localization.experiments import simulate, simulate_or_load

if __name__ == '__main__':
    
    from localization.utils.launcher import get_executor, tupify
    from localization.utils.submit import submit_jobs, product_kwargs

    cpu_executor = get_executor(
        job_name="model_sweep_rebuttal",
        cluster="slurm",
        partition="cpu",
        timeout_min=1200,
        mem_gb=10,
        parallelism=100,
        gpus_per_node=0,
    )
    
    gpu_executor = get_executor(
        job_name="model_sweep_rebuttal",
        cluster="slurm",
        partition="gpu",
        timeout_min=180,
        mem_gb=10,
        parallelism=20,
        gpus_per_node=1,
    )
    
    ## Define base config
    config = dict(
        # data config
        # num_dimensions=40,
        # xi1=2,
        # xi2=1,
        xi=(0.3, 0.7),
        # dataset_cls=datasets.NonlinearGPDataset,
        batch_size=50000,
        adjust=(-1.0, 1.0), # not really used
        class_proportion=0.5,
        # model config
        model_cls=models.SimpleNet,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        init_scale=0.001,
        num_hiddens=1,
        activation='relu',
        use_bias=False,
        learning_rate=0.1,
        num_epochs=1000,
        # learning config
        evaluation_interval=100,
        optimizer_fn=optax.sgd,
        # experiment config
        save_=True,
        wandb_=False,
    )
    
    # helper function to only sweep across subset of hyperparameters
    def filter(**config):
        # NOTE: using `simulate_or_load` will effectively skip jobs that have already been run
        #       if one needs to re-run jobs (specifically when using a new evaluation_interval), use `simulate` instead
        dataset_cls = config['dataset_cls']
        
        out = []
        if dataset_cls == datasets.NonlinearGPDataset:
            for g in jnp.logspace(-2, 2, 10):
                out.append(simulate_or_load(**config, gain=g))
        elif dataset_cls == datasets.NortaDataset:
            for k in jnp.linspace(1, 10, 10):
                out.append(simulate_or_load(**config, marginal_qdf=datasets.AlgQDF(k=k)))
            
        return out

    ## Submit jobs
    jobs = submit_jobs(
        executor=cpu_executor,
        func=filter,
        kwargs_array=product_kwargs(
            **tupify(config),
            # These are the settings we're sweeping over
            seed=tuple(np.arange(30)),
            num_dimensions=(40, 100, 400),
            # gain / AlgQDF(k)
            dataset_cls=(datasets.NonlinearGPDataset, datasets.NortaDataset,),
        ),
    )
