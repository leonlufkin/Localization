# export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl" # <- this should allow us to import from submit.py

import numpy as np
import optax
from localization import datasets, models, samplers
from localization.experiments import simulate, simulate_or_load

if __name__ == '__main__':
    
    from localization.utils.launcher import get_executor, tupify
    from submit import submit_jobs, product_kwargs

    executor = get_executor(
        job_name="model_sweep",
        cluster="slurm",
        partition="cpu",
        timeout_min=60,
        mem_gb=10,
        parallelism=30,
        gpus_per_node=0,
    )
    
    # get_submitit_executor(
    #     timeout_min=60,
    #     mem_gb=10,
    #     # export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl"
    #     # NOTE: `log_dir` should be set to a directory shared across the head
    #     # (launching) node as well as compute nodes;
    #     # can set `export RESULTS_HOME="..." external to Python or
    #     # change the below.
    #     log_dir=Path(
    #         os.path.join(os.environ.get("LOGS_HOME"), "gain_sweep")
    #         if os.environ.get("LOGS_HOME") is not None
    #         else os.path.join("/tmp", os.environ.get("USER"), "gain_sweep"),
    #         get_timestamp(),
    #     ),
    #     # NOTE: Use `cluster="debug"` to simulate a SLURM launch locally.
    #     cluster="slurm",
    #     # NOTE: This may be specific to your cluster configuration.
    #     # Run `sinfo -s` to get partition information.
    #     slurm_partition="cpu",
    #     slurm_parallelism=200,
    #     gpus_per_node=0
    # )

    ## Define base config
    config_ = dict(
        # data config
        num_dimensions=40,
        xi1=2,
        xi2=1,
        batch_size=50000,
        support=(-1, 1), # defunct
        class_proportion=0.5,
        # model config
        model_cls=models.SimpleNet,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        num_hiddens=1,
        activation='relu',
        use_bias=False,
        # learning config
        evaluation_interval=10,
        optimizer_fn=optax.sgd,
        # experiment config
        save_=True,
        wandb_=False,
    )
    
    # helper function to only sweep across subset of hyperparameters
    def filter(**kwargs):
        init_scale = kwargs['init_scale']
        activation = kwargs['activation']
        gain = kwargs['gain']
        learning_rate = kwargs['learning_rate']
        num_epochs = kwargs['num_epochs']
        
        if learning_rate == 0.1 and num_epochs == 10000:
            return
        if learning_rate == 0.02 and num_epochs == 2000:
            return
        
        if learning_rate == 0.1 and (gain == 3 or gain == 100) and init_scale == 0.01: # already ran
            return
            
        # NOTE: using `simulate_or_load` will effectively skip jobs that have already been run
        #       if one needs to re-run jobs (specifically when using a new evaluation_interval), use `simulate` instead
        return simulate_or_load(**kwargs)

    ## Submit jobs
    jobs = submit_jobs(
        executor=executor,
        func=filter,
        kwargs_array=product_kwargs(
            **tupify(config_),
            # These are the settings we're sweeping over
            seed=tuple(np.arange(30)),
            init_scale=(0.001, 0.01, 0.5), # already did 0.01
            dataset_cls=(datasets.NonlinearGPDataset, datasets.NLGPGaussianCloneDataset,),
            gain=(1.1, 3, 100,),
            learning_rate=(0.02, 0.1,),
            num_epochs=(10000, 2000),
        ),
    )
