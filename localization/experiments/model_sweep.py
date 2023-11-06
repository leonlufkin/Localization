# export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl" # <- this should allow us to import from submit.py

import numpy as np
import optax
from localization import datasets, models, samplers
from localization.experiments import simulate

if __name__ == '__main__':
    
    from localization.utils.launcher import get_executor, tupify
    from submit import submit_jobs, product_kwargs

    executor = get_executor(
        job_name="model_sweep",
        cluster="slurm",
        partition="cpu",
        timeout_min=60,
        mem_gb=10,
        parallelism=200,
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
        batch_size=1000,
        support=(-1, 1), # defunct
        class_proportion=0.5,
        # model config
        model_cls=models.SimpleNet,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        init_scale=1.,
        # learning config
        num_epochs=5000,
        evaluation_interval=10,
        optimizer_fn=optax.sgd,
        # experiment config
        seed=0,
        save_=True,
        wandb_=False,
    )
    
    # helper function to only sweep across subset of hyperparameters
    def simulate_(**kwargs):
        activation = kwargs['activation']
        num_hiddens = kwargs['num_hiddens']
        learning_rate = kwargs['learning_rate']
        use_bias = kwargs['use_bias']
        
        if activation == 'relu':
            if num_hiddens == 40 and learning_rate != 1.0:
                return
            if num_hiddens == 1 and learning_rate != 0.025:
                return
            
        if activation == 'sigmoid':
            if not use_bias:
                return
            if num_hiddens == 40 and learning_rate != 20.0:
                return
            if num_hiddens == 1 and learning_rate != 0.5:
                return
            
        return simulate(**kwargs)

    ## Submit jobs
    jobs = submit_jobs(
        executor=executor,
        func=simulate_,
        kwargs_array=product_kwargs(
            **tupify(config_),
            # These are the settings we're sweeping over
            dataset_cls=(datasets.NonlinearGPDataset, datasets.NLGPGaussianCloneDataset,),
            num_hiddens=(1, 40,),
            activation=('relu', 'sigmoid',),
            use_bias=(True, False,),
            learning_rate=(1.0, 0.025, 20.0, 0.5,),
            gain=np.linspace(0.01, 5, 10),
        ),
    )
