import os
import datetime
# export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl" # <- this should allow us to import from submit.py
from submit import get_submitit_executor, submit_jobs, product_kwargs
from pathlib import Path

import numpy as np
import jax.nn as jnn
import optax
from localization import datasets, models, samplers
from localization.experiments import simulate

def get_timestamp():
    """
    Return a date and time `str` timestamp.
    Format: YYYY-MM-DD_HH-MM-SS
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def tupify(d: dict):
    return { k: (v,) for k, v in d.items() }

if __name__ == '__main__':

    executor = get_submitit_executor(
        timeout_min=60,
        mem_gb=10,
        # export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl"
        # NOTE: `log_dir` should be set to a directory shared across the head
        # (launching) node as well as compute nodes;
        # can set `export RESULTS_HOME="..." external to Python or
        # change the below.
        log_dir=Path(
            os.path.join(os.environ.get("LOGS_HOME"), "gain_sweep")
            if os.environ.get("LOGS_HOME") is not None
            else os.path.join("/tmp", os.environ.get("USER"), "gain_sweep"),
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
        activation='relu',
        use_bias=False,
        sampler_cls=samplers.EpochSampler,
        init_fn=models.xavier_normal_init,
        init_scale=1.,
        # learning config
        num_epochs=20000,
        evaluation_interval=20,
        optimizer_fn=optax.sgd,
        # experiment config
        seed=0,
        save_=True,
        wandb_=False,
    )

    jobs = submit_jobs(
        executor=executor,
        func=simulate,
        kwargs_array=product_kwargs(
            **tupify(config_),
            # NOTE: This is the only line that changes between experiments.
            gain=np.logspace(-2, 1, 100),
            num_hiddens=(1,),
            learning_rate=(0.01,),
        ),
    )
