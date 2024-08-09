from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Union

# import asyncio
import datetime
# import pickle
import logging
import os
# import pandas as pd
from pathlib import Path
# import pprint
from tqdm.asyncio import tqdm

import submitit

class Executor(submitit.AutoExecutor):
    def starmap_array(self, fn: Callable, iterable: Iterable[Any]) -> List[Any]:
        submissions = [
            submitit.core.utils.DelayedSubmission(fn, **kwargs) for kwargs in iterable
        ]
        if len(submissions) == 0:
            print("Received an empty job array")
            return []
        return self._internal_process_submissions(submissions)


def get_timestamp():
    """Return a date and time `str` timestamp."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")


def get_submitit_executor(
    cluster: Literal["slurm", "local", "debug"],
    log_dir: Union[str, Path],
    timeout_min: int = 60,
    gpus_per_node: int = 0,
    cpus_per_task: int = 1,
    nodes: int = 1,
    mem_gb: int = 16,
    slurm_partition: Optional[Literal["debug", "cpu", "gpu"]] = None,
    slurm_parallelism: Optional[int] = None,
    slurm_exclude: Optional[int] = None,
) -> submitit.Executor:
    """Return a `submitit.Executor` with the given parameters."""

    if gpus_per_node > 4:
        raise ValueError("The cluster has no more than 4 GPUs per node.")

        slurm_setup = [
            "nvidia-smi",
            "echo",
            "printenv | grep LD_LIBRARY_PATH",
            "echo",
        ]
    else:
        slurm_setup = None

    executor = Executor(
        folder=os.path.join(log_dir, "%j"),
        cluster=cluster,
    )

    if gpus_per_node > 0:
        executor.update_parameters(
            timeout_min=timeout_min,
            gpus_per_node=gpus_per_node,
            cpus_per_task=cpus_per_task,
            nodes=nodes,
            mem_gb=mem_gb,
            slurm_partition=slurm_partition,
            slurm_mail_type="REQUEUE,BEGIN",
            slurm_mail_user="leon.lufkin@yale.edu",
            slurm_array_parallelism=slurm_parallelism,
            slurm_setup=slurm_setup,
        )
    else:
        executor.update_parameters(
        timeout_min=timeout_min,
        cpus_per_task=cpus_per_task,
        nodes=nodes,
        mem_gb=mem_gb,
        slurm_partition=slurm_partition,
        slurm_mail_type="REQUEUE,BEGIN",
        slurm_mail_user="leon.lufkin@yale.edu",
        slurm_array_parallelism=slurm_parallelism,
        slurm_setup=slurm_setup,
    )

    return executor

def submit_jobs(
    executor: Executor,
    func: Callable,
    kwargs_array: Iterable[Mapping],
):
    """Submit jobs via `executor` by mapping `func` over `kwargs_array`."""

    # Launch jobs.
    logging.info("Launching jobs...")
    jobs = executor.starmap_array(
        func,
        kwargs_array,
    )
    logging.info(f"Waiting for {len(jobs)} jobs to terminate...")
    scores = tuple(job.result() for job in jobs)
    logging.info("All jobs terminated.")

    return jobs

import itertools
def product_kwargs(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))