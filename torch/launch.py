import os
import datetime
from submit import get_submitit_executor, submit_jobs, product_kwargs
from pathlib import Path

import numpy as np
from conv_emergence import main

def get_timestamp():
    """
    Return a date and time `str` timestamp.
    Format: YYYY-MM-DD_HH-MM-SS
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
        
if __name__ == '__main__':

    # define parameter range  
    xi1 = (np.sqrt(20), 10.)
    xi2 = (0.1,)# np.sqrt(10),)
    gain = (0.1,)# 1.1, 3.0,)
    L = (20,)# 40, 100,)
    K = (40,)# 100, 300,)
    dim = (1,)
    batch_size = (1000,)
    num_epochs = (10,)#00000,)
    lr = (0.1,)
    second_layer = ('linear',)# 'learnable_bias', 0.,)
    path = ('/ceph/scratch/leonl',)  
        
    # from Erin
    executor = get_submitit_executor(
        timeout_min=480,
        mem_gb=10,
        # NOTE: `log_dir` should be set to a directory shared across the head
        # (launching) node as well as compute nodes;
        # can set `export RESULTS_HOME="..." external to Python or
        # change the below.
        log_dir=Path(
            os.path.join(os.environ.get("RESULTS_HOME"), "conv-emergence")
            if os.environ.get("RESULTS_HOME") is not None
            else os.path.join("/tmp", os.environ.get("USER"), "conv-emergence"),
            get_timestamp(),
        ),
        # NOTE: Use `cluster="debug"` to simulate a SLURM launch locally.
        cluster="debug",
        # NOTE: This may be specific to your cluster configuration.
        # Run `sinfo -s` to get partition information.
        slurm_partition="gpu",
    )
    
    jobs = submit_jobs(
        executor=executor,
        func=main,
        kwargs_array=product_kwargs(
            xi1=xi1, xi2=xi2, gain=gain,
            L=L, K=K, dim=dim,
            batch_size=batch_size, num_epochs=num_epochs, lr=lr,
            second_layer=second_layer,
            path=path
        ),
    )
    