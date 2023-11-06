# export PYTHONPATH="${PYTHONPATH}:/nfs/nhome/live/leonl" # <- this should allow us to import from submit.py

import numpy as np
import matplotlib.pyplot as plt
import optax
from localization import datasets, models, samplers
from localization.experiments import load, make_key
from localization.utils import get_executor, tupify
from submit import submit_jobs, product_kwargs

def track_top_n_overlap(argsort, n=5):
    num_steps, num_inputs = argsort.shape
    ind = slice(-n, None) if n > 0 else slice(None, -n)
    final_rank = argsort[-1,ind]
    return np.array([[ i in final_rank for i in argsort[t,ind] ] for t in range(num_steps)])

def track_top_n_single(argsort, n=5):
    num_steps, num_inputs = argsort.shape
    ind = slice(-n, None) if n > 0 else slice(None, n)
    final_rank = argsort[-1,-1] if n > 0 else argsort[-1,0]
    return np.array([ final_rank in argsort[t,ind] for t in range(num_steps)])

def track_weights(weights, n=5):
    # get ranking of positions for each neuron across time
    # greatest magnitude weights have largest value
    argsort = np.argsort(weights, axis=-1)
    
    # track consistency of top n positions across time
    is_pos = weights[-1].max(axis=1) > -weights[-1].min(axis=1)
    n_ = (2 * is_pos.astype(int) - 1) * n
    top_n_overlap =  np.mean(np.stack([ track_top_n_overlap(argsort[:,i,:], n=ni) for i, ni in enumerate(n_) ], axis=1), axis=2)
    top_n_single = np.stack([ track_top_n_single(argsort[:,i,:], n=ni) for i, ni in enumerate(n_) ], axis=1)
    
    return top_n_overlap, top_n_single

def track_peaks(n=5, **config):
    """This function sees how well the Gaussian approximation can predict the peaks of the non-Gaussian weights."""
    # load weights
    weights_nlgp, metrics_nlgp = load(**config, dataset_cls=datasets.NonlinearGPDataset)
    weights_gauss, metrics_gauss = load(**config, dataset_cls=datasets.NLGPGaussianCloneDataset)

    # get error between NLGP and Gaussian clone
    err = np.sqrt(np.mean(np.square(weights_nlgp - weights_gauss), axis=(2,)))
    
    # track peaks
    overlap, single = track_weights(weights_nlgp, n=n)

    # plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    epochs_nlgp, epochs_gauss = metrics_nlgp[:,0], metrics_gauss[:,0]
    # plot losses
    ax1.plot(epochs_nlgp, metrics_nlgp[:,1], label='NLGP')
    ax1.plot(epochs_gauss, metrics_gauss[:,1], label='Gaussian clone')
    ax1.set_xscale('log')
    ax1.set_title('Loss')
    ax1.legend()
    # plot accuracies
    ax2.plot(epochs_nlgp, metrics_nlgp[:,2], label='NLGP')
    ax2.plot(epochs_gauss, metrics_gauss[:,2], label='Gaussian clone')
    ax2.set_xscale('log')
    ax2.set_title('Accuracy')
    ax2.legend()
    # plot error
    ax3.plot(epochs_nlgp, err)
    ax3.set_title('Error')
    # plot peak tracking
    ax4.plot(epochs_nlgp, overlap, label=f'Overlap among top {n}')
    ax4.plot(epochs_nlgp, single, label=f'Final peak in top {n}')
    ax4.set_xscale('log')
    ax4.set_title('Peak tracking')
    ax4.legend()
    
    # save results
    path_key = make_key(**config, dataset_cls=datasets.NonlinearGPDataset)
    fig.savefig('/ceph/scratch/leonl/peak_tracking/' + path_key + '.png', dpi=300)

if __name__ == '__main__':

    executor = get_executor(
        job_name="model_sweep",
        cluster="slurm",
        partition="cpu",
        timeout_min=60,
        mem_gb=10,
        parallelism=200,
        gpus_per_node=0,
    )

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
    def filter(**kwargs):
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
            
        return track_peaks(**kwargs)

    ## Submit jobs
    jobs = submit_jobs(
        executor=executor,
        func=filter,
        kwargs_array=product_kwargs(
            **tupify(config_),
            # These are the settings we're sweeping over
            num_hiddens=(1, 40,),
            activation=('relu', 'sigmoid',),
            use_bias=(True, False,),
            learning_rate=(1.0, 0.025, 20.0, 0.5,),
            gain=np.linspace(0.01, 5, 10),
        ),
    )

    ## Process results
    