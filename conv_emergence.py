import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit, vmap

from jax.scipy.special import erf as gain_function
import matplotlib.pyplot as plt

from scipy.stats import entropy

import argparse
import datetime

def Z(g):
    return jnp.sqrt( (2/jnp.pi) * jnp.arcsin( (g**2) / (1 + (g**2)) ) )

def generate_gaussian(key, xi, L, dim=1, num_samples=1):
    # we are fixing dim=1 in this script
    C = jnp.abs(jnp.tile(jnp.arange(L)[:, jnp.newaxis], (1, L)) - jnp.tile(jnp.arange(L), (L, 1)))
    C = jnp.exp(-C ** 2 / (xi ** 2))
    z = jax.random.multivariate_normal(key, np.zeros(L), C, shape=(num_samples,))
    return z

# TODO(leonl): Vectorize this function with `jax.vmap` across `num_samples`!
def generate_non_gaussian(key, xi, L, g, dim=1, num_samples=1000):
    z = generate_gaussian(key, xi, L, dim=dim, num_samples=num_samples)
    x = gain_function(g * z) / Z(g)
    return x

def compute_entropy(weights, low=-10, upp=10, delta=0.1, base=2):
    entropies = np.zeros(weights.shape[0])
    for neuron, weight in enumerate(weights):
        xs = np.arange(low, upp, delta)
        count = np.zeros(len(xs)+1)
        count[0] = np.sum(weight < xs[0])
        for i in range(len(xs)-1):
            count[i] = np.sum(weight < xs[i+1]) - np.sum(weight < xs[i])
        count[-1] = np.sum(weight >= xs[-1])
        prob = count / np.sum(count)
        entropies[neuron] = entropy(prob, base=base)
    return entropies
            
    
    
    
class NLGPDataset:
    def __init__(self, L, xi1, xi2, g, batch_size=1, num_epochs=1):
        self.L = L
        self.xi1 = xi1
        self.xi2 = xi2
        self.g = g
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
    def __len__(self):
        return self.num_epochs
    
    def __getitem__(self, idx):
        num_true = np.random.binomial(self.batch_size, 0.5)
        X = jnp.zeros((0, self.L))
        y = jnp.zeros(0)
        if num_true > 0:
            X = generate_non_gaussian(self.L, self.xi1, self.g, num_samples=num_true, dim=self.dim).reshape(-1, self.D)
            y = np.ones(num_true)
        if num_true < self.batch_size:
            X_ = generate_non_gaussian(self.L, self.xi2, self.g, num_samples=self.batch_size-num_true, dim=self.dim).reshape(-1, self.D)
            y_ = -np.ones(self.batch_size-num_true)
        ind = np.random.permutation(self.batch_size)
        X = np.concatenate((X, X_), axis=0)[ind]
        y = np.concatenate((y, y_), axis=0)[ind]
        
        X = torch.from_numpy(X).float()
        y = torch.tensor(y).float()
        
        return X, y

class DataLoader:
    pass
    
class NLGPLoader(DataLoader):
    """
    Faster than generating one draw at a time, I hope.
    """
    def __init__(self, *args, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        dataset = NLGPDataset(*args, batch_size=batch_size, **kwargs)
        super().__init__(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        
def get_timestamp():
    """
    Return a date and time `str` timestamp.
    Format: MM-DD_HH-MM-SS
    """
    return datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        
def parse_args():
    # read command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--inputs", nargs="+", default=["nlgp"], help="inputs: nlgp | gp")
    parser.add_argument("--xi1", type=float, default=0.1, help="correlation length 1")
    parser.add_argument("--xi2", type=float, default=1.1, help="correlation length 2")
    parser.add_argument("--gain", type=float, default=1, help="gain of the NLGP")
    
    parser.add_argument("--L", type=int, default=400, help="linear input dimension. The input will have D**dim pixels.")
    parser.add_argument("--K", type=int, default=8, help="# of student nodes / channels of the student")
    parser.add_argument("--dim", type=int, default=1, help="input dimension: one for vector, two for images. The input will have D**dim pixels")
        
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=5000, help="number of epochs")
    parser.add_argument("--loss", type=str, default="mse", help="loss: mse | ce")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    
    parser.add_argument("--activation", default="tanh", help="activation: tanh | sigmoid | relu")
    parser.add_argument("--second_layer", default="linear", help="second layer: linear | learnable_bias | float (fixed bias value)")
        
    args = parser.parse_args()
    return vars(args)
        
def main(
    xi1, xi2, gain,
    L, K, dim,
    batch_size, num_epochs, loss='mse', lr=0.01,
    activation='tanh', second_layer='linear',
    path='.', **kwargs
):
    # set up model
    activation_fn = nn.Tanh() if activation == 'tanh' else nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()
    model = NeuralNet(input_dim=L ** dim, hidden_dim=K, activation=activation_fn, second_layer=second_layer)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # set up data
    loader = NLGPLoader(L, xi1, xi2, gain, dim, batch_size=batch_size, num_epochs=num_epochs, shuffle=False, num_workers=0)
    loss_fn = nn.MSELoss() if loss == 'mse' else nn.BCEWithLogitsLoss()

    # train
    losses = np.zeros(num_epochs)
    accs = np.zeros(num_epochs)
    iprs = []
    every_epoch = min(max(num_epochs // 100, 1), 500)
    for epoch, (X, y) in enumerate(loader):
        X, y = X.squeeze(0), y.squeeze(0)
        yhat = model(X)
        loss_ = loss_fn(yhat, y)
        opt.zero_grad()
        loss_.backward()
        opt.step()
        
        losses[epoch] = loss_.cpu().item()
        accs[epoch] = acc_ = ((yhat > 0) == (y > 0)).to(torch.float32).mean().cpu().item()
        
        # print progress, record IPR
        if epoch % every_epoch == 0 or epoch == num_epochs - 1:
            weights = model.ff1.weight.detach().cpu().numpy()
            ipr_ = np.power(weights, 4).sum(axis=1) / np.power(np.power(weights, 2).sum(axis=1), 2)
            iprs.append(ipr_)
            print(f'Epoch {epoch}: loss={losses[max(epoch-every_epoch,0):epoch+1].mean():.4f}, acc={accs[max(epoch-every_epoch,0):epoch+1].mean():.4f}, IPR>0.05={100 * np.mean(ipr_ > 0.05):.2f}%')
    
            if False: #loss_.item() < 5e-3 or acc_ > 0.99:
                print(f'Breaking early with loss={loss_:.4f}, acc={acc_:.4f}')
                losses[epoch+1:] = np.nan
                accs[epoch+1:] = np.nan
                break
    
    # make ipr an array
    iprs = np.array(iprs)
        
    # key
    key = f'__xi1={xi1:05.2f}_xi2={xi2:05.2f}_gain={gain:05.2f}_L={L:03}_K={K:03}_dim={dim}_batch_size={batch_size}_num_epochs={num_epochs}_loss={loss}_lr={lr:.3f}_activation={activation}_second_layer={second_layer}'
    print(f'key={key}')
         
    # save losses, accs, iprs
    np.savez(f'{path}/results/metrics_{key}.npz', losses=losses, accs=accs, iprs=iprs)
    print('Saved losses, accs, iprs')
        
    # save model weights
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{path}/results/weights_{key}.pt')
    print('Saved model weights')
        
if __name__ == '__main__':
    
    # get arguments
    kwargs = parse_args()
    print("Arguments:")
    for arg, val in kwargs.items():
        print(f"{arg}: {val}")
    
    # main    
    main(**kwargs)
