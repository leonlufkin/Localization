import numpy as np

from scipy.special import erf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.stats import entropy

import argparse
import datetime

def Z(g):
    return np.sqrt( (2/np.pi) * np.arcsin( (g**2) / (1 + (g**2)) ) )

def generate_gaussian(L, xi, dim=1, num_samples=1):
    if dim > 2:
        raise NotImplementedError("dim > 2 not implemented")
    
    C = np.abs(np.tile(np.arange(L)[:, np.newaxis], (1, L)) - np.tile(np.arange(L), (L, 1)))
    C = np.exp(-C ** 2 / (xi ** 2))
    
    if dim > 1:
        C = np.kron(C, C)
    
    z = np.random.multivariate_normal(np.zeros(L ** dim), C, size=num_samples)
    if dim > 1:
        z = z.reshape((num_samples, L, L))
        
    return z

def gain_function(x):
    return erf(x)

def generate_non_gaussian(L, xi, g, dim=1, num_samples=1):
    z = generate_gaussian(L, xi, dim=dim, num_samples=num_samples)
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
            
    
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def get_param_count(self) -> int:
        """Count the number of params"""
        return sum((np.prod(p.size()) for p in self.parameters()))

    def get_trainable_param_count(self) -> int:
        """Count the number of trainable params"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum((np.prod(p.size()) for p in model_parameters))

    def freeze_weights(self) -> None:
        """Freeze the model"""
        for param in self.parameters():
            param.requires_grad = False
            
class NeuralNet(Model):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 activation=nn.Sigmoid(),
                 second_layer='linear'):
        super().__init__()
        
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        self.ff1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.ff1.weight.data.normal_(0, 1/np.sqrt(input_dim)) # mean 0, variance 1/D
        self.ff1.bias.data.zero_()
        
        if second_layer == 'linear':
            self.ff2 = nn.Linear(hidden_dim, 1, bias=True)
        elif second_layer == 'learnable_bias' or isinstance(second_layer, float):
            class Bias(nn.Module):
                def __init__(self, learnable_bias=True, bias_value=0.):
                    super().__init__()
                    bias = bias_value * torch.ones(1, device=device)
                    self.bias = nn.Parameter(bias, requires_grad=learnable_bias)
                def forward(self, x):
                    x = torch.mean(x, dim=1).unsqueeze(1) # NOTE: we are taking the mean over the channels, rather than the sum
                    x = x + self.bias
                    return x
                def to(self, device):
                    return self
            if second_layer == 'learnable_bias':
                learnable_bias = True
                bias_value = 0.
            else:
                learnable_bias = False
                bias_value = second_layer
            self.ff2 = Bias(learnable_bias, bias_value)
        else:
            raise NotImplementedError("second_layer must be 'linear', 'learnable_bias', or a float")
            
        self.ff1 = self.ff1.to(device)
        self.ff2 = self.ff2.to(device)
            
    def forward(self, x):
        x = self.activation(self.ff1(x))
        out = self.ff2(x)
        return out.squeeze(1)

    def to(self, device):
        self.ff1 = self.ff1.to(device)
        self.ff2 = self.ff2.to(device)
        return self

def generate_pulse(L, xi, g, dim=1, num_samples=1):
    if dim > 1:
        raise NotImplementedError("dim > 1 not implemented")
    
    lengths = np.random.randint(0, xi, size=num_samples)
    


class NLGPDataset(Dataset):
    def __init__(self, L, xi1, xi2, g, dim=1, batch_size=1, num_epochs=1):
        self.L = L
        self.dim = dim
        self.D = L ** dim
        self.xi1 = xi1
        self.xi2 = xi2
        self.g = g
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        
    def __len__(self):
        return self.num_epochs
    
    def __getitem__(self, idx):
        num_true = np.random.binomial(self.batch_size, 0.5)
        X = np.zeros((0, self.D))
        y = np.zeros(0)
        if num_true > 0:
            X = generate_non_gaussian(self.L, self.xi1, self.g, num_samples=num_true, dim=self.dim).reshape(-1, self.D)
            y = np.ones(num_true)
        if num_true < self.batch_size:
            X_ = generate_non_gaussian(self.L, self.xi2, self.g, num_samples=self.batch_size-num_true, dim=self.dim).reshape(-1, self.D)
            y_ = -np.ones(self.batch_size-num_true)
        ind = np.random.permutation(self.batch_size)
        X = np.concatenate((X, X_), axis=0)[ind]
        y = np.concatenate((y, y_), axis=0)[ind]
        
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.tensor(y).float().to(self.device)
        
        return X, y
    
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
        
def float_if_possible(x):
    try:
        x = float(x)
    except:
        pass
    return x        
        
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
    parser.add_argument("--second_layer", type=float_if_possible, default="linear", help="second layer: linear | learnable_bias | float (fixed bias value)")
        
    args = parser.parse_args()
    return vars(args)
        
def main(
    xi1, xi2, gain,
    L, K, dim,
    batch_size, num_epochs, loss='mse', lr=0.01,
    activation='tanh', second_layer='linear',
    path='.', save_=True, **kwargs
):
    # pring args
    print("Arguments:")
    for arg, val in locals().items():
        print(f"{arg}: {val}")
    
    # set up model
    activation_fn = nn.Tanh() if activation == 'tanh' else nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()
    model = NeuralNet(input_dim=L ** dim, hidden_dim=K, activation=activation_fn, second_layer=second_layer)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # save initial weights
    # weight = model.ff1.weight.data.detach().cpu().numpy()
    # bias = model.ff1.bias.data.detach().cpu().numpy()
    # np.savez(f'{path}/initial_weights.npz', weight=weight, bias=bias)
    # return

    # set up data
    loader = NLGPLoader(L, xi1, xi2, gain, dim, batch_size=batch_size, num_epochs=num_epochs, shuffle=False, num_workers=0)
    loss_fn = nn.MSELoss() if loss == 'mse' else nn.BCEWithLogitsLoss()

    # train
    losses = np.zeros(num_epochs)
    accs = np.zeros(num_epochs)
    iprs = []
    every_epoch = min(max(num_epochs // 10, 1), 500) # 10 was 100
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
    if save_:
        key = f'__xi1={xi1:05.2f}_xi2={xi2:05.2f}_gain={gain:05.2f}_L={L:03}_K={K:03}_dim={dim}_batch_size={batch_size}_num_epochs={num_epochs}_loss={loss}_lr={lr:.3f}_activation={activation}_second_layer={second_layer}'
        print(f'key={key}')
            
        # save losses, accs, iprs
        np.savez(f'{path}/../results/metrics_{key}.npz', losses=losses, accs=accs, iprs=iprs)
        print('Saved losses, accs, iprs')
            
        # save model weights
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{path}/../results/weights_{key}.pt')
        print('Saved model weights')
        
if __name__ == '__main__':
    
    # get arguments
    kwargs = parse_args()
    # print("Arguments:")
    # for arg, val in kwargs.items():
    #     print(f"{arg}: {val}")
    
    # main    
    main(**kwargs)