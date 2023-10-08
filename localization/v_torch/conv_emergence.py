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
import ipdb

def xi_to_str(xi):
  if isinstance(xi, tuple):
    start, end = "[", ")"
  else:
    xi = (xi,)
    start, end = "", ""
  out = ",".join([ f"{x:05.2f}" for x in xi ])
  return f"{start}{out}{end}"

def make_key(task, xi1, xi2, batch_size, num_epochs, loss, lr, second_layer, L, K, activation, init_scale, gain=None, **extra_kwargs):
  return f'{task}_xi1={xi_to_str(xi1)}_xi2={xi_to_str(xi2)}'\
    f'{f"_gain={gain:.3f}" if task == "nlgp" else ""}_p=0.5'\
    f'_batch_size={batch_size}_num_epochs={num_epochs}'\
    f'_loss={loss}_lr={lr:.3f}'\
    f'_{second_layer}_L={L:03d}_K={K:03d}_activation={activation}'\
    f'_init_scale={init_scale:.3f}'

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
                 second_layer='linear',
                 second_out_size=1,
                 init_scale=1.0):
        super().__init__()
        
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        # self.ff1 = nn.Parameter(torch.zeros(input_dim, hidden_dim), requires_grad=True)
        self.ff1 = nn.Linear(input_dim, hidden_dim, bias=True)
        # self.ff2 = nn.Parameter(torch.ones(hidden_dim, second_out_size) / self.hidden_dim, requires_grad=True)
        self.ff2 = nn.Linear(hidden_dim, second_out_size, bias=True)
        
        self.ff1.weight.data.normal_(0, init_scale * 1/np.sqrt(input_dim)) # mean 0, variance 1/D
        self.ff1.bias.data[:] = 10. #zero_()
        
        # self.ff1.bias.requires_grad = False
        
        if second_layer == 'linear':
            self.ff2.weight.data.normal_(0, init_scale * 1/np.sqrt(input_dim)) # mean 0, variance 1/D
            self.ff2.bias.data.zero_()
        elif second_layer == 'mean':
            self.ff2.weight.data[:] = torch.ones(second_out_size, hidden_dim) / self.hidden_dim
            self.ff2.bias.data[:] = torch.zeros(second_out_size)
            self.ff2.weight.requires_grad = False
            self.ff2.bias.requires_grad = False
            
        # for param in self.parameters():
        #     if param.requires_grad:
        #         param.data.normal_(0, init_scale * 1/np.sqrt(input_dim)) # mean 0, variance 1/D
            
        self.ff1 = self.ff1.to(device)
        self.ff2 = self.ff2.to(device)
            
    def forward(self, x):
        # x = x @ self.ff1
        x = self.ff1(x)
        x = self.activation(x)
        # out = x @ self.ff2
        out = self.ff2(x)
        return out.squeeze(1)

    def to(self, device):
        self.ff1 = self.ff1.to(device)
        self.ff2 = self.ff2.to(device)
        return self


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
            # y_ = -np.ones(self.batch_size-num_true)
            y_ = np.zeros(self.batch_size-num_true)
        ind = np.random.permutation(self.batch_size)
        X = np.concatenate((X, X_), axis=0)[ind]
        y = np.concatenate((y, y_), axis=0)[ind]
        
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.tensor(y).float().to(self.device)
        
        return X, y
    
def generate_pulse(L, xi_lower, xi_upper, num_samples=1):
    start = np.random.randint(size=(num_samples,1), low=0, high=L)
    stop = start + np.random.randint(size=(num_samples,1), low=xi_lower, high=xi_upper) + 1
    l = np.tile(np.arange(L), (num_samples, 1))
    X = np.where(start <= l, 1., 0.) * np.where(l < stop, 1., 0.) + np.where(l < stop - L, 1., 0.)
    return 2 * X - 1    
    
class SinglePulseDataset(Dataset):
    def __init__(self, L, xi1, xi2, batch_size=1, num_epochs=1, *args, **kwargs):
        self.L = L
        self.xi1 = (int(xi1[0] * L), int(xi1[1] * L))
        self.xi2 = (int(xi2[0] * L), int(xi2[1] * L))
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        
    def __len__(self):
        return self.num_epochs
    
    def __getitem__(self, idx):
        num_true = np.random.binomial(self.batch_size, 0.5)
        X, y = np.zeros((0, self.L)), np.zeros(0)
        if num_true > 0:
            X = generate_pulse(self.L, self.xi1[0], self.xi1[1], num_samples=num_true).reshape(-1, self.L)
            y = np.ones(num_true)
        X_, y_ = np.zeros((0, self.L)), np.zeros(0)
        if num_true < self.batch_size:
            X_ = generate_pulse(self.L, self.xi2[0], self.xi2[1], num_samples=self.batch_size-num_true).reshape(-1, self.L)
            # y_ = -np.ones(self.batch_size-num_true)
            y_ = np.zeros(self.batch_size-num_true)
        ind = np.random.permutation(self.batch_size)
        X = np.concatenate((X, X_), axis=0)[ind]
        y = np.concatenate((y, y_), axis=0)[ind]
        
        # randomly flip sign of pulses in X
        sign = np.random.choice([-1, 1], size=(self.batch_size,1))
        X *= sign
        
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
        
class SinglePulseLoader(DataLoader):
    """
    Faster than generating one draw at a time, I hope.
    """
    def __init__(self, *args, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        dataset = SinglePulseDataset(*args, batch_size=batch_size, **kwargs)
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
      
def ipr(weights):
    return np.power(weights, 4).sum(axis=1) / np.power(np.power(weights, 2).sum(axis=1), 2)
        
def main(
    task,
    xi1, xi2,
    L, K, 
    gain=1.1, dim=1,
    batch_size=1000, num_epochs=200, loss='mse', lr=0.01,
    activation='tanh', second_layer='linear', init_scale=1.0,
    save_=True, **kwargs
):
    # pring args
    print("Arguments:")
    for arg, val in locals().items():
        print(f"{arg}: {val}")

    # save initial weights
    # weight = model.ff1.weight.data.detach().cpu().numpy()
    # bias = model.ff1.bias.data.detach().cpu().numpy()
    # np.savez(f'{path}/initial_weights.npz', weight=weight, bias=bias)
    # return

    # set up data
    if task == 'nlgp':
        loader = NLGPLoader(L, xi1, xi2, gain, dim, batch_size=batch_size, num_epochs=num_epochs, shuffle=False, num_workers=0)
    elif task == 'single_pulse':
        loader = SinglePulseLoader(L, xi1, xi2, batch_size=batch_size, num_epochs=num_epochs, shuffle=False, num_workers=0)
    else:
        raise NotImplementedError("task must be 'nlgp' or 'single_pulse'")
    
    # set up activation & loss functions
    second_out_size = 1
    if activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
        if second_layer == 'linear':
            loss_fn = nn.MSELoss() if loss == 'mse' else nn.BCEWithLogitsLoss()
        elif second_layer == 'mean':
            loss_fn = nn.MSELoss() if loss == 'mse' else nn.BCELoss()
    elif activation == 'relu':
        activation_fn = nn.ReLU()
        if second_layer == 'linear':
            loss_fn = nn.MSELoss() if loss == 'mse' else nn.BCEWithLogitsLoss()
        elif second_layer == 'mean':
            loss_fn = nn.MSELoss() if loss == 'mse' else  nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss()
            if loss == 'ce':
                second_out_size = 2
    elif activation == 'hard_sigmoid':
        activation_fn = nn.Hardsigmoid()
        loss_fn = nn.MSELoss() if loss == 'mse' else nn.BCEWithLogitsLoss()
    print(f"loss_fn: {loss_fn.__class__.__name__}")
    
    # set up model
    model = NeuralNet(input_dim=L ** dim, hidden_dim=K, activation=activation_fn, second_layer=second_layer, second_out_size=second_out_size, init_scale=init_scale)
    
    # set up optimizer
    # opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train
    losses, accs = np.zeros(num_epochs), np.zeros(num_epochs)
    # weights_ = model.ff1.data.detach().cpu().numpy().copy()
    weights_ = model.ff1.weight.detach().cpu().numpy().copy()
    weights = [ weights_ ]
    iprs = []
    every_epoch = min(max(num_epochs // 100, 1), 500) # 10 was 100
    iteration = 1
    
    for epoch, (X, y) in enumerate(loader):
        X, y = X.squeeze(0), y.squeeze(0)
        yhat = model(X) #- 5
        # yhat = F.relu(X @ model.ff1.weight.T + model.ff1.bias) @ model.ff2.weight + model.ff2.bias
        
        if second_out_size == 2:
            y = y.to(torch.long)
            # y = torch.tensor([[yi, 1-yi] for yi in y]).to(torch.float32)
        # try:
        #     loss_ = loss_fn(yhat, y)
        # # if loss_.item() < 0:
        # except:
        #     print(X[:10])
        #     print(y[:10])
        #     print(yhat[:10])
        #     raise RuntimeError
        # loss_ = loss_fn(yhat, y)
        # loss_.backward(retain_graph=True)
        # print('1'); yhat_ = X @ model.ff1.weight.T; loss_ = loss_fn(yhat_, y); print(loss_); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        # print('2'); yhat_ = X @ model.ff1.weight.T + model.ff1.bias; loss_ = loss_fn(yhat_, y); print(loss_); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        # print('3'); yhat_ = F.relu(X @ model.ff1.weight.T + model.ff1.bias); loss_ = loss_fn(yhat_, y); print(loss_); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        # print('4'); yhat_ = F.relu(X @ model.ff1.weight.T + model.ff1.bias) @ model.ff2.weight; print(loss_); loss_ = loss_fn(yhat_, y); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        # print('5'); yhat_ = F.relu(X @ model.ff1.weight.T + model.ff1.bias) @ model.ff2.weight + model.ff2.bias; print(loss_); loss_ = loss_fn(yhat_, y); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        loss_ = loss_fn(yhat, y)
        # #     raise ValueError(f"loss_={loss_} < 0")
        # if epoch == 0:
        #     print('init'); print(loss_); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        #     print('1'); yhat_ = X @ model.ff1.weight.T; loss_ = loss_fn(yhat_, y); print(loss_); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        #     print('2'); yhat_ = X @ model.ff1.weight.T + model.ff1.bias; loss_ = loss_fn(yhat_, y); print(loss_); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        #     print('3'); yhat_ = F.relu(X @ model.ff1.weight.T + model.ff1.bias); loss_ = loss_fn(yhat_, y); print(loss_); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        #     print('4'); yhat_ = F.relu(X @ model.ff1.weight.T + model.ff1.bias) @ model.ff2.weight; print(loss_); loss_ = loss_fn(yhat_, y); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        #     print('5'); yhat_ = F.relu(X @ model.ff1.weight.T + model.ff1.bias) @ model.ff2.weight + model.ff2.bias; print(loss_); loss_ = loss_fn(yhat_, y); loss_.backward(retain_graph=True); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        #     print('6'); yhat_ = model(X); loss_ = loss_fn(yhat_, y); print(loss_); loss_.backward(); print(model.ff1.weight.grad); print(model.ff1.bias.grad)
        #     ipdb.set_trace()
        opt.zero_grad()
        loss_.backward()
        opt.step()
        
        losses[epoch] = loss_.cpu().item()
        yhat_int = (yhat[:,1] > yhat[:,0]) if second_out_size == 2 else (yhat > 0.5)
        accs[epoch] = (yhat_int == (y > 0)).to(torch.float32).mean().cpu().item()
        
        # print progress, record IPR
        if iteration % every_epoch == 0 or iteration == num_epochs:
            # weights_ = model.ff1.data.detach().cpu().numpy().copy()
            weights_ = model.ff1.weight.detach().cpu().numpy().copy()
            weights.append( weights_ )
            ipr_ = ipr(weights_)
            iprs.append( ipr_ )
            print(f'Iteration {iteration}: loss={losses[max(epoch-every_epoch,0):epoch+1].mean():.4f}, acc={accs[max(epoch-every_epoch,0):epoch+1].mean():.4f}, IPR={ipr_.mean():.4f} ({ipr_.std():.4f})')
            
        iteration += 1
    
    # make ipr an array
    iprs = np.array(iprs)
    weights = np.stack(weights)
        
    # key
    if save_:
        # key = f'__xi1={xi1:05.2f}_xi2={xi2:05.2f}_gain={gain:05.2f}_L={L:03}_K={K:03}_dim={dim}_batch_size={batch_size}_num_epochs={num_epochs}_loss={loss}_lr={lr:.3f}_activation={activation}_second_layer={second_layer}'
        path_key = make_key(task, xi1, xi2, batch_size, num_epochs, loss, lr, second_layer, L, K, activation, init_scale, gain=gain)
            
        # save losses, accs, iprs
        np.savez(f'./weights/{path_key}.npz', weights=weights, losses=losses, accs=accs, iprs=iprs)
        print('Saved weights, losses, accs, iprs')
            
        # save model weights
        # torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{path}/../results/weights_{key}.pt')
        # print('Saved model weights')
        
    return weights, (losses, accs, iprs)
        
if __name__ == '__main__':
    
    kwargs = dict(
        task='nlgp',
        xi1 = 8,
        xi2 = 2,
        gain = 10, #0.05,
        L = 100,
        dim = 1,
        K = 40,
        batch_size = 1000,
        num_epochs = 1000,
        lr = 0.2,
        init_scale = 1.0,
        save_ = True,
        activation = 'relu',
        loss = 'ce',
        second_layer = 'mean',
    )
    
    # main    
    main(**kwargs)