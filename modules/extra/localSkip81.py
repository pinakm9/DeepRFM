# load necessary modules
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
import torch
from torch import nn
import rfm


class LocalSkip81(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.G = 8 
        self.I = 1
        self.Ng = int(self.D / self.G)
        self.idx = torch.arange(-self.I*self.G, (self.I+1)*self.G) % D
        self.idx = torch.vstack([(self.idx + self.G*i) % D for i in range(self.Ng)])
        self.idy = torch.arange(0, self.G)
        self.idy = torch.vstack([(self.idy + self.G*i) % D for i in range(self.Ng)])
        self.inner = nn.ModuleList([nn.Linear((2*self.I + 1)*self.G, self.D_r, bias=True) for i in range(self.Ng)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.G, bias=False) for i in range(self.Ng)])

    # @ut.timer
    def forward(self, x):
        return torch.hstack([x[..., self.idy[i]] + self.outer[i](torch.tanh(self.inner[i](x[..., self.idx[i]]))) for i in range(self.Ng)])

    
class DeepRF(rfm.DeepRF):
    def __init__(self, D_r, B, L0, L1, Uo, beta, name='nn', save_folder='.', normalize=False):
        """
        Args:
            D_r: dimension of the feature 
            B: number of RF blocks
            name: name of the DeepRF
            L0: left limit of tanh input for defining good rows
            L1: right limit tanh input for defining good rows
            Uo: training data
            beta: regularization parameter
        """        
        super().__init__(D_r, B, L0, L1, Uo, beta, name, save_folder, normalize)
        self.net = LocalSkip81(self.sampler.dim, D_r, B)
        self.net.to(self.device)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        self.sampler.update(Uo.T[..., self.net.idx].flatten(0, 1).T)
        self.arch = self.net.__class__

    
    def learn(self, train, seed):
        with torch.no_grad():
            for i in range(self.net.Ng):
                print(f"learning block {i}", end='\r')
                Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
                self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
                self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
                X, Y = train[self.net.idx[i]][:, :-1], train[self.net.idy[i]][:, 1:] - train[self.net.idy[i]][:, :-1]
                self.net.outer[i].weight = nn.Parameter(self.compute_W(Wb, X, Y))   


class BatchDeepRF(rfm.BatchDeepRF):
    def __init__(self, train, test, *drf_args):
        super().__init__(DeepRF, train, test, *drf_args) 

class BetaTester(rfm.BetaTester):
    def __init__(self, D_r_list: list, B_list: list, train_list: list, test, *drf_args):
        super().__init__(DeepRF, D_r_list, B_list, train_list, test, *drf_args) 