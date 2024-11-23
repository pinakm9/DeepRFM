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

class DeepRFM(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.inner = nn.ModuleList([nn.Linear(2*self.D, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.D, bias=False) for _ in range(B)])

    # @ut.timer  
    def forward(self, x):
        y = torch.hstack((x, x))
        for i in range(self.B):
            y[..., self.D:] = self.outer[i](torch.tanh(self.inner[i](y)))
        return y[..., self.D:]
    
class DeepRF(rfm.DeepRF):
    def __init__(self, D_r, B, L0, L1, Uo, beta, name='nn', save_folder='.', normalize=False, *args):
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
        self.net = DeepRFM(self.sampler.dim, D_r, B)
        self.net.to(self.device)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        self.sampler.update(torch.vstack((Uo, Uo)))
        self.arch = self.net.__class__
    

    # @ut.timer
    def learn(self, train, seed):
        Y = train[:, 1:]
        X1 = torch.vstack((train[:, :-1], train[:, :-1]))
        with torch.no_grad():
            for i in range(self.net.B):
                Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
                self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
                self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
                self.net.outer[i].weight = nn.Parameter(self.compute_W(Wb, X1, Y))
                X1[self.net.D:] =  self.net.outer[i](torch.tanh(self.net.inner[i](X1.T))).T



class BatchDeepRF(rfm.BatchDeepRF):
    def __init__(self, train, test, *drf_args):
        super().__init__(DeepRF, train, test, *drf_args) 

