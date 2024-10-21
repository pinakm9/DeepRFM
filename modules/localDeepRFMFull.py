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


class LocalDeepRFMFull(nn.Module):
    def __init__(self, D, D_r, B, G, I):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.G = G
        self.I = I 
        self.Ng = int(self.D / self.G)
        self.idx = torch.arange(-self.I*self.G, (self.I+1)*self.G) % D
        self.idx = torch.vstack([(self.idx + i*self.G) % D for i in range(self.Ng)])
        self.idy = torch.arange(0, self.G)
        self.idy = torch.vstack([(self.idy + i*self.G) % D for i in range(int(self.Ng))])
        self.p = (2*self.I + 1)*self.G
        self.inner = nn.ModuleList([nn.Linear(2*(2*self.I + 1)*self.G, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.G, bias=False) for _ in range(B)])

    # @ut.timer  
    def forward(self, x):
        z = x + 0.
        for i in range(self.B):
            y = torch.concat((x[..., self.idx], z[..., self.idx]), dim=-1)
            z = self.outer[i](torch.tanh(self.inner[i](y)))
            z = z.flatten(-2, -1)
        return z 
    
class DeepRF(rfm.DeepRF):
    def __init__(self, D_r, B, L0, L1, Uo, beta, name='nn', save_folder='.', normalize=False, G=2, I=2, *args):
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
        self.net = LocalDeepRFMFull(self.sampler.dim, D_r, B, G, I)
        self.net.to(self.device)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        x = (Uo.T[..., self.net.idx]).flatten(0, 1).T
        self.sampler.update(torch.vstack((x, x)))
        self.arch = self.net.__class__


    # @ut.timer
    def learn(self, train, seed):
        x = train.T[:-1][..., self.net.idx]
        z = train.T[:-1]
        Y = train.T[1:][..., self.net.idy][:, self.net.Ng//2, :].T
        XZ = torch.vstack((x[:, self.net.Ng//2, :].T, x[:, self.net.Ng//2, :].T))
        print(XZ.shape, Y.shape)

        with torch.no_grad():
            for i in range(self.net.B):
                Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
                self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
                self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
                self.net.outer[i].weight = nn.Parameter(self.compute_W(Wb, XZ, Y))
                y = torch.concat((x, z[..., self.net.idx]), dim=-1)
                print(y.shape)
                z = self.net.outer[i](torch.tanh(self.net.inner[i](y))).flatten(-2, -1)
                print(z.shape)
                XZ[:self.net.p] = z[..., self.net.idx][:, self.net.Ng//2, :].T


    def learn_(self, train, seed):

        idx = torch.arange(-self.net.I, self.net.I+1) + self.net.Ng//2

        x = train.T[:-1][..., self.net.idx]
        z = train.T[:-1]
        Y = train.T[1:][..., self.net.idy][:, self.net.Ng//2, :].T
        XZ = torch.vstack((x[:, self.net.Ng//2, :].T, x[:, self.net.Ng//2, :].T))

        x0 = x[:, idx, :]
        y = torch.concat((x0, x0), dim=-1)
        print(XZ.shape, Y.shape)
        print(x0.shape, y.shape)

        with torch.no_grad():
            for i in range(self.net.B):
                Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
                self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
                self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
                self.net.outer[i].weight = nn.Parameter(self.compute_W(Wb, XZ, Y))
         
                z = self.net.outer[i](torch.tanh(self.net.inner[i](y))).flatten(-2, -1)
                print(z.shape)
                XZ[:self.net.p] = z.T
                y = torch.concat((x0, z), dim=-1)



class BatchDeepRF(rfm.BatchDeepRF):
    def __init__(self, train, test, *drf_args):
        super().__init__(DeepRF, train, test, *drf_args) 