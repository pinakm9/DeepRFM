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


class LocalRFM(nn.Module):
    def __init__(self, D, D_r, B, G, I):
        """
        Args:
            D: dimension of the data
            D_r: dimension of the feature 
            B: number of RF blocks
            G: number of groups
            I: number of neighboring groups to interact with
        """
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.G = G
        self.I = I
        self.Ng = int(self.D / self.G)
        self.idx = torch.arange(-self.I*self.G, (self.I+1)*self.G) % D
        self.idx = torch.vstack([(self.idx + self.G*i) % D for i in range(self.Ng)])
        self.idy = torch.arange(0, self.G)
        self.idy = torch.vstack([(self.idy + self.G*i) % D for i in range(self.Ng)])
        self.inner = nn.ModuleList([nn.Linear((2*self.I + 1)*self.G, self.D_r, bias=True)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.G, bias=False)])

    # @ut.timer
    def forward(self, x):
        """
        Forward pass for the LocalRFM model.

        This method processes the input tensor `x` through a series of neural network 
        layers defined by `self.inner` and `self.outer`. It concatenates specific slices 
        of the input tensor based on predefined indices, applies a transformation through 
        the inner and outer layers using the hyperbolic tangent activation function, and 
        returns the processed output.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the forward pass.

        Returns
        -------
        torch.Tensor
            The transformed output tensor, flattened across the last two dimensions.
        """
        return self.outer[0](torch.tanh(self.inner[0](x[..., self.idx]))).flatten(-2, -1)

    
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
        self.net = LocalRFM(self.sampler.dim, D_r, B, G, I)
        self.net.to(self.device)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        self.sampler.update((Uo.T[:-1][..., self.net.idx])[:, self.net.Ng//2, :].T)
        self.arch = self.net.__class__

    
    # @ut.timer
    def learn(self, train, seed):
        """
        Learns the parameters of the LocalRFM model using the provided training data.

        This function selects specific slices of the input data, samples weights and biases
        from the normal distribution, and updates the model's layers using the sampled parameters.

        Parameters
        ----------
        train : torch.Tensor
            The input training data tensor.
        seed : int
            A seed value for random number generation, ensuring reproducibility.

        Returns
        -------
        None
        """
        X = train.T[:-1][..., self.net.idx][:, self.net.Ng//2, :].T
        Y = train.T[1:][..., self.net.idy][:, self.net.Ng//2, :].T

        with torch.no_grad():
            Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
            self.net.inner[0].weight = nn.Parameter(Wb[:, :-1])
            self.net.inner[0].bias = nn.Parameter(Wb[:, -1])
            self.net.outer[0].weight = nn.Parameter(self.compute_W(Wb, X, Y))


     # @ut.timer
    def learn_(self, train, seed):
        """
        Learns the parameters of the LocalRFM model using the provided training data.

        This function selects specific slices of the input data, samples weights and biases
        from the normal distribution, and updates the model's layers using the sampled parameters.

        Parameters
        ----------
        train : torch.Tensor
            The input training data tensor.
        seed : int
            A seed value for random number generation, ensuring reproducibility.

        Returns
        -------
        X, Y : torch.Tensor
            The input and target data tensors used for training.
        """
        X = train.T[:-1][..., self.net.idx][:, self.net.Ng//2, :].T
        Y = train.T[1:][..., self.net.idy][:, self.net.Ng//2, :].T

        with torch.no_grad():
            Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
            self.net.inner[0].weight = nn.Parameter(Wb[:, :-1])
            self.net.inner[0].bias = nn.Parameter(Wb[:, -1])
            self.net.outer[0].weight = nn.Parameter(self.compute_W(Wb, X, Y))
        return X, Y
 


class BatchDeepRF(rfm.BatchDeepRF):
    def __init__(self, train, test, *drf_args):
        """
        Initializes a BatchDeepRF object for training and testing.

        Parameters
        ----------
        train : np.array
            Training data array.
        test : np.array
            Test data array.
        *drf_args : tuple
            Additional arguments to be passed for the DeepRF initialization.

        Returns
        -------
        None
        """
        super().__init__(DeepRF, train, test, *drf_args) 