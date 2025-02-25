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


class SkipRFM(nn.Module):
    def __init__(self, D, D_r, B):
        """
        Initialize the SkipRFM model.

        Parameters
        ----------
        D : int
            Input dimensionality.
        D_r : int
            Number of random features.
        B : int
            Number of random maps (always set to 1).
        """
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = 1
        self.inner = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(1)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.D, bias=False) for _ in range(1)])

    # @ut.timer  
    def forward(self, x):
        """
        Applies the SkipRFM model to input data and returns the transformed output.

        Args:
            x (torch.Tensor): Input tensor with shape (D,), where D is the input dimensionality.

        Returns:
            torch.Tensor: Output tensor with the same dimensionality as the input, representing
                        the result of the forward pass through the model, modified by a skip connection.
        """
        return x + self.outer[0](torch.tanh(self.inner[0](x)))
    

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
        super().__init__(D_r, 1, L0, L1, Uo, beta, name, save_folder, normalize)
        self.net = SkipRFM(self.sampler.dim, D_r, B)
        self.net.to(self.device)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        self.arch = self.net.__class__

    
    # @ut.timer
    def learn(self, train, seed):
        """
        Learns the parameters of the DeepRF model.

        Parameters
        ----------
        train : torch.Tensor
            The training data.
        seed : int
            The seed for the random number generator.

        Returns
        -------
        None
        """
        Y = train[:, 1:]-train[:, :-1]
        with torch.no_grad():
            Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
            self.net.inner[0].weight = nn.Parameter(Wb[:, :-1])
            self.net.inner[0].bias = nn.Parameter(Wb[:, -1])
            self.net.outer[0].weight = nn.Parameter(self.compute_W(Wb, train[:, :-1], Y))

 

            
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
            Additional arguments to be passed to the DeepRF initialization.

        Returns
        -------
        None
        """
        super().__init__(DeepRF, train, test, *drf_args) 


     

