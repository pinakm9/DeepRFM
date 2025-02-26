import numpy as np
import torch
import utility as ut
from joblib import Parallel, delayed


class GoodRowSampler:
    """
    One shot hit and run sampler for good rows
    """
    def __init__(self, L0, L1, Uo):
        """
        Args:
            L0: left limit of tanh input for defining good rows
            L1: right limit tanh input for defining good rows
            Uo: training data
        """
        self.L0 = L0
        self.L1 = L1 
        self.Uo = Uo
        self.dim = Uo.T.shape[-1]
        self.device = Uo.device
        self.lims = torch.stack((torch.min(Uo, dim=1)[0], torch.max(Uo, dim=1)[0]), dim=1)
   

    
    def update(self, Uo):
        """
        Updates the sampler with new training data.

        This method updates the internal state of the sampler with the new
        training data tensor, recalculating its dimensionality and limits.

        Args:
            Uo: torch.Tensor
                The new training data to update the sampler with.
        """
        self.Uo = Uo
        self.dim = Uo.T.shape[-1]
        self.lims = torch.stack((torch.min(Uo, dim=1)[0], torch.max(Uo, dim=1)[0]), dim=1)


    def x_plus(self, s):
        """
        Args:
            s: sign vector
        """
        return torch.tensor([self.lims[d, s[d]] for d in range(self.dim)], device=self.device)
    
    
    def x_minus(self, s):
        """
        Args:
            s: sign vector
        """
        return torch.tensor([self.lims[d, (1+s[d]) % 2] for d in range(self.dim)], device=self.device)
    
    def x_plus_vec(self, s):
        """
        Args:
            s: sign tensor
        """
        return torch.vstack([self.lims[d][s[:, d]] for d in range(self.dim)]).T

    def x_minus_vec(self, s):
            """
            Args:
                s: sign tensor
            """
            return torch.vstack([self.lims[d][(1 + s[:, d]) % 2] for d in range(self.dim)]).T
    
    # @ut.timer
    def sample_vec(self, n_rows, seed=None):
        """
        Samples multiple rows using the one-shot hit and run algorithm in parallel.
        Args:
            n_rows (int): The number of points to sample.
            seed (int): An optional seed for the random number generator.
        Returns:
            A tensor of shape (n_rows, dim+1) with the sampled points.
        """
        torch.manual_seed(seed if seed is not None else torch.seed())
        # choose bias
        b = (self.L1 - self.L0) * torch.rand(n_rows, device=self.device) + self.L0
        # # choose an orthant
        s = torch.randint(2, size=(n_rows, self.dim), device=self.device)
        s_ = s.clone()
        s_[s_==0] = -1
        # # choose a direction in the orthant
        d = torch.abs(torch.randn(size=(n_rows, self.dim), device=self.device)) * s_
        d /= torch.linalg.norm(d, dim=1, keepdim=True)

        # # determine line segment
        d_plus = torch.sum(d * self.x_plus_vec(s), axis=1)
        d_minus = torch.sum(d * self.x_minus_vec(s), axis=1)
        q = (self.L0 - b) / d_minus
        p = (self.L1 - b) / d_plus


        # # pick a point on the line
        dpg = d_plus > 0
        dpl = d_plus < 0
        dmg = d_minus > 0
        dml = d_minus < 0

        a1 = b.clone()

        idx = torch.logical_and(dpg, dml)
        a1[idx] = torch.min(torch.vstack((p[idx], q[idx])).T, axis=1)[0]

        idx = torch.logical_and(dpg, dmg)
        a1[idx] = p[idx]

        idx = torch.logical_and(dpl, dml)
        a1[idx] = q[idx]

        idx = torch.logical_and(dpl, dmg)
        a1[idx] = 1e6*torch.ones(idx.sum(), device=self.device)            

        a = a1 * torch.rand(a1.shape, device=self.device)
        subset_flag = torch.randint(2, size=a.shape, device=self.device).to(torch.double) 
        subset_flag[subset_flag==0] = -1.  

        return torch.concat([a[:, None]*d, b.reshape(-1, 1)], dim=1) * subset_flag[:, None]

    

    def sample_(self, sample_b=True):
        """
        Samples a single row using the one-shot hit-and-run algorithm.
        Args:
            sample_b (bool): Whether to sample the bias or not.
        Returns:
            A tensor of shape (dim+1,) with the sampled point and bias.
        """
        # choose bias
        b = (self.L1 - self.L0) * torch.rand(1, device=self.device)[0] + self.L0
        # choose an orthant
        s = torch.randint(2, size=(self.dim,), device=self.device)
        # choose a direction in the orthant
        d = torch.abs(torch.randn(size=(self.dim,), device=self.device)) * torch.tensor([1 if e else -1 for e in s], device=self.device)
        d /= torch.linalg.norm(d)
        # print("Direction is ", d)
        # determine line segment
        d_plus = d @ self.x_plus(s)
        d_minus = d @ self.x_minus(s)
        q = (self.L0 - b) / d_minus
        p = (self.L1 - b) / d_plus


        if d_plus > 0:
            if d_minus < 0:
                a1 = min((p, q)) 
            else:
                a1 = p 
        else:
            if d_minus < 0:
                a1 = q
            else:
                a1 = 1e6             

        # pick a point on the line 
        a = a1 * torch.rand(1, device=self.device)[0]
        subset_flag = torch.randint(2, size=(1,))[0]
        if sample_b:
            wb = torch.hstack([a*d, b])
            # decide which subset of the solution set we want to sample
            if subset_flag == 1:
                return wb
            else:
                return -wb
        else:
            if subset_flag == 1:
                return a*d
            else:
                return -a*d
    
    @ut.timer
    def sample_parallel(self, n_rows, sample_b=True, seed=None):
        """
        Samples a specified number of rows using the `sample_` method in parallel.

        This function does the same thing as `sample`, but it uses joblib's
        `Parallel` to do the computation in parallel. This can be much faster
        when sampling a large number of rows.

        Args:
            n_rows (int): The number of rows to sample.
            sample_b (bool, optional): If True, sample the bias term as well.
                Defaults to True.
            seed (int, optional): A seed for the random number generator.
                If None, the current seed is used. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (n_rows, dim+1) with the sampled points.
        """
        torch.manual_seed(seed if seed is not None else torch.seed())
        result = Parallel(n_jobs=-1)(delayed(self.sample_)(sample_b) for _ in range(n_rows))
        return torch.vstack(result)
    
    # @ut.timer
    def sample(self, n_rows, sample_b=True, seed=None):
        """
        Samples a specified number of rows using the `sample_` method.

        Args:
            n_rows (int): The number of rows to sample.
            sample_b (bool, optional): A flag indicating whether to sample with the additional component `b`.
                Defaults to True.
            seed (int, optional): A seed for random number generation to ensure reproducibility.
                If None, a random seed will be used.

        Returns:
            torch.Tensor: A tensor containing the sampled rows.
        """
        torch.manual_seed(seed if seed is not None else torch.seed())
        return torch.vstack([self.sample_(sample_b) for _ in range(n_rows)])
    
    def test_rows(self, rows):
        """
        Tests whether each row in the input tensor is a "good row" according to the criteria defined by the `is_row` method.

        Args:
            rows (torch.Tensor): A tensor containing multiple rows to be tested.

        Returns:
            torch.Tensor: A 1D tensor of boolean values where each element indicates whether the corresponding row is a "good row".
        """
        return torch.hstack([self.is_row(row) for row in rows])
    
    def is_row(self, row):
        """
        Determines if a given row satisfies the "good row" criteria.

        This method checks whether the provided row, after ensuring its last element is non-negative,
        lies within the bounds defined by L0 and L1 when projected into the hypercube's orthant.

        Args:
            row (torch.Tensor): A tensor representing a single row, where the last element is the bias term 
                                and the preceding elements are the coordinates in the hypercube.

        Returns:
            bool: True if the row is considered a "good row" according to the defined criteria, False otherwise.
        """
        if row[-1] < 0:
            row *= -1
        # find orthant
        s = ((torch.sign(row) + 1) / 2).int()
        return (self.x_minus(s) @ row[:-1] + row[-1] > self.L0) and (self.x_plus(s) @ row[:-1] + row[-1] < self.L1)
    
    def range_(self, row):
        """
        Computes the range of values in the direction of the given row.

        Args:
            row (torch.Tensor): A tensor representing a single row, where the last element is the bias term 
                                and the preceding elements are the coordinates in the hypercube.

        Returns:
            torch.Tensor: A tensor of shape (2,) containing the minimum and maximum values.
        """
        y = self.Uo.T @ row[:-1] + row[-1]
        return torch.hstack((torch.min(y), torch.max(y)))
    
    @ut.timer
    def range(self, rows):
        """
        Computes the range of values for each row in the input tensor.

        Args:
            rows (torch.Tensor): A tensor containing multiple rows.

        Returns:
            torch.Tensor: A tensor of shape (n_rows, 2) containing the minimum and maximum values for each row.
        """
        return torch.vstack([self.range_(row) for row in rows])
    
    @ut.timer
    def range_parallel(self, rows):
        """
        Computes the range of values for each row in the input tensor, using joblib for parallelization.

        Args:
            rows (torch.Tensor): A tensor containing multiple rows.

        Returns:
            torch.Tensor: A tensor of shape (n_rows, 2) containing the minimum and maximum values for each row.
        """
        return torch.vstack(Parallel(n_jobs=-1)(delayed(self.range_)(row) for row in rows))
    
