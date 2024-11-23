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
        Args:
            n_rows: number of rows to sample
        """
        torch.manual_seed(seed if seed is not None else torch.seed())
        result = Parallel(n_jobs=-1)(delayed(self.sample_)(sample_b) for _ in range(n_rows))
        return torch.vstack(result)
    
    # @ut.timer
    def sample(self, n_rows, sample_b=True, seed=None):
        """
        Args:
            n_rows: number of rows to sample
        """
        torch.manual_seed(seed if seed is not None else torch.seed())
        return torch.vstack([self.sample_(sample_b) for _ in range(n_rows)])
    
    def test_rows(self, rows):
        return torch.hstack([self.is_row(row) for row in rows])
    
    def is_row(self, row):
        if row[-1] < 0:
            row *= -1
        # find orthant
        s = ((torch.sign(row) + 1) / 2).int()
        return (self.x_minus(s) @ row[:-1] + row[-1] > self.L0) and (self.x_plus(s) @ row[:-1] + row[-1] < self.L1)
    
    def range_(self, row):
        y = self.Uo.T @ row[:-1] + row[-1]
        return torch.hstack((torch.min(y), torch.max(y)))
    
    @ut.timer
    def range(self, rows):
        return torch.vstack([self.range_(row) for row in rows])
    
    @ut.timer
    def range_parallel(self, rows):
        return torch.vstack(Parallel(n_jobs=-1)(delayed(self.range_)(row) for row in rows))
    
