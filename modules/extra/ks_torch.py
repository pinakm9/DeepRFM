from torch.fft import fft, ifft
import numpy as np
import utility as ut
import os
import torch

torch.set_default_dtype(torch.float64)

class KS:
    #
    # Solution of the 1D Kuramoto-Sivashinsky equation
    #
    # u_t + u*u_x + u_xx + u_xxxx = 0,
    # with periodic BCs on x \in [0, 2*pi*L]: u(x+2*pi*L,t) = u(x,t).
    #
    # The nature of the solution depends on the system size L and on the initial
    # condition u(x,0).  Energy enters the system at long wavelengths via u_xx
    # (an unstable diffusion term), cascades to short wavelengths due to the
    # nonlinearity u*u_x, and dissipates via diffusion with u_xxxx.
    #
    # Spatial  discretization: spectral (Fourier)
    # Temporal discretization: exponential time differencing fourth-order Runge-Kutta
    # see AK Kassam and LN Trefethen, SISC 2005

    def __init__(self, L=16, N=128, dt=0.25, nsteps=None, tend=150, iout=1, device='cpu'):
        #
        # Initialize
        # L  = float(L); dt = float(dt); tend = float(tend)
        if (nsteps is None):
            nsteps = int(tend/dt)
        else:
            nsteps = int(nsteps)
            # override tend
            tend = dt*nsteps
        #
        # save to self
        self.L      = L
        self.N      = N
        self.dx     = 2*torch.pi*L/N
        self.dt     = dt
        self.nsteps = nsteps
        self.iout   = iout
        self.nout   = int(nsteps/iout)
        self.device = device
        # precompute Fourier-related quantities
        self.x  = torch.tensor(2*torch.pi*self.L*np.r_[0:self.N]/self.N, device=self.device)
        self.k  = torch.tensor(np.r_[0:self.N/2, 0, -self.N/2+1:0]/self.L, device=self.device) # Wave numbers
        # Fourier multipliers for the linear term Lu
        self.l = self.k**2 - self.k**4
        
        # set initial condition
        u0 = (torch.cos(self.x / 16) * (1 + torch.sin(self.x / 16))).to(self.device)
        v0 = fft(u0)
        # and save to self
        self.v   = v0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
        #
        # initialize simulation arrays
        # nout+1 so we store the IC as well
        self.vv = torch.zeros((self.nout+1, self.N), device=self.device)
        self.uu = torch.zeros((self.N, self.nout+1), device=self.device)
        self.tt = torch.zeros(self.nout+1, device=self.device)
        # store the IC in [0]
        self.vv[0] = v0
        self.uu[:, 0] = u0
        self.tt[0] = 0.
        #
        # precompute ETDRK4 scalar quantities:
        self.E  = torch.exp(self.dt*self.l)
        self.E2 = torch.exp(self.dt*self.l/2.)
        self.M  = 16                                           # no. of points for complex means
        self.r  = torch.exp(1j*torch.pi*(torch.tensor(np.r_[1:self.M+1]-0.5/self.M, device=self.device))) # roots of unity
        self.LR = self.dt*torch.repeat_interleave(self.l[:,None], self.M, axis=1) + torch.repeat_interleave(self.r[None,:], self.N, axis=0)
        self.Q  = self.dt*torch.real(torch.mean((torch.exp(self.LR/2.) - 1.)/self.LR, 1))
        self.f1 = self.dt*torch.real(torch.mean( (-4. -    self.LR              + torch.exp(self.LR)*( 4. - 3.*self.LR + self.LR**2) )/(self.LR**3) , 1) )
        self.f2 = self.dt*torch.real(torch.mean( ( 2. +    self.LR              + torch.exp(self.LR)*(-2. +    self.LR             ) )/(self.LR**3) , 1) )
        self.f3 = self.dt*torch.real(torch.mean( (-4. - 3.*self.LR - self.LR**2 + torch.exp(self.LR)*( 4. -    self.LR             ) )/(self.LR**3) , 1) )
        self.g  = -0.5j*self.k
        
        

    def step(self):
        #
        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)
        #
        v = self.v;                           Nv = self.g*fft(torch.real(ifft(v))**2)
        a = self.E2*v + self.Q*Nv;            Na = self.g*fft(torch.real(ifft(a))**2)
        b = self.E2*v + self.Q*Na;            Nb = self.g*fft(torch.real(ifft(b))**2)
        c = self.E2*a + self.Q*(2.*Nb - Nv);  Nc = self.g*fft(torch.real(ifft(c))**2)
        #
        self.v = self.E*v + Nv*self.f1 + 2.*(Na + Nb)*self.f2 + Nc*self.f3
        self.stepnum += 1
        self.t       += self.dt

    # @ut.timer
    def simulate(self):
        o = 0
        for n in range(1, self.nsteps+1):
            try:
                self.step()
            except FloatingPointError:
                #
                o += 1
                print('overflow', o, end='\r')
                # cut time series to last saved solution and return
                self.nout = self.ioutnum
                return -1
            if ( (self.iout>0) and (n%self.iout==0) ):
                self.ioutnum += 1
                self.vv[self.ioutnum, :] = self.v
                self.uu[:, self.ioutnum] = torch.real(ifft(self.v))
                self.tt[self.ioutnum]    = self.t




@ut.timer
def gen_data(dt=0.25, train_seed=22, train_size=int(2e5), test_seed=43, test_num=500, test_size=1000, save_folder=None):
    L    = 200/(2*np.pi)
    N    = 512
    ninittransients = 40000
    tend = dt * (train_size + test_size*test_num + ninittransients)  
    dns  = KS(L=L, N=N, dt=dt, tend=tend)
    dns.simulate()
    u = dns.uu[:, ninittransients+1:]
    train = u[:, :train_size]
    test = torch.moveaxis(u.T[train_size:].reshape(test_num, -1, N), 1, 2)
    indices = torch.randperm(test.shape[0])
    test = test[indices]
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(f'{save_folder}/train.torchy', train.cpu().numpy())
        torch.save(f'{save_folder}/test.torchy', test.cpu().numpy())
    return train, test