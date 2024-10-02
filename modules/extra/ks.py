from numpy import pi
from scipy.fftpack import fft, ifft
import numpy as np
import utility as ut
import os

np.seterr(over='raise', invalid='raise')

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

    def __init__(self, L=16, N=128, dt=0.25, nsteps=None, tend=150, iout=1):
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
        self.dx     = 2*pi*L/N
        self.dt     = dt
        self.nsteps = nsteps
        self.iout   = iout
        self.nout   = int(nsteps/iout)
        # precompute Fourier-related quantities
        self.x  = 2*pi*self.L*np.r_[0:self.N]/self.N
        self.k  = np.r_[0:self.N/2, 0, -self.N/2+1:0]/self.L # Wave numbers
        # Fourier multipliers for the linear term Lu
        self.l = self.k**2 - self.k**4
        
        # set initial condition
        u0 = np.cos(self.x / L) * (1 + np.sin(self.x / L))
        v0 = fft(u0)
        # and save to self
        self.v   = v0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
        #
        # initialize simulation arrays
        # nout+1 so we store the IC as well
        self.vv = np.zeros((self.nout+1, self.N))
        self.uu = np.zeros((self.N, self.nout+1))
        self.tt = np.zeros(self.nout+1)
        # store the IC in [0]
        self.vv[0] = v0
        self.uu[:, 0] = u0
        self.tt[0] = 0.
        #
        # precompute ETDRK4 scalar quantities:
        self.E  = np.exp(self.dt*self.l)
        self.E2 = np.exp(self.dt*self.l/2.)
        self.M  = 16                                           # no. of points for complex means
        self.r  = np.exp(1j*pi*(np.r_[1:self.M+1]-0.5)/self.M) # roots of unity
        self.LR = self.dt*np.repeat(self.l[:,np.newaxis], self.M, axis=1) + np.repeat(self.r[np.newaxis,:], self.N, axis=0)
        self.Q  = self.dt*np.real(np.mean((np.exp(self.LR/2.) - 1.)/self.LR, 1))
        self.f1 = self.dt*np.real( np.mean( (-4. -    self.LR              + np.exp(self.LR)*( 4. - 3.*self.LR + self.LR**2) )/(self.LR**3) , 1) )
        self.f2 = self.dt*np.real( np.mean( ( 2. +    self.LR              + np.exp(self.LR)*(-2. +    self.LR             ) )/(self.LR**3) , 1) )
        self.f3 = self.dt*np.real( np.mean( (-4. - 3.*self.LR - self.LR**2 + np.exp(self.LR)*( 4. -    self.LR             ) )/(self.LR**3) , 1) )
        self.g  = -0.5j*self.k
        

    def step(self):
        #
        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)
        #
        print(f"step = {self.stepnum}", end='\r')
        v = self.v;                           Nv = self.g*fft(np.real(ifft(v))**2)
        a = self.E2*v + self.Q*Nv;            Na = self.g*fft(np.real(ifft(a))**2)
        b = self.E2*v + self.Q*Na;            Nb = self.g*fft(np.real(ifft(b))**2)
        c = self.E2*a + self.Q*(2.*Nb - Nv);  Nc = self.g*fft(np.real(ifft(c))**2)
        #
        self.v = self.E*v + Nv*self.f1 + 2.*(Na + Nb)*self.f2 + Nc*self.f3
        self.stepnum += 1
        self.t       += self.dt

    @ut.timer
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
                self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
                self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
                return -1
            if ( (self.iout>0) and (n%self.iout==0) ):
                self.ioutnum += 1
                self.vv[self.ioutnum, :] = self.v
                self.uu[:, self.ioutnum] = np.real(ifft(self.v))
                self.tt[self.ioutnum]    = self.t




@ut.timer
def gen_data(dt=0.001, train_seed=22, train_size=int(2e5), test_seed=43, test_num=100, test_size=1000, save_folder=None,\
              L=200/(2*np.pi), N=512, ninittransients = 40000, iout=250):
    #------------------------------------------------------------------------------
    # define data and initialize simulation
    nsteps = (train_size + test_size*test_num + ninittransients)*iout  
    dns  = KS(L=L, N=N, dt=dt, nsteps=nsteps, iout=iout)
    dns.simulate()
    u = dns.uu[:, ninittransients+1:]
    train = u[:, :train_size]
    test = np.moveaxis(u.T[train_size:].reshape(test_num, -1, N), 1, 2)
    # np.random.shuffle(test)
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.save(f'{save_folder}/train.npy', train)
        np.save(f'{save_folder}/test.npy', test)
    return train, test