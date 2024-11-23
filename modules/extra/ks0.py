import numpy as np
import utility as ut
from scipy.fftpack import fft, ifft
import os

def ks(L=16, N=128, dt=0.25, tmax=150):
    # kursiv.py - solution of Kuramoto-Sivashinsky equation by ETDRK4 scheme
    #
    # u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,2*pi*L]
    # computation is based on v = fft(u), so linear term is diagonal
    # compare p27.m in Trefethen, "Spectral Methods in MATLAB", SIAM 2000
    # AK Kassam and LN Trefethen, July 2002
    # Spatial grid and initial condition:
    x = 2 * np.pi * L * np.arange(0, N) / N
    u = np.cos(x / L) * (1 + np.sin(x / L))
    v = fft(u)

    # Precompute various ETDRK4 scalar quantities:
    h = dt  # time step
    k = np.concatenate((np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0))) / L  # wave numbers
    l = k**2 - k**4  # Fourier multipliers
    E = np.exp(h * l)
    E2 = np.exp(h * l / 2)
    M = 16 # no. of points for complex means
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # roots of unity
    LR = h * l[:, np.newaxis] + r[np.newaxis, :]  # broadcasting
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    # Main time-stepping loop:
    nmax = round(tmax / h)
    uu = np.zeros((N, nmax+1))
    uu[:, 0] = u
    g = -0.5j * k

    for n in range(1, nmax + 1):
        #t = n * h
        Nv = g * fft(np.real(ifft(v))**2)
        a = E2 * v + Q * Nv
        Na = g * fft(np.real(ifft(a))**2)
        b = E2 * v + Q * Na
        Nb = g * fft(np.real(ifft(b))**2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = g * fft(np.real(ifft(c))**2)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        
        u = np.real(ifft(v))
        uu[:, n] = u
          
    return uu


class KS32:
    @ut.timer
    def __init__(self, dt=0.25, N=128, tmax=150) -> None:
        # kursiv.py - solution of Kuramoto-Sivashinsky equation by ETDRK4 scheme
        #
        # u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,32*pi]
        # computation is based on v = fft(u), so linear term is diagonal
        # compare p27.m in Trefethen, "Spectral Methods in MATLAB", SIAM 2000
        # AK Kassam and LN Trefethen, July 2002
        # Spatial grid and initial condition:
        self.tmax = tmax
        self.N = N
        self.x = 32 * np.pi * np.arange(1, self.N + 1) / self.N
        self.u = np.cos(self.x / 16) * (1 + np.sin(self.x / 16))
        self.h = dt  # time step
        

        # Precompute various ETDRK4 scalar quantities:
        self.v = fft(self.u)
        self.k = np.concatenate((np.arange(0, self.N // 2), [0], np.arange(-self.N // 2 + 1, 0))) / 32  # wave numbers
        self.L = self.k**2 - self.k**4  # Fourier multipliers
        self.E = np.exp(self.h * self.L)
        self.E2 = np.exp(self.h * self.L / 2)
        self.M = 16  # no. of points for complex means
        r = np.exp(1j * np.pi * (np.arange(1, self.M + 1) - 0.5) / self.M)  # roots of unity
        self.LR = self.h * self.L[:, np.newaxis] + r[np.newaxis, :]  # broadcasting
        self.Q = self.h * np.real(np.mean((np.exp(self.LR / 2) - 1) / self.LR, axis=1))
        self.f1 = self.h * np.real(np.mean((-4 - self.LR + np.exp(self.LR) * (4 - 3 * self.LR + self.LR**2)) / self.LR**3, axis=1))
        self.f2 = self.h * np.real(np.mean((2 + self.LR + np.exp(self.LR) * (-2 + self.LR)) / self.LR**3, axis=1))
        self.f3 = self.h * np.real(np.mean((-4 - 3 * self.LR - self.LR**2 + np.exp(self.LR) * (4 - self.LR)) / self.LR**3, axis=1))

        # Main time-stepping loop:
        self.nmax = round(self.tmax / self.h)
        self.uu = np.zeros((self.N, self.nmax+1))
        self.uu[:, 0] = self.u
        self.g = -0.5j * self.k
        n = 1
        o = 0

        while n < self.nmax + 1:
            try:
                self.Nv = self.g * fft(np.real(ifft(self.v))**2)
                self.a = self.E2 * self.v + self.Q * self.Nv
                self.Na = self.g * fft(np.real(ifft(self.a))**2)
                self.b = self.E2 * self.v + self.Q * self.Na
                self.Nb = self.g * fft(np.real(ifft(self.b))**2)
                self.c = self.E2 * self.a + self.Q * (2 * self.Nb - self.Nv)
                self.Nc = self.g * fft(np.real(ifft(self.c))**2)
                self.v = self.E * self.v + self.Nv * self.f1 + 2 * (self.Na + self.Nb) * self.f2 + self.Nc * self.f3
                self.u = np.real(ifft(self.v))
                self.uu[:, n] = self.u
                n += 1
            except:
                o += 1
                print('overflow', o, end='\r')
                n -= 10
                self.v = fft(self.uu[:, n-1])




@ut.timer
def gen_data(dt=0.25, train_seed=22, train_size=int(2e5), test_seed=43, test_num=100, test_size=1000, save_folder=None):
    #------------------------------------------------------------------------------
    # define data and initialize simulation
    N = 128
    ninittransients = 40000
    tmax = dt * (train_size + test_size*test_num + ninittransients)  
    dns  = ks32(dt=dt, N=N, tmax=tmax)
    #------------------------------------------------------------------------------
    # simulate initial transient
    dns.simulate()
    # convert to physical space
    dns.fou2real()
    u = dns.uu[ninittransients+1:]
    train = u[:train_size].T
    test = np.moveaxis(u[train_size:].reshape(test_num, -1, N), 1, 2)
    np.random.shuffle(test)
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.save(f'{save_folder}/train.npy', train)
        np.save(f'{save_folder}/test.npy', test)
    return train, test

