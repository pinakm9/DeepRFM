import numpy as np
import matplotlib.pyplot as plt

# Fourier spectral method to solve the Kuramoto-Sivashinsky equation
def ks(N=128, L=32, tmax=100, dt=0.01):
    """
    Solves the Kuramoto-Sivashinsky equation using Fourier spectral methods.

    Args:
    - N: Number of grid points
    - L: Domain size (length of the periodic domain)
    - tmax: Maximum time for the simulation
    - dt: Time step size
    - plot_interval: Interval of time steps for plotting the solution
    """
    # Define spatial domain and wavenumbers
    x = np.linspace(0, L, N, endpoint=False)  # Spatial grid
    k = np.fft.fftfreq(N, L/N) * 2 * np.pi    # Wavenumbers
    k2 = k**2                                 # k^2 term for second derivative
    k4 = k**4                                 # k^4 term for fourth derivative
    
    # Initial condition (random)
    u = np.cos(2 * np.pi * x / L) * (1 + 0.1 * np.random.randn(N))

    # Fourier transform of the initial condition
    u_hat = np.fft.fft(u)

    # Precompute linear operator in Fourier space for efficiency
    L_hat = -k2 + k4  # Linear term in Fourier space

    # Define the function to compute the RHS of the KS equation in Fourier space
    def rhs_ks(u_hat):
        u = np.fft.ifft(u_hat).real               # Transform back to physical space
        nonlinear_term = -0.5j * k * np.fft.fft(u**2)  # u * u_x term in Fourier space
        return L_hat * u_hat + nonlinear_term     # Linear + Nonlinear terms

    # Time-stepping loop using RK4
    t, n = 0.0, 1
    uu = np.zeros((int(tmax/dt), N))
    uu[0] = u
    while t < tmax:
        # RK4 integration
        k1 = dt * rhs_ks(u_hat)
        k2 = dt * rhs_ks(u_hat + 0.5 * k1)
        k3 = dt * rhs_ks(u_hat + 0.5 * k2)
        k4 = dt * rhs_ks(u_hat + k3)
        u_hat = u_hat + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        uu[n] = np.fft.ifft(u_hat).real
        # Update time
        t += dt


