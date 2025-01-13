# load necessary modules
import numpy as np 
from scipy.integrate import odeint
import os, sys 
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
from joblib import Parallel, delayed


# L63 system
def L63(u, alpha=10., rho=28., beta=8./3.):
    """
    The Lorenz 1963 (L63) system is a set of ordinary differential equations
    that model fluid convection in the atmosphere. 

    Parameters
    ----------
    u : array_like
        The system state, comprising three components: x, y, and z.
    alpha : float (default: 10)
        The Prandtl number.
    rho : float (default: 28)
        The Rayleigh number.
    beta : float (default: 8/3)
        The aspect ratio of the layer.

    Returns
    -------
    du/dt : array_like
        The time derivative of u.
    """
    x, y, z = u #np.split(u, 3, axis=-1)
    p = alpha * (y - x)
    q = (rho - z) * x - y
    r = x * y - beta * z
    return np.array([p, q, r])

# single trajectory generator for L63
def generate_trajectory(state0, dt, n_steps):
    """
    Generate a single trajectory of the L63 system.

    Parameters
    ----------
    state0 : array_like
        The initial state.
    dt : float
        The time step size.
    n_steps : int
        The number of time steps to integrate.

    Returns
    -------
    trajectory : array_like
        The trajectory of the system.
    """
    return odeint(lambda x, t: L63(x), state0, np.arange(0, n_steps*dt, dt))

# multiple trajectories generator for L63
# @ut.time
def generate_trajectories(num_trajectories, dt, n_steps):
    """
    Generate multiple trajectories of the L63 system.

    Parameters
    ----------
    num_trajectories : int
        The number of trajectories to generate.
    dt : float
        The time step size.
    n_steps : int
        The number of time steps to integrate.

    Returns
    -------
    trajectories : array_like
        The trajectories of the system, of shape (num_trajectories, 3, n_steps).
    """
    trajectories = np.zeros((num_trajectories, 3, n_steps))
    random_points =  np.random.normal(size=(num_trajectories, 3))
    generate = lambda *args: generate_trajectory(*args)[-1]
    states0 = Parallel(n_jobs=-1)(delayed(generate)(random_points[i], dt, int(40/dt)) for i in range(num_trajectories))
    results = Parallel(n_jobs=-1)(delayed(generate_trajectory)(state0, dt, n_steps) for state0 in states0)
    for i in range(num_trajectories):
        trajectories[i, :, :] = results[i].T 
    return trajectories

@ut.timer
def gen_data(dt=0.01, train_seed=22, train_size=int(1e5), test_seed=43, test_num=500, test_size=1000, save_folder=None):
    """
    Generate data for the Lorenz 63 system.

    Parameters
    ----------
    dt : float, optional
        The time step size. Defaults to 0.01.
    train_seed : int, optional
        The seed for the random number generator for the training data.
        Defaults to 22.
    train_size : int, optional
        The number of time steps to save for the training data.
        Defaults to 1e5.
    test_seed : int, optional
        The seed for the random number generator for the test data.
        Defaults to 43.
    test_num : int, optional
        The number of test cases to generate. Defaults to 500.
    test_size : int, optional
        The number of time steps to save for each test case.
        Defaults to 1000.
    save_folder : str, optional
        The folder to save the data to. If None, the data is not saved.
        Defaults to None.

    Returns
    -------
    train : array_like
        The training data.
    test : array_like
        The test data.
    """
    np.random.seed(train_seed)
    train = generate_trajectories(1, dt, train_size)[0]
    np.random.seed(test_seed)
    test = generate_trajectories(1, dt, test_num*test_size)
    test = np.moveaxis(test[0].T.reshape(test_num, -1, 3), 1, 2)
    np.random.shuffle(test)
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.save(f'{save_folder}/train.npy', train)
        np.save(f'{save_folder}/test.npy', test)
    return train, test