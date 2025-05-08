import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from scipy.interpolate import interp1d

from taskspec.utils import find_project_root
datapath = find_project_root() + "/examples/Eg3"

x_range = (-3., 3.)
y_range = (-3., 3.)
z_range = (-3., 3.)
N_sample = 100000
N_samplet = 1000000

scale = [x_range[1], y_range[1], z_range[1]]

n_dim = 3

a0 = 0.3
def energy(xyz, a=a0):
    assert xyz.shape[1] == 3
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    x2 = np.square(x)
    y2 = np.square(y)
    E = np.square(x2-1) + 0.25*np.square(y2-1) + x2 * y2 + 0.5 * np.square(z) + \
        a * (1/3 * x**3 - x + 1/3 * y**3 + y * (1-x2) )
    return E

def ngrad(xyz, a=a0):
    assert xyz.shape[1] == 3
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    x2 = np.square(x)
    y2 = np.square(y)
    dxdt = 4 * x * (x2 - 1) + 2 * x * y2 + a * (x2 - 1 - 2 * x * y)
    dydt = y * (y2 - 1) + 2 * x2 * y + a * (y2 + 1 - x2)
    dzdt = z
    return -np.stack([dxdt, dydt, dzdt], axis=1)

critical_points = np.array([[1,0,0], [-1,0,0], [-0.536709, -0.966876,0]])

Nnode = 100
xends = np.array([[-1., 0.5, 0.0], [1., 0.5, 0.0]])
xend = np.array([[-1., -0.5, 0.0], [1., -0.5, 0.0]])
ker_eps_rho = 0.01

def X_alg(model, xnodes, N_iter=1000, dt=0.01, eps=1e-6, printflag=False):
    Nnode = xnodes.shape[0]
    for _ in range(N_iter):
        xnodesnew = xnodes + dt * model(xnodes)
        arc = np.linalg.norm(xnodesnew[1:] - xnodesnew[:-1], axis=-1, ord=2)
        arc = np.insert(arc, 0, 0) / np.sum(arc)
        itp = interp1d(np.cumsum(arc), xnodesnew, axis=0, kind='linear', fill_value="extrapolate")
        xnodesnew = itp(np.linspace(0, 1, Nnode))
        print(_, np.linalg.norm(xnodesnew - xnodes)) if printflag else None
        if np.linalg.norm(xnodesnew - xnodes) < eps:
            break
        else:
            xnodes = xnodesnew
    return xnodes
