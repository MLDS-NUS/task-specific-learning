import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp

from taskspec.utils import find_project_root
datapath = find_project_root() + "/examples/Eg1"

x_range = (-25., 25.)
y_range = (-25., 25.)
z_range = (0., 50.)
N_sample = 100000
tau = 0.01
scale_min = np.array([x_range[0], y_range[0], z_range[0]])
scale = np.array([x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0]])

ker_eps_rho = 1.0

n_dim = 3
c_sigma = 10.0
c_rho = 28.0
c_beta = 8.0 / 3.0
c0 = np.array([c_sigma, c_rho, c_beta])

def lorenz_system(xyz, c=c0):
    assert xyz.shape[1] == 3
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    c_sigma, c_rho, c_beta = c
    dxdt = c_sigma * (y - x)
    dydt = c_rho * x - y - x * z
    dzdt = x * y - c_beta * z
    return np.stack([dxdt, dydt, dzdt], axis=1)

def lorenz_jac(xyz, c=c0):
    assert xyz.shape[1] == 3
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    c_sigma, c_rho, c_beta = c
    J = np.zeros((*xyz.shape, 3))
    J[:, 0, 0] = -c_sigma
    J[:, 0, 1] = c_sigma
    J[:, 1, 0] = c_rho - z
    J[:, 1, 1] = -1.0
    J[:, 1, 2] = -x
    J[:, 2, 0] = y
    J[:, 2, 1] = x
    J[:, 2, 2] = -c_beta
    return J

def lorenz_system_vec(x):
    return lorenz_system(np.reshape(x, (-1,3))).flatten()

def lorenz_jac_vec(x):
    return block_diag(*lorenz_jac(np.reshape(x, (-1,3))))

def rhs(t,x):
    return lorenz_system_vec(x)
def rhs_jac(t,x):
    return lorenz_jac_vec(x)

tol = 0.01
rtol = 1e-3 * tol
atol = 1e-6 * tol
def tm(tau):
    def true_model(x):
        return solve_ivp(rhs, 
                        (0, tau), 
                        x.flatten(), 
                        method='Radau', 
                        t_eval=np.array([tau]), 
                        dense_output=False, 
                        vectorized=False, 
                        rtol=rtol, atol=atol, jac=rhs_jac).y[:,0].reshape((-1,n_dim))
    return true_model

def X_alg(model, x, N_target):
    result = np.zeros((N_target+1, *x.shape))
    for i in range(N_target):
        result[i] = x
        x = model(x)
    result[N_target] = x
    return result

def cal_traj(model, x, N_target):
    supp = np.zeros((N_target, *x.shape))
    traj = np.zeros((N_target, *x.shape))
    for i in range(N_target):
        supp[i] = x
        x = model(x)
        traj[i] = x
    return supp, traj
