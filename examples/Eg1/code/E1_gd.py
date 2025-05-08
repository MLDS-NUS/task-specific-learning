#!/usr/bin/env python
# coding: utf-8

from taskspec.utils import is_jupyter_notebook
import os, sys
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
if is_jupyter_notebook():
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
else:
    print(sys.argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

randomseed = 0
if not is_jupyter_notebook():
    randomseed = int(sys.argv[2])
print("Random seed", randomseed)

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
from scipy.integrate import solve_ivp
from taskspec.utils_1 import x_range, y_range, z_range, N_sample, ker_eps_rho, n_dim, tau, rhs, rhs_jac, datapath
from taskspec.method import estimate_measure
from tqdm import tqdm
print("datapath", datapath)

if os.path.exists(f'{datapath}/data/data/E1_dr{randomseed}.npz'):
    print("data file exists")
    if not is_jupyter_notebook():
        print("Exiting")
        sys.exit()

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 
x_sample = np.random.uniform(low=[x_range[0], y_range[0], z_range[0]], 
                             high=[x_range[1], y_range[1], z_range[1]], 
                             size=(N_sample,n_dim) )

initial_state = x_sample.flatten()

t_span = (0, tau)
Tnum = 1
t_eval = np.array([tau])

tol = 0.01
rtol = 1e-3 * tol
atol = 1e-6 * tol

NN = 1000
solutiony = np.zeros((N_sample, n_dim, Tnum))
for i in tqdm(range(0, N_sample, NN)):
    solution = solve_ivp(rhs, 
                        t_span, 
                        initial_state[n_dim*i:n_dim*(i+NN)], 
                        method='Radau', 
                        t_eval=t_eval, 
                        dense_output=False, 
                        vectorized=False, 
                        rtol=rtol, atol=atol, jac=rhs_jac)
    solutiony[i:i+NN] = solution.y.reshape((NN, n_dim, Tnum))

xdata = x_sample
ydata = solutiony[:, :, -1]
print(xdata.shape, ydata.shape)

rho_train = estimate_measure(xdata, xdata, ker_eps=ker_eps_rho, method=True)

np.savez(f'{datapath}/data/data/E1_dr{randomseed}.npz', 
         xdata=xdata, ydata=ydata, rho_train=rho_train, ker_eps_rho=ker_eps_rho, randomseed=randomseed)

if randomseed==0:
    np.random.seed(100)
    initial_tests = np.random.uniform(low=[0.75*x_range[0]+0.25*x_range[1], 0.75*y_range[0]+0.25*y_range[1], 0.75*z_range[0]+0.25*z_range[1]],
                                    high=[0.25*x_range[0]+0.75*x_range[1], 0.25*y_range[0]+0.75*y_range[1], 0.25*z_range[0]+0.75*z_range[1]],
                                    size=(100, n_dim))
    np.savez(f'{datapath}/data/E1_initial.npz', initial_tests=initial_tests)

