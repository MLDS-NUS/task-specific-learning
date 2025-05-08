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

alpha = 0.0
N_target = 100
ker_nu = 1.0
if not is_jupyter_notebook():
    alpha = float(sys.argv[3])    

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
from scipy.integrate import solve_ivp
from taskspec.utils_1 import N_sample, ker_eps_rho, n_dim, tau, rhs, rhs_jac, datapath, tm, X_alg
from taskspec.method import estimate_measure
from tqdm import tqdm

if os.path.exists(f'{datapath}/data/data_as/E1_as_dr{alpha}_{randomseed}.npz'):
    print("data file exists")
    if not is_jupyter_notebook():
        print("Exiting")
        sys.exit()

N_sample2 = int(N_sample * alpha)  # data near trajectory
N_sample1 = N_sample - N_sample2

databasename = f'{datapath}/data/data/E1_dr{randomseed}.npz'
with np.load(databasename, allow_pickle=True) as data:
    xdata0, ydata0 = data['xdata'], data['ydata']
    
np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

with np.load(f'{datapath}/data/E1_initial.npz') as data:
    initial_test = data['initial_tests'][randomseed][None,:]
assert initial_test.shape == (1, n_dim)
true_model = tm(tau)

Sftrue = X_alg(true_model, initial_test, N_target)[:N_target,0]

x_sample2 = np.random.normal(loc=np.repeat(Sftrue, N_sample2//(N_target), axis=0), scale=ker_nu,
                             size=(N_sample2, n_dim) )
x_sample = np.concatenate([xdata0[:N_sample1], x_sample2], axis=0)

initial_state = x_sample.flatten()

t_span = (0, tau)
Tnum = 1
t_eval = np.array([tau])

tol = 0.01
rtol = 1e-3 * tol
atol = 1e-6 * tol

#simulate short trajectory with high accuracy
NN = 1000
solutiony = np.zeros((x_sample.shape[0], n_dim, Tnum))
for i in tqdm(range(0, x_sample.shape[0], NN)):
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

np.savez(f'{datapath}/data/data_as/E1_as_dr{alpha}_{randomseed}.npz', 
         alpha=alpha, N_target=N_target, ker_nu=ker_nu,
         xdata=xdata, ydata=ydata, rho_train=rho_train, ker_eps_rho=ker_eps_rho, randomseed=randomseed)

