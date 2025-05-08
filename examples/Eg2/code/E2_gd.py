#!/usr/bin/env python
# coding: utf-8

from taskspec.utils import is_jupyter_notebook
import os, sys
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
if is_jupyter_notebook():
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
else:
    print(sys.argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

randomseed = 19
init_id = 0
if not is_jupyter_notebook():
    randomseed = int(sys.argv[2])

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
from taskspec.utils_2 import x_range, y_range, u_range, N_sample, ker_eps_rho, n_dim_u, n_dim, f_true, datapath, NT, Nx, bounds, Efunc_all, Egrad_all, forward_map, x_ref
from taskspec.method import estimate_measure
from scipy.optimize import minimize

if os.path.exists(f'{datapath}/data/data/E2_dr{randomseed}.npz'):
    print("data file exists")
    if not is_jupyter_notebook():
        print("Exiting")
        sys.exit()

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 
x_sample = np.random.uniform(low=[x_range[0], y_range[0], u_range[0]*1.1], 
                             high=[x_range[1], y_range[1], u_range[1]*1.1], 
                             size=(N_sample, n_dim) )
x_sample[:,2] = np.clip(x_sample[:,2], u_range[0], u_range[1])

xdata = x_sample
ydata = f_true(x_sample)

rho_train = estimate_measure(xdata, xdata, ker_eps=ker_eps_rho, method=True)

np.savez(f'{datapath}/data/data/E2_dr{randomseed}.npz', xdata=xdata, ydata=ydata, rho_train=rho_train, ker_eps_rho=ker_eps_rho, randomseed=randomseed)

if randomseed==0:
    u_tmp = tf.Variable(np.zeros((Nx,NT,n_dim_u)), dtype=tf.float64) # 1*20*1 array
    x_tracking = x_ref[init_id][None,:,:]
    x_initial = x_tracking[0,0][None,:]
                    
    Efunc = lambda u: Efunc_all(u, f_true, x_initial, x_tracking)
    Egrad = lambda u: Egrad_all(u, f_true, x_initial, x_tracking)
    
    result = minimize(Efunc, u_tmp[0,:,0], jac=Egrad, bounds=bounds, method='L-BFGS-B', tol=1e-12)
    u_tmp.assign(result.x[None,:,None])

    print(f"Optimal x: {result.x}")
    print(f"Function value at optimal x: {result.fun}")
    x_final = forward_map(f_true, x_initial, u_tmp)
    u_best = u_tmp.numpy()[0,:,0]
    
    np.savez(f'{datapath}/data/E2_ini{init_id}.npz', 
         x_initial=x_initial, x_tracking=x_tracking, 
         u_best=u_best, x_final=x_final)

