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

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
from taskspec.utils_3 import n_dim, ngrad, datapath, N_samplet
from tqdm import tqdm

tau = 0.0001
D = 0.1
if not is_jupyter_notebook():
    D = float(sys.argv[2])
ker_eps_rho = 0.01
randomseed = 0
np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

NX = 10
x = np.zeros((NX, n_dim))
xdata = np.zeros((N_samplet, n_dim))
NN = int(0.1/tau)
for j in range(NN*100):
    x = x + tau * ngrad(x) + np.sqrt(2 * D * tau) * np.random.normal(size=x.shape)

for i in tqdm(range(N_samplet//NX) ):
    for j in range(NN):
        x = x + tau * ngrad(x) + np.sqrt(2 * D * tau) * np.random.normal(size=x.shape)
    xdata[i*NX:(i+1)*NX] = x

ydata = ngrad(xdata)

np.savez(f'{datapath}/data/E3_d{D}_{randomseed}.npz', xdata=xdata, ydata=ydata, D=D, tau=tau)

