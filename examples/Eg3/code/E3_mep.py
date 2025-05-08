#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from taskspec.utils_3 import ngrad, datapath, N_samplet, X_alg, Nnode, xends
randomseed = 0
np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

xnodes_init = np.linspace(xends[0], xends[1], Nnode)
dt = 0.01
xnodes_true = X_alg(ngrad, -xnodes_init, N_iter=10000, dt=dt, eps=1e-8)

np.savez(f'{datapath}/data/E3_gtstring.npz', xnodes_true=xnodes_true)

xdata = np.repeat(xnodes_true, N_samplet//(Nnode), axis=0)

np.random.shuffle(xdata)
noise_level = 0.1
xdata = xdata + noise_level * np.random.normal(size=xdata.shape)
ydata = ngrad(xdata)

np.savez(f'{datapath}/data/E3_dm_{randomseed}.npz', xdata=xdata, ydata=ydata, noise_level=noise_level)

