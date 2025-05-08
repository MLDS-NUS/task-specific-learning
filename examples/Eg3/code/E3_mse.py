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
    
nodes = 32
randomseed = 0
D = 1.0
mep = 0
if not is_jupyter_notebook():
    nodes = int(sys.argv[2])
    mep = int(sys.argv[3])
    randomseed = int(sys.argv[4])
    D = float(sys.argv[5])
print("nodes:", nodes, "mep:", mep, "randomseed:", randomseed, "D:", D)

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
from tqdm.keras import TqdmCallback
from taskspec.utils_3 import scale, N_sample, ker_eps_rho, n_dim, datapath, N_samplet
from taskspec.method import estimate_measure
from taskspec.model import DenseNet_Energy, Grad_model

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 
datafile = f'{datapath}/data/data/E3_d{mep}_{D}_{randomseed}.npz'

if not os.path.exists(datafile):
    N_sample2 = int(N_sample * mep / 100) 
    N_sample1 = N_sample - N_sample2
    indices1 = np.random.choice(N_samplet, N_sample1, replace=False) 
    indices2 = np.random.choice(N_samplet, N_sample2, replace=False) 
    
    databasename = f'{datapath}/data/E3_d{D}_{0}.npz'
    assert os.path.exists(databasename)
    with np.load(databasename, allow_pickle=True) as data1:
        xdata1, ydata1 = data1['xdata'], data1['ydata']
        assert xdata1.shape == (N_samplet, n_dim)
        assert ydata1.shape == (N_samplet, n_dim)
        
    databasename = f'{datapath}/data/E3_dm_{0}.npz'
    assert os.path.exists(databasename)
    with np.load(databasename, allow_pickle=True) as data2:
        xdata2, ydata2 = data2['xdata'], data2['ydata']
        assert xdata2.shape == (N_samplet, n_dim)
        assert ydata2.shape == (N_samplet, n_dim)

    xdata = np.concatenate([xdata1[indices1], xdata2[indices2]], axis=0)
    ydata = np.concatenate([ydata1[indices1], ydata2[indices2]], axis=0)

    rho_train = estimate_measure(xdata, xdata, ker_eps=ker_eps_rho, method=True)
    np.savez(datafile, xdata=xdata, ydata=ydata, rho_train=rho_train, ker_eps_rho=ker_eps_rho)
else:
    with np.load(datafile, allow_pickle=True) as data:
        xdata, ydata, rho_train = data['xdata'], data['ydata'], data['rho_train']
        assert ker_eps_rho == data['ker_eps_rho']

assert xdata.shape == (N_sample, n_dim)
assert ydata.shape == (N_sample, n_dim)
assert rho_train.shape == (N_sample, )

# Training Model

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 
V_model = DenseNet_Energy(n_dim, nodes=nodes, scale=scale)
model = Grad_model(V_model)
model.build((None, n_dim))

weights_file = f'{datapath}/weights/weights_mse/E3_w{nodes}_{mep}_{randomseed}_{D}.npy'
if os.path.exists(weights_file):
    print("Loading", weights_file)
    weights1 = np.load(weights_file, allow_pickle=True)
    model.set_weights(weights1)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=1000, decay_rate=0.9, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, patience=1000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=100000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(), earlystopping],
    verbose=0,
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=1000, decay_rate=0.9, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=1000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=100000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(), earlystopping],
    verbose=0,
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=1000, decay_rate=0.9, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-7, patience=1000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=100000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(), earlystopping],
    verbose=0,
)

weights = model.get_weights()
np.save(weights_file, weights)

