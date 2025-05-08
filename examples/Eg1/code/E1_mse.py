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
    
nodes = 1024
randomseed = 8
if not is_jupyter_notebook():
    nodes = int(sys.argv[2])
    randomseed = int(sys.argv[3])
print("Nodes: ", nodes, " Random Seed", randomseed)

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
from tqdm.keras import TqdmCallback
from taskspec.utils_1 import N_sample, tau, n_dim, scale_min, scale, datapath
from taskspec.model import ResNet

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

databasename = f'{datapath}/data/data/E1_dr{randomseed}.npz'
assert os.path.exists(databasename)
with np.load(databasename, allow_pickle=True) as data:
    print("loading data from", databasename)
    xdata, ydata = data['xdata'], data['ydata']
    assert xdata.shape == (N_sample, n_dim)
    assert ydata.shape == (N_sample, n_dim)
    rho_train = data['rho_train']
    assert rho_train.shape == (N_sample, )

model = ResNet(n_dim, nodes, tau, scale_min, scale)
model.build((None, n_dim))

weights_file = f'{datapath}/weights/weights_mse/E1_dr{randomseed}_{nodes}wts.npz'
if os.path.exists(weights_file):
    print("Loading", weights_file)
    weights1 = np.load(weights_file, allow_pickle=True)['weights']
    model.set_weights(weights1)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=1000, decay_rate=1.0, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=1000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=100000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(verbose=0), earlystopping],
    verbose=0,
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=1000, decay_rate=1.0, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=100000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(verbose=0), earlystopping],
    verbose=0,
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=1000, decay_rate=1.0, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-8, patience=10000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=1000000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(verbose=0), earlystopping],
    verbose=0,
)

weights = model.get_weights()
np.savez(weights_file, weights=weights)

def weighted_mse(y_true, y_pred, sample_weight):
    mse = tf.square(y_true - y_pred)
    weighted_mse = tf.reduce_mean(mse * sample_weight)
    return weighted_mse
with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, 1)
gradients = tape.gradient(loss, model.trainable_variables)
print("Gradient norm", tf.linalg.global_norm(gradients).numpy())

