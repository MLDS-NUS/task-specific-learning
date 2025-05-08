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
    
nodes = 128
randomseed = 0
alpha = 0.25
N_target = 100
if not is_jupyter_notebook():
    randomseed = int(sys.argv[2]) # In this study, alpha and nodes are fixed
print("Nodes: ", nodes, " Random Seed", randomseed, " alpha=", alpha)

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

databasename = f'{datapath}/data/data_as/E1_as_dr{alpha}_{randomseed}.npz'
assert os.path.exists(databasename)
with np.load(databasename, allow_pickle=True) as data:
    print("loading data from", databasename)
    xdata, ydata = data['xdata'], data['ydata']
    assert xdata.shape == (N_sample, n_dim)
    assert ydata.shape == (N_sample, n_dim)
    rho_train = data['rho_train']
    assert rho_train.shape == (N_sample, )
    assert alpha == data['alpha']
    assert N_target == data['N_target']
    

# Training Model

model = ResNet(n_dim, nodes, tau, scale_min, scale)
model.build((None, n_dim))

weights_file = f'{datapath}/weights/weights_as_mse/E1_as_dr{alpha}_{randomseed}_{nodes}wts.npz'
print("Loading existing", weights_file)
weights_mse = np.load(weights_file, allow_pickle=True)['weights']
model.set_weights(weights_mse)

def weighted_mse(y_true, y_pred, sample_weight):
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    weighted_mse = tf.reduce_mean(mse * sample_weight)
    return weighted_mse
with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, 1)
gradients = tape.gradient(loss, model.trainable_variables)
print("Loss", loss.numpy(), "Gradient norm", tf.linalg.global_norm(gradients).numpy())

# reweighting methods: 1/rho
model.set_weights(weights_mse)
sample_weights1 = 1 / rho_train
sample_weights1 = sample_weights1 / tf.reduce_mean(sample_weights1)

with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, sample_weights1)
gradients = tape.gradient(loss, model.trainable_variables)
print("Loss", loss.numpy(), "Gradient norm", tf.linalg.global_norm(gradients).numpy())

model.set_weights(weights_mse)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=10000, decay_rate=1.0, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-7, patience=100000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=1000000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(verbose=0), earlystopping],
    sample_weight=sample_weights1,
    verbose=0,
)

weights1 = model.get_weights()

print("Loss", loss.numpy(), "Gradient norm", tf.linalg.global_norm(gradients).numpy())
with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, sample_weights1)
gradients = tape.gradient(loss, model.trainable_variables)
print("Loss", loss.numpy(), "Gradient norm", tf.linalg.global_norm(gradients).numpy())

# reweighting methods: loss
model.set_weights(weights_mse)
sample_weights2 = tf.reduce_mean(tf.square(model(xdata)-ydata), axis=-1)
sample_weights2 = sample_weights2 / tf.reduce_mean(sample_weights2)

with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, sample_weights2)
gradients = tape.gradient(loss, model.trainable_variables)
print("Loss", loss.numpy(), "Gradient norm", tf.linalg.global_norm(gradients).numpy())

model.set_weights(weights_mse)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=10000, decay_rate=1.0, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-7, patience=100000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=1000000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(verbose=0), earlystopping],
    sample_weight=sample_weights2,
    verbose=0,
)

weights2 = model.get_weights()

print("Loss", loss.numpy(), "Gradient norm", tf.linalg.global_norm(gradients).numpy())
with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, sample_weights2)
gradients = tape.gradient(loss, model.trainable_variables)
print("Loss", loss.numpy(), "Gradient norm", tf.linalg.global_norm(gradients).numpy())

weights_rw_file = f'{datapath}/weights/weights_re/E1_re_dr{alpha}_{randomseed}wts.npz'

np.savez(weights_rw_file, weights_mse=weights_mse, weights1=weights1, weights2=weights2, 
         sample_weights1=sample_weights1, sample_weights2=sample_weights2)

