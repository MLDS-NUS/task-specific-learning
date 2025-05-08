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

nodes = 8
randomseed = 0
init_id = 0
if not is_jupyter_notebook():
    nodes = int(sys.argv[2])
    randomseed = int(sys.argv[3])
print("nodes:", nodes, "randomseed:", randomseed, "init_id:", init_id)

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
from tqdm.keras import TqdmCallback
from taskspec.utils_2 import N_sample, ker_eps_rho, scale, n_dim_u, n_dim, datapath, NT, Nx, bounds, Efunc_all, Egrad_all, forward_map, x_ref
from taskspec.model import DenseNet
from scipy.optimize import minimize

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

databasename = f'{datapath}/data/data/E2_dr{randomseed}.npz'
assert os.path.exists(databasename)
with np.load(databasename, allow_pickle=True) as data:
    print("loading data from", databasename)
    xdata, ydata = data['xdata'], data['ydata']
    assert xdata.shape == (N_sample, n_dim)
    assert ydata.shape == (N_sample, 1)
    rho_train = data['rho_train']
    assert rho_train.shape == (N_sample, )
    assert ker_eps_rho == data['ker_eps_rho']

# Training Model

model = DenseNet(n_dim, 1, nodes=nodes, scale=scale)
model.build((None, n_dim))

weights_file = f'{datapath}/weights/weights_mse/E2_dr{randomseed}_{nodes}wts_{init_id}.npz'
if os.path.exists(weights_file):
    print("Loading", weights_file)
    weights1 = np.load(weights_file, allow_pickle=True)['weights']
    model.set_weights(weights1)

def weighted_mse(y_true, y_pred, sample_weight):
    mse = tf.square(y_true - y_pred)
    weighted_mse = tf.reduce_mean(mse * sample_weight)
    return weighted_mse
with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, 1)
gradients = tape.gradient(loss, model.trainable_variables)
print("Gradient norm", tf.linalg.global_norm(gradients).numpy())

while loss > 1:
    print("retraining")
    model = DenseNet(n_dim, 1, nodes=nodes, scale=scale)
    model.build((None, n_dim))
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.005, decay_steps=1000, decay_rate=0.9, staircase=True)
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
    
    with tf.GradientTape() as tape:
        predictions = model(xdata, training=True)
        loss = weighted_mse(ydata, predictions, 1)
    gradients = tape.gradient(loss, model.trainable_variables)
    print("Gradient norm", tf.linalg.global_norm(gradients).numpy())

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=1000, decay_rate=0.9, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=1000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=1000000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(verbose=0), earlystopping],
    verbose=0,
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=1000, decay_rate=0.95, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mse")
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-8, patience=10000, restore_best_weights=True)
results = model.fit(
    x=xdata,
    y=ydata,
    epochs=10000000,
    batch_size=N_sample,
    callbacks=[TqdmCallback(verbose=0), earlystopping],
    verbose=0,
)

weights = model.get_weights()
np.savez(weights_file, weights=weights)

with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, 1)
gradients = tape.gradient(loss, model.trainable_variables)
print("Gradient norm", tf.linalg.global_norm(gradients).numpy())

u_tmp = tf.Variable(np.zeros((Nx,NT,n_dim_u)), dtype=tf.float64) # 1*20*1 array
x_tracking = x_ref[init_id][None,:,:]
x_initial = x_tracking[0,0][None,:]
                
Efunc = lambda u: Efunc_all(u, model, x_initial, x_tracking)
Egrad = lambda u: Egrad_all(u, model, x_initial, x_tracking)

result = minimize(Efunc, u_tmp[0,:,0], jac=Egrad, bounds=bounds, method='L-BFGS-B', tol=1e-12)
u_tmp.assign(result.x[None,:,None])

print(f"Optimal x: {result.x}")
print(f"Function value at optimal x: {result.fun}")
x_model = forward_map(model, x_initial, u_tmp)
u_model = u_tmp.numpy()[0,:,0]

assert np.linalg.norm(u_tmp) > 0

weights = model.get_weights()
np.savez(weights_file, weights=weights, x_initial=x_initial, x_tracking=x_tracking, 
         u_model=u_model, x_model=x_model)

