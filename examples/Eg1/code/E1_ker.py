#!/usr/bin/env python
# coding: utf-8

import os

nodes = 128
N_target = 25
randomseed = 0
initial_id = randomseed
print("nodes:", nodes, "N_target:", N_target, "randomseed:", randomseed, "initial_id:", initial_id)

KER_DETERMINE = True

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

from matplotlib import pyplot as plt
plt.style.use('default')
from taskspec.utils_1 import N_sample, tau, ker_eps_rho, n_dim, scale_min, scale, datapath
from taskspec.utils_1 import X_alg, tm
from taskspec.model import ResNet
from taskspec.method import estimate_measure, estimate_loss, estimate_measure_weight

with np.load(f'{datapath}/data/E1_initial.npz') as data:
    initial_test = data['initial_tests'][randomseed][None,:]
assert initial_test.shape == (1, n_dim)
true_model = tm(tau)

traininglr, trainingstep = 0.005, 1000
trainingstepmin, trainingstepadd = 0, 10
MAX_ITER = 1000
print(f"traininglr: {traininglr}  trainingstep: {trainingstep}")
print(f"trainingstepmin: {trainingstepmin}  trainingstepadd: {trainingstepadd}")
print(f"MAX_ITER: {MAX_ITER}")

weight_const, w_const = 0.0, 0.5 # const weight for all data, const weight for all called points
print(f"weight_const: {weight_const}  w_const: {w_const}")
M_softmax = 10.0
print(f"M_softmax: {M_softmax}")

databasename = f'{datapath}/data/data/E1_dr{randomseed}.npz'
assert os.path.exists(databasename)
with np.load(databasename, allow_pickle=True) as data:
    print("loading data from", databasename)
    xdata, ydata = data['xdata'], data['ydata']
    assert xdata.shape == (N_sample, n_dim)
    assert ydata.shape == (N_sample, n_dim)
    rho_train = data['rho_train']
    assert rho_train.shape == (N_sample, )
    assert ker_eps_rho == data['ker_eps_rho']
    print(f"ker_eps_rho: {ker_eps_rho}")

model = ResNet(n_dim, nodes, tau, scale_min, scale)
model.build((None, n_dim))

weights_file = f'{datapath}/weights/weights_mse/E1_dr{randomseed}_{nodes}wts.npz'
if os.path.exists(weights_file):
    print("Loading", weights_file)
    weights1 = np.load(weights_file, allow_pickle=True)['weights']

model.set_weights(weights1)
traj_true = X_alg(true_model, initial_test, N_target)[:,0]
new_sample = X_alg(model, initial_test, N_target)[:,0]

# check MSE result
def weighted_mse(y_true, y_pred, sample_weight):
    mse = tf.square(y_true - y_pred)
    weighted_mse = tf.reduce_mean(mse * sample_weight)
    return weighted_mse
with tf.GradientTape() as tape:
    predictions = model(xdata, training=True)
    loss = weighted_mse(ydata, predictions, 1)
gradients = tape.gradient(loss, model.trainable_variables)
gradientnorm = tf.linalg.global_norm(gradients).numpy()
print(f"Loading MSE weights: loss {loss}   gradientnorm {gradientnorm}")

# determine ker_eps_new
if KER_DETERMINE:
    from scipy.signal import find_peaks
    ker_check = ker_eps_rho * 2. **np.array(range(-5,10))
    ker_res = np.zeros((ker_check.shape[0], new_sample.shape[0]))
    for i in range(ker_check.shape[0]):
        ker_res[i] = estimate_measure(new_sample, new_sample, ker_check[i]) 
    ker_diff = np.linalg.norm(ker_res[1:] - ker_res[:-1], axis=-1)**2
    ker_diff = ker_diff[1:] + ker_diff[:-1]
    peaks, _ = find_peaks(-ker_diff, distance=1)
    ker_eps_new = ker_check[1+peaks][0]

rho_theta = estimate_measure(new_sample, new_sample, ker_eps_new)  # rho is the new called measure
loss_train = np.sum(np.square(model(xdata) - ydata), axis=-1)

def estimate_loss(x_call, x_data, rho_data, data_loss, ker_eps=0.1):
    # estimate loss on x_call based on data_loss on x_data wtih rho_data measure
    # x_call.shape = (n_call, dim)
    # x_data.shape = (n_data, dim)
    # rho_data.shape = (n_data, )
    # data_loss.shape = (n_data, )
    e = x_data[None,:,:] - x_call[:,None,:] # (n_call, n_data, dim)
    e = np.sum(np.square(e), axis=-1) # (n_call, n_data)
    e = e/ker_eps
    exp_e = np.exp( -e )
    w_x = exp_e / rho_data #+ 1e-8 # (n_call, n_data)
    w_x = w_x / np.mean(w_x, axis=-1, keepdims=True)  # normalize for each point # (n_call, n_data)
    px = np.mean(w_x * data_loss, axis=-1, keepdims=False) # (n_call,)
    return px # (n_call,)

# determine ker_nu: method 1
if KER_DETERMINE:
    kth=10
    dist = np.zeros((new_sample.shape[0],))
    nearn = np.zeros((new_sample.shape[0],kth), dtype=np.int64)
    for i in range(new_sample.shape[0]):
        thisdist = np.square(np.linalg.norm(new_sample[i] - xdata, axis=-1))
        nearn[i] = np.argpartition(thisdist, kth=kth)[:kth]
        dist[i] = np.mean(thisdist[nearn[i]])
    nearn = np.sort(np.unique(nearn))

# determine ker_loss
if KER_DETERMINE:
    ker_check = ker_eps_rho * 2. **np.array(range(-5,5))
    ker_res = np.zeros((ker_check.shape[0], nearn.shape[0]))
    for i in range(ker_check.shape[0]):
        for j in range(nearn.shape[0]):
            ker_res[i,j] = estimate_loss(xdata[nearn[j]][None,:], 
                                    np.delete(xdata, nearn[j], axis=0), 
                                    np.delete(rho_train, nearn[j], axis=0), 
                                    np.delete(loss_train, nearn[j], axis=0), ker_check[i]) 
    loss_err = np.linalg.norm(ker_res - loss_train[nearn], axis=-1)
    ker_loss = ker_check[np.argmin(loss_err)]
    print("ker_loss", ker_loss)

# determine ker_loss
if KER_DETERMINE:
    cxy = np.zeros((ker_check.shape[0],))
    for i in range(ker_check.shape[0]):
        cxy[i] = np.sum(ker_res[i] * loss_train[nearn]) / np.sum(ker_res[i]**2)
    loss_err = np.linalg.norm(ker_res*cxy[:,None] - loss_train[nearn], axis=-1)
    ker_loss = ker_check[np.argmin(loss_err)]
    print("ker_loss", ker_loss)

new_loss = estimate_loss(new_sample, xdata, rho_train, loss_train, ker_loss) # estimate loss for new sample
w_new = tf.nn.softmax(M_softmax * new_loss / np.mean(new_loss)) 
w_new = (w_new + w_const) / rho_theta
w_new = w_new / np.mean(w_new) 

print(f"ker_eps_new: {ker_eps_new}  ker_loss: {ker_loss} ")

