#!/usr/bin/env python
# coding: utf-8

KER_DETERMINE = True

nodes = 32
randomseed = 8
D = 0.1
mep = 0

traininglr = 0.001
trainingstepmin, trainingstepmax = 4, 1024
MAX_ITER = 2000
print(f"traininglr: {traininglr} trainingstepmin: {trainingstepmin}  trainingstepmax: {trainingstepmax}")
print(f"MAX_ITER: {MAX_ITER}")

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
from taskspec.utils_3 import x_range, y_range, scale, N_sample, ker_eps_rho, n_dim, ngrad, datapath, X_alg, Nnode, xends
from taskspec.model import DenseNet_Energy, Grad_model
from taskspec.method import estimate_measure

weight_const, w_const = 0.0, 0.5 # const weight for all data, const weight for all called points
print(f"weight_const: {weight_const}  w_const: {w_const}")
M_softmax = 10.0
print(f"M_softmax: {M_softmax}")

datafile = f'{datapath}/data/data/E3_d{mep}_{D}_{randomseed}.npz'
with np.load(datafile, allow_pickle=True) as data:
    xdata, ydata, rho_train = data['xdata'], data['ydata'], data['rho_train']
    assert ker_eps_rho == data['ker_eps_rho']
assert xdata.shape == (N_sample, n_dim)
assert ydata.shape == (N_sample, n_dim)
assert rho_train.shape == (N_sample, )

# Training Model

V_model = DenseNet_Energy(n_dim, nodes=nodes, scale=scale)
model = Grad_model(V_model)
model.build((None, n_dim))

weights_file = f'{datapath}/weights/weights_mse/E3_w{nodes}_{mep}_{randomseed}_{D}.npy'
print("weights_file:", weights_file)
weights1 = np.load(weights_file, allow_pickle=True)
model.set_weights(weights1)

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

with np.load(f'{datapath}/data/E3_gtstring.npz', allow_pickle=True) as data:
    xnodes_true = data['xnodes_true']
    assert xnodes_true.shape == (Nnode, n_dim)
    fu_real = ngrad(xnodes_true)

xnodes_init = np.linspace(xends[0], xends[1], Nnode)
new_sample = X_alg(model, -xnodes_init, N_iter=10000, dt=0.01)

desloss = np.mean(np.sum(
    np.square(new_sample - xnodes_true),
    axis=0), axis=-1)
print('desloss', desloss)

def renew_nodes(new_sample, N_iter=100):
    new_sample = X_alg(model, new_sample, N_iter=N_iter, dt=0.01)
    return new_sample

loss_train = np.sum(np.square(model(xdata) - ydata), axis=-1)
new_sample = renew_nodes(new_sample)

# determine ker_eps_new
if KER_DETERMINE:
    from scipy.signal import find_peaks
    ker_check = ker_eps_rho * 2. **np.linspace(-7,8,51)
    ker_res = np.zeros((ker_check.shape[0], new_sample.shape[0]))
    for i in range(ker_check.shape[0]):
        ker_res[i] = estimate_measure(new_sample, new_sample, ker_check[i]) 
    ker_diff = np.linalg.norm(ker_res[1:] - ker_res[:-1], axis=-1)**2
    ker_diff = ker_diff[1:] + ker_diff[:-1]
    peaks, _ = find_peaks(-ker_diff, distance=1)
    ker_eps_new = ker_check[1+peaks][0]
    print("ker_eps_new", ker_eps_new)

# determine ker_nu: method 1
if KER_DETERMINE:
    kth=5
    dist = np.zeros((new_sample.shape[0],))
    nearn = np.zeros((new_sample.shape[0],kth), dtype=np.int64)
    for i in range(new_sample.shape[0]):
        thisdist = np.square(np.linalg.norm(new_sample[i] - xdata, axis=-1))
        nearn[i] = np.argpartition(thisdist, kth=kth)[:kth]
        dist[i] = np.mean(thisdist[nearn[i]])
    # nearn = np.sort(np.unique(nearn))

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
    kth=20
    dist = np.zeros((new_sample.shape[0],))
    nearn = np.zeros((new_sample.shape[0],kth), dtype=np.int64)
    for i in range(new_sample.shape[0]):
        thisdist = np.square(np.linalg.norm(new_sample[i] - xdata, axis=-1))
        nearn[i] = np.argpartition(thisdist, kth=kth)[:kth]
        dist[i] = np.mean(thisdist[nearn[i]])
    nearn = np.sort(np.unique(nearn))

# determine ker_loss
if KER_DETERMINE:
    ker_check = ker_eps_rho * 2. **np.linspace(-5,5,21)
    ker_res = np.zeros((ker_check.shape[0], nearn.shape[0]))
    for i in range(ker_check.shape[0]):
        for j in range(nearn.shape[0]):
            ker_res[i,j] = estimate_loss(xdata[nearn[j]][None,:], 
                                    np.delete(xdata, nearn[j], axis=0), 
                                    np.delete(rho_train, nearn[j], axis=0), 
                                    np.delete(loss_train, nearn[j], axis=0), ker_check[i]) 
    loss_err = np.linalg.norm(ker_res - loss_train[nearn], axis=-1, ord=np.inf)
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

print(f"ker_eps_new: {ker_eps_new}  ker_loss: {ker_loss} ")

