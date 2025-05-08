#!/usr/bin/env python
# coding: utf-8

KER_DETERMINE = True

nodes = 12
randomseed = 5
init_id = 0

import os
import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

from matplotlib import pyplot as plt
plt.style.use('default')
from taskspec.utils_2 import N_sample, ker_eps_rho, n_dim_u, n_dim, datapath, n_dim, f_true, scale, X_alg, NT, Nx, bounds, Efunc_all, Egrad_all, forward_map, x_ref
from taskspec.model import DenseNet, PolyModel
from taskspec.method import estimate_measure
from scipy.optimize import minimize

weight_const, w_const = 0.0, 0.5 # const weight for all data, const weight for all called points
print(f"weight_const: {weight_const}  w_const: {w_const}")
M_softmax = 10.0
print(f"M_softmax: {M_softmax}")

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
    print(f"ker_eps_rho: {ker_eps_rho}")
    

u_tmp = tf.Variable(np.zeros((Nx,NT,n_dim_u)), dtype=tf.float64) # 1*20*1 array
x_tracking = x_ref[init_id][None,:,:]
x_initial = x_tracking[0,0][None,:]

with np.load(f'{datapath}/data/E2_ini{init_id}.npz', allow_pickle=True) as data:
    assert np.linalg.norm(x_initial - data['x_initial']) < 1e-8
    assert np.linalg.norm(x_tracking - data['x_tracking']) < 1e-8
    x_initial = data['x_initial']
    x_tracking = data['x_tracking']
    u_best = data['u_best'][None,:,None]
    x_final = data['x_final']

if nodes < 5:
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=nodes + 1, include_bias=False)
    xdata_poly = poly.fit_transform(xdata/scale)
    pd = poly.powers_
    
    model = PolyModel(n_dim, 1, pd, scale=scale)
    from sklearn.linear_model import LinearRegression
    tmp = LinearRegression()
    tmp.fit(xdata_poly, ydata, sample_weight=np.ones(N_sample))
    weights1 = model.get_weights()
    weights1[0][:] = tmp.coef_.T
    weights1[1][:] = tmp.intercept_
    
    model.set_weights(weights1)
    
    
else:    
    model = DenseNet(n_dim, 1, nodes=nodes, scale=scale)
    model.build((None, n_dim))
    weights_file = f'{datapath}/weights/weights_mse/E2_dr{randomseed}_{nodes}wts_{init_id}.npz'
    with np.load(weights_file, allow_pickle=True) as data:
        weights1 = data['weights']
        x_initial = data['x_initial']
        x_tracking =  data['x_tracking']
        u_model = data['u_model']
        x_model = data['x_model']
    model.set_weights(weights1)
    u_tmp.assign(u_model[None,:,None])

model.set_weights(weights1)
Efunc = lambda u: Efunc_all(u, model, x_initial, x_tracking)
Egrad = lambda u: Egrad_all(u, model, x_initial, x_tracking)
result = minimize(Efunc, np.array(u_tmp).flatten(), jac=Egrad, bounds=bounds, method='L-BFGS-B', tol=1e-10)
u_tmp.assign(result.x[None,:,None])
u_tmp1 = u_tmp.numpy()

x_model = forward_map(model, x_initial, u_tmp)
u_model = u_tmp.numpy()[0,:,0]

# the loss the differnce between the real trajectory and the target trajectory
x_real = forward_map(f_true, x_initial, u_best)
xu_traj_real, x_traj_real = X_alg(f_true, x_initial, u_best)
fu_real = f_true(xu_traj_real)
best_traj_loss = tf.reduce_sum(tf.square(x_real - x_tracking)).numpy()
print('Best trajectory loss:', best_traj_loss)

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

result = minimize(Efunc, np.array(u_tmp).flatten(), jac=Egrad, bounds=bounds, method='L-BFGS-B', tol=1e-10)
u_tmp.assign(result.x[None,:,None])
new_sample, x_traj = X_alg(model, x_initial, u_tmp)

# determine ker_eps_new
if KER_DETERMINE:
    from scipy.signal import find_peaks
    ker_check = ker_eps_rho * 2. **np.linspace(-2,8,51)
    ker_res = np.zeros((ker_check.shape[0], new_sample.shape[0]))
    for i in range(ker_check.shape[0]):
        ker_res[i] = estimate_measure(new_sample, new_sample, ker_check[i]) 
    ker_diff = np.linalg.norm(ker_res[1:] - ker_res[:-1], axis=-1)**2
    ker_diff = ker_diff[1:] + ker_diff[:-1]
    peaks, _ = find_peaks(-ker_diff, distance=1)
    ker_eps_new = ker_check[1+peaks][0]
    print("ker_eps_new", ker_eps_new)

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
    ker_check = ker_eps_rho * 2. **np.linspace(-2,8,51)
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

new_loss = estimate_loss(new_sample, xdata, rho_train, loss_train, ker_loss) # estimate loss for new sample
w_new = tf.nn.softmax(M_softmax * new_loss / np.mean(new_loss)) 
w_new = (w_new + w_const) / rho_theta
w_new = w_new / np.mean(w_new) 

print(f"ker_eps_new: {ker_eps_new}  ker_loss: {ker_loss} ")

