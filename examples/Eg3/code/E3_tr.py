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
randomseed = 1
D = 0.2
mep = 0
if not is_jupyter_notebook():
    nodes = int(sys.argv[2])
    mep = int(sys.argv[3])
    randomseed = int(sys.argv[4])
    D = float(sys.argv[5])
print("nodes:", nodes, "mep:", mep, "randomseed:", randomseed, "D:", D)

ker_eps_new = 0.001
ker_loss = 0.001
ker_nu = 0.01
print("ker_eps_new:", ker_eps_new, "ker_loss:", ker_loss, "ker_nu:", ker_nu)
Nterminals = 10

traininglr = 0.001
trainingstepmin, trainingstepmax = 4, 1024
MAX_ITER = 1000
print(f"traininglr: {traininglr} trainingstepmin: {trainingstepmin}  trainingstepmax: {trainingstepmax}")
print(f"MAX_ITER: {MAX_ITER}")

import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format}, precision=8, linewidth=200)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.config.list_physical_devices()

from matplotlib import pyplot as plt
plt.style.use('default')
import time
from taskspec.utils_3 import scale, N_sample, ker_eps_rho, n_dim, ngrad, X_alg, datapath, X_alg, Nnode, xends
from taskspec.model import DenseNet_Energy, Grad_model
from taskspec.method import estimate_measure, estimate_loss, estimate_measure_weight, print_res_info

outputfile = f'{datapath}/weights/weights_ts/E3_t{nodes}_{mep}_{randomseed}_{D}.npz'

np.random.seed(randomseed)
tf.random.set_seed(randomseed) 

weight_const, w_const = 0.0, 0.5 # const weight for all data, const weight for all called points
print(f"weight_const: {weight_const}  w_const: {w_const}")
M_softmax = 10.0
print(f"M_softmax: {M_softmax}")

datafile = f'{datapath}/data/data/E3_d{mep}_{D}_{randomseed}.npz'
assert os.path.exists(datafile)
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
assert os.path.exists(weights_file)
print("Loading", weights_file)
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
rho_theta = estimate_measure(new_sample, new_sample, ker_eps_new)  # rho is the new called measure
new_loss = estimate_loss(new_sample, xdata, rho_train, loss_train, ker_loss) # estimate loss for new sample
w_new = tf.nn.softmax(M_softmax * new_loss / np.mean(new_loss)) 
w_new = (w_new + w_const) / rho_theta
w_new = w_new / np.mean(w_new) 
nu_train = estimate_measure_weight(xdata, new_sample, w_new, ker_nu) # weighted loss
M_train = nu_train / rho_train
M_train = M_train / np.mean(M_train)
final_weight = weight_const + M_train
current_loss = np.mean(loss_train * final_weight)
current_loss_aftertraining = np.mean(loss_train * final_weight)

def compare(weights=None):
    if weights is not None:
        model.set_weights(weights)
    
    traj_loss = np.mean(np.sum(    np.square(new_sample - xnodes_true),    axis=0), axis=-1)
    
    point_loss = np.sum(np.square(ngrad(new_sample) - model(new_sample)), axis=-1)
    point_loss_l2 = np.mean(point_loss)
    point_loss_oo = np.sqrt(np.max(point_loss))
    
    new_loss_l2 = np.mean(new_loss)
    new_loss_oo = np.sqrt(np.max(new_loss))
    
    trainingmse = np.sqrt(tf.reduce_mean(tf.square(model(xdata) - ydata)))    
        
    return [traj_loss, 
            point_loss_l2, point_loss_oo, 
            new_loss_l2, new_loss_oo, 
            trainingmse, current_loss, current_loss_aftertraining]
    
# 0: Final target, unknown
# 1: true l2 loss on S(ft), unknown
# 2: true oo loss on S(ft), unknown
# 3: estimate l2 loss on S(ft), known
# 4: estimate oo loss on S(ft), known, l3n
# 5: training mse, known, l1n
# 6: current_loss_beforetraining, known, l4n before training
# 7: current_loss_aftertraining, known, l4n after training

model.set_weights(weights1)
new_sample = renew_nodes(new_sample)
u_tmp1 = np.copy(new_sample)

# save best weights with lowest L3N and L4N loss
res = compare()
print(0, np.array(res))

weights2 = None
u_tmp2 = None

ind_l3 = 0
best_l3 = res[4]
weights_l3 = np.copy(weights1)
u_l3 = np.copy(new_sample)

previous_weight = np.array(final_weight)
previous_training_loss = 0.0
currentstep = trainingstepmin

def get_save_date():
    return {
        'res': np.array(res),
        'weights1': weights1,
        'weights2': weights2,
        'ind_l3': ind_l3,
        'best_l3': best_l3,
        'weights_l3': weights_l3,
        'traininglr': traininglr,
        'trainingstepmin': trainingstepmin,
        'trainingstepmax': trainingstepmax,
        'nodes': nodes,
        'ker_eps_new': ker_eps_new,
        'ker_loss': ker_loss,
        'ker_nu': ker_nu,
        'weight_const': weight_const,
        'w_const': w_const,
        'M_softmax': M_softmax,
        'u_tmp1': u_tmp1,
        'u_tmp2': u_tmp2,
        'u_l3': u_l3,
    }

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=traininglr), loss="mse")
SGDflag = True

print("nodes:", nodes, "randomseed:", randomseed, "mep:", mep, "D:", D)
print_res_info()
calculateloss = compare()
print(f"Init.", np.array(calculateloss), currentstep)
res = [calculateloss]

for i in range(MAX_ITER):
    start_time = time.time()
    last_weights = model.get_weights()
    last_u_tmp = np.copy(new_sample)
    current_loss = np.mean(loss_train * final_weight) # loss before training        
    
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-7, patience=currentstep//4, restore_best_weights=True)
    results2 = model.fit(
        x=xdata, y=ydata,
        sample_weight=final_weight,
        epochs=currentstep,
        batch_size=N_sample,
        callbacks=[earlystopping],
        verbose=0,
    ) 
        
    loss_train = np.sum(np.square(model(xdata) - ydata), axis=-1) 
    current_loss_aftertraining = np.mean(loss_train * final_weight)    
    new_sample = renew_nodes(new_sample)
    assert np.isfinite(new_sample).all()
    rho_theta = estimate_measure(new_sample, new_sample, ker_eps_new)  # rho is the new called measure
    new_loss = estimate_loss(new_sample, xdata, rho_train, loss_train, ker_loss) # estimate loss for new sample
    w_new = tf.nn.softmax(M_softmax * new_loss / np.mean(new_loss)) 
    w_new = (w_new + w_const) / rho_theta
    w_new = w_new / np.mean(w_new) 
    nu_train = estimate_measure_weight(xdata, new_sample, w_new, ker_nu) # weighted loss
    M_train = nu_train / rho_train
    M_train = M_train / np.mean(M_train)
    previous_weight = np.array(final_weight)
    final_weight = weight_const + M_train    
    assert np.isfinite(final_weight).all()
    current_diff = np.linalg.norm(final_weight - previous_weight, ord=np.inf)
    
    calculateloss = compare()
    final_time = time.time() - start_time
    
    if calculateloss[4] > res[-1][4] and SGDflag == False:
        model.set_weights(last_weights)
        loss_train = np.sum(np.square(model(xdata) - ydata), axis=-1) 
        new_sample = np.copy(last_u_tmp)
        new_sample = renew_nodes(new_sample)
        assert np.isfinite(new_sample).all()
        rho_theta = estimate_measure(new_sample, new_sample, ker_eps_new)  # rho is the new called measure
        new_loss = estimate_loss(new_sample, xdata, rho_train, loss_train, ker_loss) # estimate loss for new sample
        w_new = tf.nn.softmax(M_softmax * new_loss / np.mean(new_loss)) 
        w_new = (w_new + w_const) / rho_theta
        w_new = w_new / np.mean(w_new) 
        nu_train = estimate_measure_weight(xdata, new_sample, w_new, ker_nu) # weighted loss
        M_train = nu_train / rho_train
        M_train = M_train / np.mean(M_train)
        final_weight = weight_const + M_train    
        assert np.isfinite(final_weight).all()
        current_diff = np.linalg.norm(final_weight - previous_weight, ord=np.inf)
        calculateloss = compare()
        
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=traininglr), loss="mse")
        SGDflag = True
        currentstep = np.max([currentstep//8, trainingstepmin])       
        print(f"{i:5}", np.array(calculateloss), currentstep, np.array([current_diff, final_time]))
        print("===== large l3n, use sgd, apply last weights")
        res = res + [calculateloss]
        continue
    
    if current_diff < 1.0 or calculateloss[4] < res[-1][4]:
        currentstep = np.min([currentstep * 2, trainingstepmax])
        if currentstep >= trainingstepmax and SGDflag == True:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=traininglr/10), loss="mse")
            print("===== achieved max, use adam")
            SGDflag = False
            
    else:
        currentstep = np.max([currentstep//8, trainingstepmin])
        if SGDflag == False:
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=traininglr), loss="mse")
            print("===== weights changes, use sgd ")
            SGDflag = True
    
    print(f"{i:5}", np.array(calculateloss), currentstep, np.array([current_diff, final_time]))
    res = res + [calculateloss]
    
    # restore early stopping weights
    if calculateloss[4] < best_l3:
        ind_l3 = i; best_l3 = calculateloss[4]; weights_l3 = model.get_weights(); u_l3 = np.copy(new_sample)
    
    if i > Nterminals:
        if np.std(np.array(res)[-Nterminals:,4]) <= 1e-5 and SGDflag == False:
            print("Converged because ", np.std(np.array(res)[-Nterminals:,4]))
            # break
        if i > ind_l3 + 100 and res[-1][4] > res[-2][4]:
            print("Early stopping because ", i, ind_l3)
            # break
    
weights2 = model.get_weights()
u_tmp2 = np.copy(new_sample)

np.savez(outputfile, **get_save_date())

