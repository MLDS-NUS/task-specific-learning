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
    
nodes = 16
randomseed = 0
init_id = 0
if not is_jupyter_notebook():
    nodes = int(sys.argv[2])
    randomseed = int(sys.argv[3])
print("nodes:", nodes, "randomseed:", randomseed, "init_id:", init_id)

ker_eps_new = 0.01
ker_loss = 0.01
ker_nu = 0.05
print("ker_eps_new:", ker_eps_new, "ker_loss:", ker_loss, "ker_nu:", ker_nu)
Nterminals = 10

if nodes == 0:
    traininglr = 0.5
elif nodes <= 1:
    traininglr = 0.05
elif nodes <= 5:
    traininglr = 0.005
elif nodes <= 8:
    traininglr = 0.0001    
else:
    traininglr = 0.00005
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
from taskspec.utils_2 import N_sample, ker_eps_rho, n_dim_u, n_dim, datapath, N_sample, f_true, scale, X_alg, NT, Nx, bounds, Efunc_all, Egrad_all, forward_map, x_ref
from taskspec.model import DenseNet, PolyModel
from taskspec.method import estimate_measure, estimate_loss, estimate_measure_weight, print_res_info
from scipy.optimize import minimize

outputfile = f'{datapath}/weights/weights_ts/E2_t{nodes}_{randomseed}_{init_id}.npz'

weight_const, w_const = 0.0, 0.5 # const weight for all data, const weight for all called points
print(f"weight_const: {weight_const}  w_const: {w_const}")
M_softmax = 10.0
print(f"M_softmax: {M_softmax}")

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

if nodes <= 5:
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
    u_tmp.assign(u_best*0.01)
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
result = minimize(Efunc, np.array(u_tmp).flatten(), jac=Egrad, bounds=bounds, method='L-BFGS-B', tol=1e-10, options={'maxiter': 1000})
u_tmp.assign(result.x[None,:,None])
u_tmp1 = u_tmp.numpy()

assert np.linalg.norm(u_tmp) > 0

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
assert gradientnorm < 1e-2

loss_train = np.sum(np.square(model(xdata) - ydata), axis=-1)
result = minimize(Efunc, np.array(u_tmp).flatten(), jac=Egrad, bounds=bounds, method='L-BFGS-B', tol=1e-10)
u_tmp.assign(result.x[None,:,None])
new_sample, x_traj = X_alg(model, x_initial, u_tmp)
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

#This is the true trajectory
def compare(weights=None):
    if weights is not None:
        model.set_weights(weights)
    x_traj_true = forward_map(f_true, x_initial, u_tmp)
    traj_loss = np.sum(np.square(x_traj_true - x_tracking))# real loss to optimize
    
    point_loss = np.sum(np.square(f_true(new_sample) - model(new_sample)), axis=-1)
    point_loss_l2 = np.mean(point_loss)
    point_loss_oo = np.sqrt(np.max(point_loss))
    
    new_loss_l2 = np.mean(new_loss)
    new_loss_oo = np.sqrt(np.max(new_loss))
    
    trainingmse = np.sqrt(tf.reduce_mean(tf.square(model(xdata) - ydata)))    
    
    return [traj_loss - best_traj_loss, 
            point_loss_l2, point_loss_oo, 
            new_loss_l2, new_loss_oo, 
            trainingmse, current_loss, current_loss_aftertraining,
            ]

# 0: Final target, unknown
# 1: true l2 loss on S(ft), unknown
# 2: true oo loss on S(ft), unknown
# 3: estimate l2 loss on S(ft), known
# 4: estimate oo loss on S(ft), known, l3n
# 5: training mse, known, l1n
# 6: current_loss_beforetraining, known, l4n before training
# 7: current_loss_aftertraining, known, l4n after training

# save best weights with lowest L3N and L4N loss
res = compare()
print(0, np.array(res))

weights2 = None
u_tmp2 = None
ind_l3 = 0
best_l3 = res[4]
weights_l3 = np.copy(weights1)
u_l3 = np.copy(u_tmp.numpy())

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

print("nodes:", nodes, "randomseed:", randomseed, "initial_id:", init_id)
print_res_info()
calculateloss = compare()
print(f"Init.", np.array(calculateloss), currentstep)
res = [calculateloss]

for i in range(MAX_ITER):
    start_time = time.time()
    last_weights = model.get_weights()
    last_u_tmp = u_tmp.numpy()
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
    result = minimize(Efunc, np.array(u_tmp).flatten(), jac=Egrad, bounds=bounds, method='L-BFGS-B', tol=1e-12, options={'maxiter': 1000})
    u_tmp.assign(result.x[None,:,None])
    new_sample, x_traj = X_alg(model, x_initial, u_tmp)
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
        u_tmp.assign(last_u_tmp)
        new_sample, x_traj = X_alg(model, x_initial, u_tmp)
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
        ind_l3 = i; best_l3 = calculateloss[4]; weights_l3 = model.get_weights(); u_l3 = np.copy(u_tmp.numpy())
    
    if i > Nterminals:
        if np.std(np.array(res)[-Nterminals:,4]) <= 1e-5:
            print("Converged because ", np.std(np.array(res)[-Nterminals:,4]))
            break
        if i > ind_l3 + 100 and res[-1][4] > res[-2][4]:
            print("Early stopping because ", i, ind_l3)
            break
    
weights2 = model.get_weights()
u_tmp2 = u_tmp.numpy()

np.savez(outputfile, **get_save_date())

