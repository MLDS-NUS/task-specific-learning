import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from taskspec.utils import find_project_root
datapath = find_project_root() + "/examples/Eg2"

ker_eps_rho = 0.01
n_dim_x, n_dim_u, n_dim_xx = 2, 1, 4
n_dim = n_dim_x + n_dim_u
g=9.81; l=10; kR=kL=0.01
para = {'gol': g/l, 'kLol': kL/l, 'kR': kR}
def f_true(xu, para=para):
    #Input N*3 array, output N array
    assert xu.shape[1] == n_dim
    x1, x2, u = xu[:,0], xu[:,1], xu[:,2]
    return (-para['gol']*tf.sin(x1) - u*tf.cos(x1) - para['kLol']*x2*x2*tf.sign(x2) - para['kR']*tf.sign(x2))[:,None]

x_range = (0, 2 * np.pi)
y_range = (-5., 5.)
u_range = (-10., 10.)
N_sample = 100000

scale = [x_range[1], y_range[1], u_range[1]]

T = 1.0
NT = 20
tau = T/NT
Nstep = 10 # every tau
tau_N = tau/Nstep
Nx = 1

x_ref = np.zeros((10, NT+1, n_dim_xx))
x_ref[0] = np.linspace(np.array([[np.pi + 1.0, -1.0, 0., 0.]]), 
                                np.array([[np.pi, 0, 0, 0]]), 
                                NT+1, axis=1) 

bounds = [(u_range[0], u_range[1])] * NT

def forward_map_end(model, x_initial, u_tmp):
    #Input N*4 array, N*Nt*1 array    
    x12 = tf.constant(x_initial[:, :n_dim_x], dtype=tf.float64)
    x34 = tf.constant(x_initial[:, n_dim_x:], dtype=tf.float64)
    for i in range(NT):
        for _ in range(Nstep):
            rhs = model(tf.concat([x12, u_tmp[:,i]], axis=1))[:,0]
            x12 = x12 + tau_N * tf.stack([x12[:,1], rhs], axis=1)
        x34 = x34 + tau * tf.stack([x34[:,1]+.5*u_tmp[:,i,0]*tau, u_tmp[:,i,0]], axis=1)
    return tf.concat([x12,x34], axis=1) #N*4 array

def opt_loss_end(model, x_initial, u_tmp, x_target):
    x_traj = forward_map_end(model, x_initial, u_tmp)
    return tf.reduce_sum(tf.square(x_traj - x_target))  

def Efunc_end(u, model, x_initial, x_target):
    return opt_loss_end(f_true, x_initial, u[None,:,None], x_target) #+ tf.reduce_sum(tf.square(u)) * eps

def Egrad_end(u, model, x_initial, x_target):
    u_tmp = tf.Variable(u[None,:,None], dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(u_tmp)
        loss = opt_loss_end(model, x_initial, u_tmp, x_target) #+ tf.reduce_sum(tf.square(u_tmp)) * eps
    grads = tape.gradient(loss, u_tmp)
    return grads.numpy()[0,:,0]


def find_optimal_control(model, x_initial, u_tmp, x_target, lr=1e-2, epochs=100, printflag=False):
    u_tmp = tf.Variable(u_tmp, dtype=tf.float64)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            tape.watch(u_tmp)
            loss = opt_loss_end(model, x_initial, u_tmp, x_target)            
        grads = tape.gradient(loss, u_tmp)
        uTnew = tf.clip_by_value(u_tmp - lr*grads, u_range[0], u_range[1])
        
        if tf.norm(uTnew - u_tmp) < 1e-6:
            break
        u_tmp.assign(uTnew)
        if i % (epochs//10) == 0:
            print(i, 'loss:', loss.numpy()) if printflag else None
    return u_tmp


def forward_map(model, x_initial, u_tmp):
    #Input N*4 array, N*Nt*1 array
    x_traj = x_initial[:,None,:]
    x12 = tf.constant(x_initial[:, :n_dim_x], dtype=tf.float64)
    x34 = tf.constant(x_initial[:, n_dim_x:], dtype=tf.float64)
    for i in range(NT):
        for _ in range(Nstep):
            rhs = model(tf.concat([x12, u_tmp[:,i]], axis=1))[:,0]
            x12 = x12 + tau_N * tf.stack([x12[:,1], rhs], axis=1)
        x34 = x34 + tau * tf.stack([x34[:,1]+.5*u_tmp[:,i,0]*tau, u_tmp[:,i,0]], axis=1)
        x_traj = tf.concat([x_traj, tf.concat([x12, x34], axis=1)[:,None,:]], axis=1)
    return x_traj #N*Nt+1*4 array

def opt_loss(model, x_initial, u_tmp, x_tracking):
    x_traj = forward_map(model, x_initial, u_tmp)
    return tf.reduce_sum(tf.square(x_traj - x_tracking))

def Efunc_all(u, model, x_initial, x_tracking_target):
    return opt_loss(model, x_initial, u[None,:,None], x_tracking_target)

def Egrad_all(u, model, x_initial, x_tracking_target):
    u_tmp = tf.Variable(u[None,:,None], dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(u_tmp)
        loss = opt_loss(model, x_initial, u_tmp, x_tracking_target)
    grads = tape.gradient(loss, u_tmp)
    return grads.numpy()[0,:,0]


def X_alg(model, x_initial, u_tmp):
    x_traj = np.zeros((Nx, NT+1, n_dim_xx))
    x_traj[:, 0] = x_initial[:,None,:]
    xu_traj = np.zeros((Nx, NT*Nstep, n_dim))
    x12 = tf.constant(x_initial[:, :n_dim_x], dtype=tf.float64)
    x34 = tf.constant(x_initial[:, n_dim_x:], dtype=tf.float64)
    for i in range(NT):
        for _ in range(Nstep):
            xu = tf.concat([x12, u_tmp[:,i]], axis=1)
            xu_traj[:,i*Nstep + _, :] = xu.numpy()
            rhs = model(xu)[:,0]
            x12 = x12 + tau_N * tf.stack([x12[:,1], rhs], axis=1)
        x34 = x34 + tau * tf.stack([x34[:,1]+.5*u_tmp[:,i,0]*tau, u_tmp[:,i,0]], axis=1)
        x_traj[:,i+1,:] = tf.concat([x12, x34], axis=1).numpy()[:,None,:]
    return xu_traj.reshape((-1, n_dim)), x_traj #give xu array 

