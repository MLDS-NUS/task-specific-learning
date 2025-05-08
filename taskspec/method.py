import numpy as np

def estimate_measure(x_call, x_data, ker_eps=0.1, method=False):
    # estimate measure on x_call based on x_data
    # x_call.shape = (n_call, dim)
    # x_data.shape = (n_data, dim)
    n_call = x_call.shape[0]
    if method:        
        px = np.zeros((n_call, ))
        for i in range(n_call):
            e = np.sum(np.square(x_data - x_call[i]), axis=-1) # (n_data,)
            px[i] = np.sum(np.exp( -e / ker_eps)) #()
    else:
        e = x_data[None,:,:] - x_call[:,None,:] # (n_call, n_data, dim)
        e = np.sum(np.square(e), axis=-1) # (n_call, n_data)
        e = np.clip(e/ker_eps, 0, 100.0)
        exp_e = np.exp( -e ) + 1e-12
        px = np.mean(exp_e, axis=-1, keepdims=False) # (n_call,1)
    px = px / np.mean(px)
    return px # (n_call,1)


def estimate_loss(x_call, x_data, rho_data, data_loss, ker_eps=0.1):
    # estimate loss on x_call based on data_loss on x_data wtih rho_data measure
    # x_call.shape = (n_call, dim)
    # x_data.shape = (n_data, dim)
    # rho_data.shape = (n_data, )
    # data_loss.shape = (n_data, )
    e = x_data[None,:,:] - x_call[:,None,:] # (n_call, n_data, dim)
    e = np.sum(np.square(e), axis=-1) # (n_call, n_data)
    e = np.clip(e/ker_eps, 0, 1000.0)
    exp_e = np.exp( -e ) + 1e-12
    w_x = exp_e / rho_data #+ 1e-8 # (n_call, n_data)
    w_x = w_x / np.mean(w_x, axis=-1, keepdims=True)  # normalize for each point # (n_call, n_data)
    px = np.mean(w_x * data_loss, axis=-1, keepdims=False) # (n_call,)
    return px # (n_call,)


def estimate_measure_weight(x_call, x_data, weight_data, ker_eps=0.1):
    # estimate weights on x_call based on x_data with weight_data
    # x_call.shape = (n_call, dim)
    # x_data.shape = (n_data, dim)
    # rho_data.shape = (n_data, )
    # data_loss.shape = (n_data, )
    e = x_data[None,:,:] - x_call[:,None,:] # (n_call, n_data, dim)
    e = np.sum(np.square(e), axis=-1) # (n_call, n_data)
    e = np.clip(e/ker_eps, 0, 1000.0)
    exp_e = np.exp( -e ) + 1e-12
    
    px = np.mean(exp_e * weight_data, axis=-1, keepdims=False) # (n_call,)
    px = px / np.mean(px)
    return px  # (n_call,)

def print_res_info():
    print("======     0    ,     1    ,     2    ,     3    ,     4    ,     5    ,     6    ,     7    ")
    print(" Iter  traj_loss, ptloss_l2, ptloss_oo, esloss_l2, esloss_oo, train_mse, Optbefore, Opt_after")
    print("       -Target--, -Unknown-, -Unknown-, --Known--, --Known--, --Known--, --Known--, --Known--")
