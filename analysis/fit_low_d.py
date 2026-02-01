from pathlib import Path
import analysis_utilities as au
import numpy as np
import pickle
import loading_utilities as lu
from matplotlib import pyplot as plt
import scipy.optimize as sio
import torch
import time


def get_cum_var(mat):
    _, S, _ = np.linalg.svd(mat)
    var = S ** 2
    cum_var = np.cumsum(var / var.sum())
    return cum_var

run_params = lu.get_run_params(param_name='../analysis_params/paper_figures.yml')

saved_run_folder = Path(run_params['saved_run_folder'])
model_folders = run_params['model_folders']
for k in model_folders:
    model_folders[k] = Path(model_folders[k])

# get the models
models = {}
posterior_dicts = {}
for mf in model_folders:
    model_file = open(saved_run_folder / model_folders[mf] / 'models' / 'model_trained.pkl', 'rb')
    models_in = pickle.load(model_file)
    model_file.close()

    models[mf] = au.normalize_model(models_in)[0]

    post_file = open(saved_run_folder / model_folders[mf] / 'posterior_test.pkl', 'rb')
    posterior_dicts[mf] = pickle.load(post_file)
    post_file.close()

W = models['unconstrained'].dynamics_weights
W_diag = np.diag(np.diag(W))
W_torch = torch.tensor(W)
n = W.shape[0]

W_zero_diag = W.copy()
W_zero_diag[np.eye(W_zero_diag.shape[0], dtype=bool)] = 0

zero_diag_cum_var = get_cum_var(W_zero_diag)
full_diag_cum_var = get_cum_var(W)


def outer_loss_svd_torch(p):
    w_diag_sub = W_torch - p * torch.eye(n)
    var_vals = torch.linalg.svdvals(w_diag_sub)**2
    var_cumsum = torch.cumsum(var_vals / var_vals.sum(), dim=0)
    auc = var_cumsum.sum() / n

    return -auc


p_hat = au.scipy_minimize_with_grad(outer_loss_svd_torch, 1).x
fit_diag_cum_var = get_cum_var(W - p_hat * np.eye(n))

###### low d approx

def loss_low_d_torch(p):
    R = p[: n * d].reshape((n, d))
    C = p[n * d :].reshape((n, d))

    diag_est = torch.diag(torch.diag(W_torch)) - torch.diag(torch.diag(R @ C.T))
    W_tilde = R @ C.T + diag_est

    return torch.sum((W_torch - W_tilde)**2)

def loss_low_d_torch(p):
    R = p[: n * d].reshape((n, d))
    C = p[n * d :].reshape((n, d))

    diag_est = torch.diag(torch.diag(W_torch)) - torch.diag(torch.diag(R @ C.T))
    W_tilde = R @ C.T + diag_est

    return torch.sum((W_torch - W_tilde)**2)

min_dim = 1
max_dim = 20
rng_seed = 0
rng = np.random.default_rng(rng_seed)
low_d_cum_var = np.zeros(max_dim - min_dim + 1)
init_std = 0.01
init_diag = 0.9

R_hat = init_std * rng.standard_normal((n, min_dim-1))
C_hat = init_std * rng.standard_normal((n, min_dim-1))

start = time.time()
for di, d in enumerate(range(min_dim, max_dim + 1)):
    R_0 = init_std * rng.standard_normal((n, d))
    C_0 = init_std * rng.standard_normal((n, d))
    R_0[:, :-1] = R_hat
    C_0[:, :-1] = C_hat
    p_0 = np.concatenate((R_0.reshape(-1), C_0.reshape(-1)))

    p_hat = au.scipy_minimize_with_grad(loss_low_d_torch, p_0).x
    R_hat = p_hat[: n * d].reshape((n, d))
    C_hat = p_hat[n * d:].reshape((n, d))

    diag_est = np.diag(np.diag(W)) - np.diag(np.diag(R_hat @ C_hat.T))

    W_hat = R_hat @ C_hat.T + diag_est
    error = np.sum((W - W_hat)**2)
    low_d_cum_var[di] = 1 - error / np.sum((W - W_diag)**2)

    print('run', di + 1, '/', max_dim, 'completed')
    print(time.time() - start, 's elapsed')
    debug=1

plt.figure()
plt.plot(full_diag_cum_var, label='full matrix')
plt.plot(fit_diag_cum_var, label='matrix - fit diagonal')
plt.plot(zero_diag_cum_var, label='diagonal = 0')
plt.plot(low_d_cum_var, label='low_d')
plt.xlabel('singular values')
plt.ylabel('cumulative variance')
plt.legend()
plt.ylim((0, 1))

plt.show()

a=1
