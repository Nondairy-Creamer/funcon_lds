from pathlib import Path
from matplotlib import pyplot as plt
import pickle
import analysis_utilities as au
import numpy as np

sub_pre_stim = True
window = [-60, 120]
path_dense = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0/20230914_201648')
path_sparse = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap/20231030_200902')

# get the dense model files
model_dense_file = open(path_dense / 'models' / 'model_trained.pkl', 'rb')
model_dense = pickle.load(model_dense_file)
model_dense_file.close()

post_dense_file = open(path_dense / 'posterior_test.pkl', 'rb')
post_dense = pickle.load(post_dense_file)
post_dense_file.close()

# get the sparse model files
model_sparse_file = open(path_sparse / 'models' / 'model_trained.pkl', 'rb')
model_sparse = pickle.load(model_sparse_file)
model_sparse_file.close()

post_sparse_file = open(path_sparse / 'posterior_test.pkl', 'rb')
post_sparse = pickle.load(post_sparse_file)
post_sparse_file.close()

# get the data
data_train_file = open(path_dense / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

data_test_file = open(path_dense / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

inputs_test = data_test['inputs']

# get the IRFs of the models and data
# dense
model_sampled_dense = post_dense['model_sampled']
# get the impulse response functions
model_irf_dense, model_irf_sem_dense, model_irf_all_dense = \
    au.get_impulse_response_function(model_sampled_dense, inputs_test, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

model_irm_dense = np.mean(model_irf_dense[-window[0]:], axis=0)

# sparse
model_sampled_sparse = post_sparse['model_sampled']
inputs_sparse = data_test['inputs']
# get the impulse response functions
model_irf_sparse, model_irf_sem_sparse, model_irf_all_sparse = \
    au.get_impulse_response_function(model_sampled_sparse, inputs_test, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

model_irm_sparse = np.mean(model_irf_sparse[-window[0]:], axis=0)

# data
emissions_train = data_train['emissions']
inputs_train = data_train['inputs']
# get the impulse response functions
data_irf_train, data_irf_sem_train, data_irf_all_train = \
    au.get_impulse_response_function(emissions_train, inputs_train, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

data_irm_train = np.mean(data_irf_train[-window[0]:], axis=0)


emissions_test = data_test['emissions']
# get the impulse response functions
data_irf_test, data_irf_sem_test, data_irf_all_test = \
    au.get_impulse_response_function(emissions_test, inputs_test, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

data_irm_test = np.mean(data_irf_test[-window[0]:], axis=0)

model_irm_sparse[np.eye(model_irm_sparse.shape[0], dtype=bool)] = np.nan
model_irm_dense[np.eye(model_irm_dense.shape[0], dtype=bool)] = np.nan
data_irm_test[np.eye(data_irm_test.shape[0], dtype=bool)] = np.nan
data_irm_train[np.eye(data_irm_train.shape[0], dtype=bool)] = np.nan

nan_mask = np.isnan(model_irm_sparse) | np.isnan(model_irm_dense) | np.isnan(data_irm_test) | np.isnan(data_irm_train)

model_irm_sparse[nan_mask] = np.nan
model_irm_dense[nan_mask] = np.nan
data_irm_test[nan_mask] = np.nan
data_irm_train[nan_mask] = np.nan

sparse_to_dense = au.nan_corr(model_irm_sparse, model_irm_dense)[0]
sparse_to_measured = au.nan_corr(model_irm_sparse, data_irm_test)[0]
dense_to_measured = au.nan_corr(model_irm_dense, data_irm_test)[0]
train_to_test = au.nan_corr(data_irm_train, data_irm_test)[0]

all_nan = np.mean(np.isnan(data_irm_test), axis=0) > 0.75
data_irm_test_plot = data_irm_test.copy()
data_irm_test_plot = data_irm_test_plot[~all_nan, :][:, ~all_nan]
data_irm_train_plot = data_irm_train.copy()
data_irm_train_plot = data_irm_train_plot[~all_nan, :][:, ~all_nan]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(data_irm_train_plot, interpolation='nearest')
plt.subplot(1, 2, 2)
plt.imshow(data_irm_test_plot, interpolation='nearest')

x = data_irm_train.reshape(-1)
y = data_irm_test.reshape(-1)
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]
plt.figure()
plt.scatter(x, y, s=1)

plt.figure()
plt.bar(np.arange(4), [sparse_to_dense, sparse_to_measured, dense_to_measured, train_to_test])
plt.show()











