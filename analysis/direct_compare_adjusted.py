from pathlib import Path
from matplotlib import pyplot as plt
import pickle
import analysis_utilities as au
import numpy as np
import pickle

# Testing the adjusted frac explainable variance
# m = 100
# n = 50
#
# rng = np.random.default_rng(0)
# true_mean = rng.standard_normal(m)
# noise_std = 1/10
# nan_mask = rng.uniform(size=(m, n)) < 0.2
#
# measured = true_mean[:, None] + noise_std * rng.standard_normal((m, n))
# measured[nan_mask] = np.nan
# model = true_mean[:, None] + noise_std * rng.standard_normal((m, n))
# model[nan_mask] = np.nan
#
# c = au.frac_explainable_var(measured, model)


def get_irms(data_in):
    irms = np.zeros((data_in[0].shape[1]**2, len(data_in)))

    for i in range(len(data_in)):
        irf_out = au.get_impulse_response_function([data_in[i]], [inputs_test[i]], window=window,
                                                   sub_pre_stim=sub_pre_stim, return_pre=True)[0]

        irm_out = np.nanmean(irf_out[-window[0]:], axis=0)
        irm_out[np.eye(irm_out.shape[0], dtype=bool)] = np.nan

        irms[:, i] = irm_out.reshape(-1)

    return irms

force_calc = False
sub_pre_stim = True
window = [-60, 120]
path_dense = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0/20230914_201648')
path_dense_rand = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_randCellID/20231114_144009/')
path_sparse = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap/20231030_200902')
path_sparse_rand = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap_rand/20231114_125113/')

# get the dense model files
model_dense_file = open(path_dense / 'models' / 'model_trained.pkl', 'rb')
model_dense = pickle.load(model_dense_file)
model_dense_file.close()

post_dense_file = open(path_dense / 'posterior_test.pkl', 'rb')
post_dense = pickle.load(post_dense_file)
post_dense_file.close()

# get the dense rand model files
model_dense_rand_file = open(path_dense_rand / 'models' / 'model_trained.pkl', 'rb')
model_dense_rand = pickle.load(model_dense_rand_file)
model_dense_rand_file.close()
#
# post_dense_rand_file = open(path_dense_rand / 'posterior_test.pkl', 'rb')
# post_dense_rand = pickle.load(post_dense_rand_file)
# post_dense_rand_file.close()

# get the sparse model files
model_sparse_file = open(path_sparse / 'models' / 'model_trained.pkl', 'rb')
model_sparse = pickle.load(model_sparse_file)
model_sparse_file.close()

post_sparse_file = open(path_sparse / 'posterior_test.pkl', 'rb')
post_sparse = pickle.load(post_sparse_file)
post_sparse_file.close()

# get the sparse random model files
model_sparse_rand_file = open(path_sparse_rand / 'models' / 'model_trained.pkl', 'rb')
model_sparse_rand = pickle.load(model_sparse_rand_file)
model_sparse_rand_file.close()

# post_sparse_rand_file = open(path_sparse_rand / 'posterior_test.pkl', 'rb')
# post_sparse_rand = pickle.load(post_sparse_rand_file)
# post_sparse_rand_file.close()

# get the data
data_train_file = open(path_dense / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

data_test_file = open(path_dense / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

inputs_test = data_test['inputs']

# get the IRMs of the models and data
# dense
model_sampled_dense = post_dense['model_sampled']
model_irm_dense = get_irms(model_sampled_dense)

# dense random
# model_sampled_dense_rand = []
# for i in inputs_test:
#     model_sampled_dense_rand.append(model_dense_rand.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])
#
# model_irm_dense_rand = get_irms(model_sampled_dense_rand)
model_irm_dense_rand = model_irm_dense.copy()

# sparse
model_sampled_sparse = post_sparse['model_sampled']
model_irm_sparse = get_irms(model_sampled_sparse)

# sparse random
if force_calc:
    model_sampled_sparse_rand = []
    for i in inputs_test:
        model_sampled_sparse_rand.append(model_sparse_rand.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])

    sparse_sampled_file = open('/home/mcreamer/Documents/python/funcon_lds/analysis/sampled_sparse.pkl', 'wb')
    pickle.dump(model_sampled_sparse_rand, sparse_sampled_file)
    sparse_sampled_file.close()
else:
    sparse_sampled_file = open('/home/mcreamer/Documents/python/funcon_lds/analysis/sampled_sparse.pkl', 'rb')
    model_sampled_sparse_rand = pickle.load(sparse_sampled_file)
    sparse_sampled_file.close()

model_irm_sparse_rand = get_irms(model_sampled_sparse_rand)

# test data
emissions_test = data_test['emissions']
data_irm_test = get_irms(emissions_test)

# process the IRMs
required_num_stim = 5
non_nan_mask = (np.sum(~np.isnan(model_irm_dense), axis=1) >= required_num_stim) & (np.sum(~np.isnan(model_irm_dense_rand), axis=1) >= required_num_stim) & \
               (np.sum(~np.isnan(model_irm_sparse), axis=1) >= required_num_stim) & (np.sum(~np.isnan(model_irm_sparse_rand), axis=1) >= required_num_stim) & \
               (np.sum(~np.isnan(data_irm_test), axis=1) >= required_num_stim)

# model_irm_dense[nan_mask, :] = np.nan
# model_irm_dense_rand[nan_mask, :] = np.nan
# model_irm_sparse[nan_mask, :] = np.nan
# model_irm_sparse_rand[nan_mask, :] = np.nan
# data_irm_test[nan_mask, :] = np.nan

model_irm_dense = model_irm_dense[non_nan_mask, :]
model_irm_dense_rand = model_irm_dense_rand[non_nan_mask, :]
model_irm_sparse = model_irm_sparse[non_nan_mask, :]
model_irm_sparse_rand = model_irm_sparse_rand[non_nan_mask, :]
data_irm_test = data_irm_test[non_nan_mask, :]

dense_to_sparse = au.nan_corr(model_irm_dense, model_irm_sparse)[0]
sparse_to_sparse_rand = au.nan_corr(model_irm_sparse, model_irm_sparse_rand)[0]

dense_to_measured = au.frac_explainable_var(data_irm_test, model_irm_dense)
dense_rand_to_measured = au.frac_explainable_var(data_irm_test, model_irm_dense_rand)
sparse_to_measured = au.frac_explainable_var(data_irm_test, model_irm_sparse)
sparse_rand_to_measured = au.frac_explainable_var(data_irm_test, model_irm_sparse_rand)

# plotting
plt.figure()
plot_x = np.arange(2)
plt.bar(plot_x, [dense_to_sparse, sparse_to_sparse_rand])
plt.xticks(plot_x, labels=['sparse_to_dense', 'sparse_to_sparse rand'])
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

plt.figure()
plot_x = np.arange(4)
plt.bar(plot_x, [dense_rand_to_measured, dense_to_measured, sparse_to_measured, sparse_rand_to_measured])
plt.xticks(plot_x, labels=['dense_rand_to_measured', 'dense_to_measured', 'sparse_to_measured', 'sparse_rand_to_measured'])
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

plt.show()

a=1









