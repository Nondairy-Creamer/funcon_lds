from pathlib import Path
from matplotlib import pyplot as plt
import analysis_utilities as au
import numpy as np
import pickle
import wormneuroatlas as wa


def get_irms(data_in):
    irfs, irfs_sem, irfs_all = au.get_impulse_response_function(data_in, inputs_test, window=window,
                                            sub_pre_stim=sub_pre_stim, return_pre=True)

    irms = np.nanmean(irfs[-window[0]:], axis=0)
    irms[np.eye(irms.shape[0], dtype=bool)] = np.nan

    num_neurons = irfs.shape[1]
    num_stim = np.zeros((num_neurons, num_neurons))
    for ni in range(num_neurons):
        for nj in range(num_neurons):
            resp_to_stim = irfs_all[ni][:, -window[0]:, nj]
            num_obs_when_stim = np.sum(np.mean(~np.isnan(resp_to_stim), axis=1) >= 0.5)
            num_stim[nj, ni] += num_obs_when_stim

    irms[num_stim < required_num_stim] = np.nan

    return irms

required_num_stim = 5
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

data_corr = data_train['data_corr']

data_test_file = open(path_dense / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

inputs_test = data_test['inputs']

# get synapse signs
cell_ids = data_test['cell_ids']
watlas = wa.NeuroAtlas()
chem_synapse_sign = watlas.get_anatomical_connectome(signed=True)
atlas_ids = list(watlas.neuron_ids)
atlas_ids[atlas_ids.index('AWCON')] = 'AWCR'
atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCL'
atlas_inds = [atlas_ids.index(i) for i in cell_ids]
chem_synapse_sign = chem_synapse_sign[np.ix_(atlas_inds, atlas_inds)]
nan_mask = chem_synapse_sign == 0

model_weights = model_sparse.dynamics_weights[:model_sparse.dynamics_dim, :model_sparse.dynamics_dim]

model_weights = (model_weights > np.std(model_weights)/8).astype(float) - (model_weights < -np.std(model_weights)/8).astype(float)
chem_synapse_sign = (chem_synapse_sign > np.std(chem_synapse_sign)/8).astype(float) - (chem_synapse_sign < -np.std(chem_synapse_sign)/8).astype(float)
chem_synapse_sign[nan_mask] = np.nan

a = au.nan_corr(chem_synapse_sign, model_weights)[0]

# get the IRMs of the models and data
# dense
model_sampled_dense = post_dense['model_sampled']
model_irm_dense = get_irms(model_sampled_dense)

# dense random
if force_calc:
    model_sampled_dense_rand = []
    for i in inputs_test:
        model_sampled_dense_rand.append(model_dense_rand.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])

    dense_sampled_file = open(path_dense_rand / 'sampled_dense_rand.pkl', 'wb')
    pickle.dump(model_sampled_dense_rand, dense_sampled_file)
    dense_sampled_file.close()
else:
    dense_sampled_file = open(path_dense_rand / 'sampled_dense_rand.pkl', 'rb')
    model_sampled_dense_rand = pickle.load(dense_sampled_file)
    dense_sampled_file.close()

model_irm_dense_rand = get_irms(model_sampled_dense_rand)

# sparse
model_sampled_sparse = post_sparse['model_sampled']
model_irm_sparse = get_irms(model_sampled_sparse)

# sparse random
if force_calc:
    model_sampled_sparse_rand = []
    for i in inputs_test:
        model_sampled_sparse_rand.append(model_sparse_rand.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])

    sparse_sampled_file = open(path_sparse_rand / 'sampled_sparse_rand.pkl', 'wb')
    pickle.dump(model_sampled_sparse_rand, sparse_sampled_file)
    sparse_sampled_file.close()
else:
    sparse_sampled_file = open(path_sparse_rand / 'sampled_sparse_rand.pkl', 'rb')
    model_sampled_sparse_rand = pickle.load(sparse_sampled_file)
    sparse_sampled_file.close()

model_irm_sparse_rand = get_irms(model_sampled_sparse_rand)

# test data
emissions_test = data_test['emissions']
data_irm_test = get_irms(emissions_test)

# process the IRMs
nan_mask = np.isnan(model_irm_dense) | np.isnan(model_irm_dense_rand) | \
           np.isnan(model_irm_sparse) | np.isnan(model_irm_sparse_rand) | \
           np.isnan(data_irm_test) | np.isnan(data_corr)

model_irm_dense[nan_mask] = np.nan
model_irm_dense_rand[nan_mask] = np.nan
model_irm_sparse[nan_mask] = np.nan
model_irm_sparse_rand[nan_mask] = np.nan
data_irm_test[nan_mask] = np.nan
data_corr[nan_mask] = np.nan

dense_to_sparse = au.nan_corr(model_irm_dense, model_irm_sparse)[0]
sparse_to_sparse_rand = au.nan_corr(model_irm_sparse, model_irm_sparse_rand)[0]

corr_to_measured, corr_to_measured_ci = au.nan_corr(data_irm_test, data_corr)
dense_to_measured, dense_to_measured_ci = au.nan_corr(data_irm_test, model_irm_dense)
dense_rand_to_measured, dense_rand_to_measured_ci = au.nan_corr(data_irm_test, model_irm_dense_rand)
sparse_to_measured, sparse_to_measured_ci = au.nan_corr(data_irm_test, model_irm_sparse)
sparse_rand_to_measured, sparse_rand_to_measured_ci = au.nan_corr(data_irm_test, model_irm_sparse_rand)

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
# plot_x = np.arange(5)
# y_val = np.array([dense_to_measured, corr_to_measured, dense_rand_to_measured, sparse_to_measured, sparse_rand_to_measured])
# y_val_ci = np.stack([dense_to_measured_ci, corr_to_measured_ci, dense_rand_to_measured_ci, sparse_to_measured_ci, sparse_rand_to_measured_ci]).T
plot_x = np.arange(4)
y_val = np.array([dense_to_measured, dense_rand_to_measured, sparse_to_measured, sparse_rand_to_measured])
y_val_ci = np.stack([dense_to_measured_ci, dense_rand_to_measured_ci, sparse_to_measured_ci, sparse_rand_to_measured_ci]).T
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
# plt.xticks(plot_x, labels=['dense dynamics matrix', 'data correlation', 'scrambled cell labels', 'anatomy constrained\n dynamics matrix', 'scrambled anatomy'])
plt.xticks(plot_x, labels=['full model', 'full model\n+ scrambled labels', 'anatomy constrained', 'scrambled anatomy\nconstrained'])
plt.ylabel('correlation to measured IRMs')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

chem_synapse, gap_junction, pep = au.load_anatomical_data(cell_ids)
weights = np.abs(model_sparse.dynamics_weights[:model_sparse.dynamics_dim, :])
weights = np.split(weights, model_sparse.dynamics_lags, axis=1)
score, score_ci, left_recon, right_recon = au.compare_matrix_sets(chem_synapse, weights)

left_recon = left_recon.reshape(-1)
right_recon = right_recon.reshape(-1)

zero_loc = (left_recon != 0) & (right_recon != 0)
left_recon = left_recon[zero_loc]

right_recon = np.abs(right_recon[zero_loc])
score = au.nan_corr(left_recon, right_recon)[0]

p = np.polyfit(left_recon, right_recon, 1)

plt.figure()
plt.scatter(left_recon.reshape(-1), np.abs(right_recon.reshape(-1)))
x_range = plt.xlim()
y = p[0] * np.array(x_range) + p[1]
plt.plot(x_range, y, color='k')
plt.xlabel('synapse count')
plt.ylabel('model weights')
plt.title(str(score))

plt.savefig('/home/mcreamer/Documents/google_drive/leifer_pillow_lab/talks/2023_11_15_CosyneAbstract/anatomy_weights_vs_model.pdf')

plt.show()

a=1









