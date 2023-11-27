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
path_dense_randID = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_randCellID/20231114_144009/')
path_sparse = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap/20231030_200902')
path_sparse_randA = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap_rand/20231114_125113/')

# get the dense model files
model_dense_file = open(path_dense / 'models' / 'model_trained.pkl', 'rb')
model_dense = pickle.load(model_dense_file)
model_dense_file.close()

post_dense_file = open(path_dense / 'posterior_test.pkl', 'rb')
post_dense = pickle.load(post_dense_file)
post_dense_file.close()

# get the dense rand model files
model_dense_randID_file = open(path_dense_randID / 'models' / 'model_trained.pkl', 'rb')
model_dense_randID = pickle.load(model_dense_randID_file)
model_dense_randID_file.close()
#
# post_dense_randID_file = open(path_dense_randID / 'posterior_test.pkl', 'rb')
# post_dense_randID = pickle.load(post_dense_randID_file)
# post_dense_randID_file.close()

# get the sparse model files
model_sparse_file = open(path_sparse / 'models' / 'model_trained.pkl', 'rb')
model_sparse = pickle.load(model_sparse_file)
model_sparse_file.close()

post_sparse_file = open(path_sparse / 'posterior_test.pkl', 'rb')
post_sparse = pickle.load(post_sparse_file)
post_sparse_file.close()

# get the sparse random model files
model_sparse_randA_file = open(path_sparse_randA / 'models' / 'model_trained.pkl', 'rb')
model_sparse_randA = pickle.load(model_sparse_randA_file)
model_sparse_randA_file.close()

# post_sparse_randA_file = open(path_sparse_randA / 'posterior_test.pkl', 'rb')
# post_sparse_randA = pickle.load(post_sparse_randA_file)
# post_sparse_randA_file.close()

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

model_corr_dense = np.identity(model_dense.dynamics_dim_full)
for i in range(1000):
    model_corr_dense = model_dense.dynamics_weights @ model_corr_dense @ model_dense.dynamics_weights.T +\
                       model_dense.dynamics_cov
model_corr_dense = model_corr_dense[:model_dense.dynamics_dim, :model_dense.dynamics_dim]

# dense random
if force_calc:
    model_sampled_dense_randID = []
    for i in inputs_test:
        model_sampled_dense_randID.append(model_dense_randID.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])

    dense_sampled_file = open(path_dense_randID / 'sampled_dense_rand.pkl', 'wb')
    pickle.dump(model_sampled_dense_randID, dense_sampled_file)
    dense_sampled_file.close()
else:
    dense_sampled_file = open(path_dense_randID / 'sampled_dense_rand.pkl', 'rb')
    model_sampled_dense_randID = pickle.load(dense_sampled_file)
    dense_sampled_file.close()

model_irm_dense_randID = get_irms(model_sampled_dense_randID)

model_corr_dense_randID = np.identity(model_dense_randID.dynamics_dim_full)
for i in range(1000):
    model_corr_dense_randID = model_dense_randID.dynamics_weights @ model_corr_dense_randID @ model_dense_randID.dynamics_weights.T \
                              + model_dense_randID.dynamics_cov
model_corr_dense_randID = model_corr_dense_randID[:model_dense_randID.dynamics_dim, :model_dense_randID.dynamics_dim]

# sparse
model_sampled_sparse = post_sparse['model_sampled']
model_irm_sparse = get_irms(model_sampled_sparse)

model_corr_sparse = np.identity(model_sparse.dynamics_dim_full)
for i in range(1000):
    model_corr_sparse = model_sparse.dynamics_weights @ model_corr_sparse @ model_sparse.dynamics_weights.T + model_sparse.dynamics_cov
model_corr_sparse = model_corr_sparse[:model_sparse.dynamics_dim, :model_sparse.dynamics_dim]

# sparse random
if force_calc:
    model_sampled_sparse_randA = []
    for i in inputs_test:
        model_sampled_sparse_randA.append(model_sparse_randA.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])

    sparse_sampled_file = open(path_sparse_randA / 'sampled_sparse_rand.pkl', 'wb')
    pickle.dump(model_sampled_sparse_randA, sparse_sampled_file)
    sparse_sampled_file.close()
else:
    sparse_sampled_file = open(path_sparse_randA / 'sampled_sparse_rand.pkl', 'rb')
    model_sampled_sparse_randA = pickle.load(sparse_sampled_file)
    sparse_sampled_file.close()

model_irm_sparse_randA = get_irms(model_sampled_sparse_randA)

model_corr_sparse_randA = np.identity(model_sparse_randA.dynamics_dim_full)
for i in range(1000):
    model_corr_sparse_randA = model_sparse_randA.dynamics_weights @ model_corr_sparse_randA @ model_sparse_randA.dynamics_weights.T \
                              + model_sparse_randA.dynamics_cov
model_corr_sparse_randA = model_corr_sparse_randA[:model_sparse_randA.dynamics_dim, :model_sparse_randA.dynamics_dim]

# test data
emissions_test = data_test['emissions']
data_irm_test = get_irms(emissions_test)

# process the IRMs
nan_mask = np.isnan(model_irm_dense) | np.isnan(model_irm_dense_randID) | \
           np.isnan(model_irm_sparse) | np.isnan(model_irm_sparse_randA) | \
           np.isnan(data_irm_test) | np.isnan(data_corr)

model_irm_dense[nan_mask] = np.nan
model_irm_dense_randID[nan_mask] = np.nan
model_irm_sparse[nan_mask] = np.nan
model_irm_sparse_randA[nan_mask] = np.nan
data_irm_test[nan_mask] = np.nan
data_corr[nan_mask] = np.nan

dense_to_sparse = au.nan_corr(model_irm_dense, model_irm_sparse)[0]
sparse_to_sparse_randA = au.nan_corr(model_irm_sparse, model_irm_sparse_randA)[0]

corr_to_measured_irm_irm, corr_to_measured_irm_ci = au.nan_corr(data_irm_test, data_corr)
dense_to_measured_irm, dense_to_measured_irm_ci = au.nan_corr(data_irm_test, model_irm_dense)
dense_randID_to_measured_irm, dense_randID_to_measured_irm_ci = au.nan_corr(data_irm_test, model_irm_dense_randID)
sparse_to_measured_irm, sparse_to_measured_irm_ci = au.nan_corr(data_irm_test, model_irm_sparse)
sparse_randA_to_measured_irm, sparse_randA_to_measured_irm_ci = au.nan_corr(data_irm_test, model_irm_sparse_randA)

plt.figure()
# plot_x = np.arange(5)
plot_x = np.arange(3)
y_val = np.array([sparse_to_measured_irm, dense_randID_to_measured_irm, sparse_randA_to_measured_irm])
y_val_ci = np.stack([sparse_to_measured_irm_ci, dense_randID_to_measured_irm_ci, sparse_randA_to_measured_irm_ci]).T
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'scrambled anatomy\nconstrained'])
plt.ylabel('similarity to measured IRMs')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

plt.savefig('/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/similarity_to_irm.pdf')

sparse_to_corr, sparse_to_corr_ci = au.nan_corr(model_corr_sparse, data_corr)
#TODO this should be sparse_randID when it comes up
dense_randID_to_corr, dense_randID_to_corr_ci = au.nan_corr(model_corr_dense_randID, data_corr)
sparse_randA_to_corr, sparse_randA_to_corr_ci = au.nan_corr(model_corr_sparse_randA, data_corr)

plt.figure()
plot_x = np.arange(3)
y_val = np.array([sparse_to_corr, dense_randID_to_corr, sparse_randA_to_corr])
y_val_ci = np.stack([sparse_to_corr_ci, dense_randID_to_corr_ci, sparse_randA_to_corr_ci]).T
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'scrambled anatomy\nconstrained'])
plt.ylabel('similarity to measured correlations')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

plt.savefig('/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/similarity_to_corr.pdf')

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

plt.show()

a=1









