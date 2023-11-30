from pathlib import Path
from matplotlib import pyplot as plt
import analysis_utilities as au
import numpy as np
import pickle
import analysis_methods as am


required_num_stim = 5
force_calc = False
sub_pre_stim = True
window = [-60, 120]
cell_ids_chosen = ['AVER', 'AVJL', 'AVJR', 'M3R', 'RMDVL', 'RMDVR', 'RMEL', 'RMER', 'URXL', 'AVDR']

# paths for the full fit, with randomized IDs, and with randomized anatomy
model_path = [Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap/20231030_200902'),
              ]

# get the models
models = []
posterior_dicts = []
for mp in model_path:
    model_file = open(mp / 'models' / 'model_trained.pkl', 'rb')
    models.append(pickle.load(model_file))
    model_file.close()

    post_file = open(mp / 'posterior_test.pkl', 'rb')
    posterior_dicts.append(pickle.load(post_file))
    post_file.close()

# get the data (the same for all runs)
data_train_file = open(model_path[0] / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

if 'data_corr' in data_train.keys():
    data_corr = data_train['data_corr']
else:
    data_corr = au.nan_corr_data(data_train['emissions'])

    data_train['data_corr'] = data_corr

    data_train_file = open(model_path[0] / 'data_train.pkl', 'wb')
    pickle.dump(data_train, data_train_file)
    data_train_file.close()

data_test_file = open(model_path[0] / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

# test data
emissions_test = data_test['emissions']
inputs_test = data_test['inputs']
cell_ids = data_test['cell_ids']
data_irms, data_irfs, data_irfs_sem = \
    au.simple_get_irms(emissions_test, inputs_test, required_num_stim=required_num_stim,
                       window=window, sub_pre_stim=sub_pre_stim)

# get anatomical data
chem_conn, gap_conn, pep_conn = au.load_anatomical_data(cell_ids=cell_ids)

# get the model weights
model_weights = []
for m in models:
    this_model_weights = au.stack_weights(m.dynamics_weights[:m.dynamics_dim, :], m.dynamics_lags)
    this_model_weights[:, np.eye(m.dynamics_dim, dtype=bool)] = np.nan
    model_weights.append(this_model_weights)

# get the IRMs of the models and data
# dense
model_irms = []
model_irfs = []
model_irfs_sem = []
for pd in posterior_dicts:
    this_model_irms, this_model_irfs, this_model_irfs_sem = \
        au.simple_get_irms(pd['model_sampled'], inputs_test, required_num_stim=required_num_stim,
                           window=window, sub_pre_stim=sub_pre_stim)
    model_irms.append(this_model_irms)
    model_irfs.append(this_model_irfs)
    model_irfs_sem.append(this_model_irfs_sem)

# find nan locations across the irms and correlations and set them all to nan so we compare apples-to-apples
nan_mask = np.isnan(model_irms[0])
for i in model_irms[1:]:
    nan_mask = nan_mask | np.isnan(i)

nan_mask = nan_mask | np.isnan(data_irms)

# set matricies to nan with nan mask
for i in range(len(model_irms)):
    model_irms[i][nan_mask] = np.nan

data_irms[nan_mask] = np.nan
data_corr[nan_mask] = np.nan
for mw in range(len(model_weights)):
    model_weights[mw][:, nan_mask] = np.nan

# compare model IRMs to measured IRMs
model_irms_score = []
model_irms_score_ci = []
for mi in model_irms:
    model_irms_to_measured_irms, model_irms_to_measured_irms_ci = au.nan_corr(mi, data_irms)
    model_irms_score.append(model_irms_to_measured_irms)
    model_irms_score_ci.append(model_irms_to_measured_irms_ci)

# compare model weights to synpase count
anat_mask = (chem_conn + gap_conn) > 0
anatomy_list = [chem_conn[anat_mask], gap_conn[anat_mask]]
# sub sample the weights down to where there are known connections
model_weights_anat = []
for mw in model_weights:
    split_weights = np.split(mw, mw.shape[0])
    model_weights_anat.append([np.abs(i[0, anat_mask]) for i in split_weights])

num_models = len(model_weights_anat)
weights_to_sc = []
weights_to_sc_ci = []
anatomy_comb = []
weights_comb = []
for i in range(num_models):
    this_weights_to_sc, this_weights_to_sc_ci, this_anatomy_comb, this_weights_comb = \
        au.compare_matrix_sets(anatomy_list, model_weights_anat[i], positive_weights=True)
    weights_to_sc.append(this_weights_to_sc)
    weights_to_sc_ci.append(this_weights_to_sc_ci)
    anatomy_comb.append(this_anatomy_comb)
    weights_comb.append(this_weights_comb)

# compare model weights with binary connections
# binarize model weights
std_factor = 3
weights_bin = [np.sum(np.abs(i), axis=0) for i in model_weights]
weights_bin = [(i > (np.nanstd(i) / std_factor)).astype(float) for i in weights_bin]
for i in range(len(weights_bin)):
    weights_bin[i][nan_mask] = np.nan

weights_to_anat = []
weights_to_anat_ci = []

for wb in weights_bin:
    this_weights_to_anat, this_weights_to_anat_ci = au.nan_corr(anat_mask, wb)
    weights_to_anat.append(this_weights_to_anat)
    weights_to_anat_ci.append(this_weights_to_anat_ci)

# plotting
am.plot_missing_neuron(data=data_test, posterior_dict=posterior_dicts[0], sample_rate=models[0].sample_rate)

