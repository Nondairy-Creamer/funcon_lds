from pathlib import Path
from matplotlib import pyplot as plt
import analysis_utilities as au
import numpy as np
import pickle
import wormneuroatlas as wa


required_num_stim = 5
force_calc = False
sub_pre_stim = True
window = [-60, 120]
cell_ids_chosen = ['AVER', 'AVJL', 'AVJR', 'M3R', 'RMDVL', 'RMDVR', 'RMEL', 'RMER', 'URXL', 'AVDR']
fig_save_path = Path('/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/fig2')

# paths for the full fit, with randomized IDs, and with randomized anatomy
model_path = [Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap/20231030_200902'),
              Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0/20231031_011900'),
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
# TODO change back to chem_conn
chem_mask = chem_conn > 0
anat_mask = (chem_conn + gap_conn) > 0
anatomy_list = [chem_conn[anat_mask], gap_conn[anat_mask]]
# sub sample the weights down to where there are known connections
model_weights_anat = []
model_weights_anat_abs = []
model_weights_chem = []
for mw in model_weights:
    split_weights = np.split(mw, mw.shape[0])
    model_weights_anat_abs.append([np.abs(i[0, anat_mask]) for i in split_weights])
    model_weights_anat.append([i[0, anat_mask] for i in split_weights])
    model_weights_chem.append([i[0, chem_mask] for i in split_weights])

num_models = len(model_weights_anat_abs)
weights_to_sc = []
weights_to_sc_ci = []
anatomy_comb = []
weights_comb = []
model_to_model = []
for i in range(num_models):
    this_weights_to_sc, this_weights_to_sc_ci, this_anatomy_comb, this_weights_comb = \
        au.compare_matrix_sets(anatomy_list, model_weights_anat_abs[i], positive_weights=True)
    weights_to_sc.append(this_weights_to_sc)
    weights_to_sc_ci.append(this_weights_to_sc_ci)
    anatomy_comb.append(this_anatomy_comb)
    weights_comb.append(this_weights_comb)

model_to_model_sim, model_to_model_sim_ci, constrained_weights, unconstrained_weights = \
    au.compare_matrix_sets(model_weights_anat[0], model_weights_anat[1], positive_weights=False)

# compare to predicted synapse sign
watlas = wa.NeuroAtlas()
signed_connections = watlas.get_anatomical_connectome(signed=True)
chem_sign = watlas.get_chemical_synapse_sign()

cmplx = np.logical_and(np.any(chem_sign == -1, axis=0),
                       np.any(chem_sign == 1, axis=0))
chem_sign = np.nansum(chem_sign, axis=0)
chem_sign[cmplx] = 0
chem_sign[chem_sign == 0] = np.nan
chem_sign[chem_sign > 1] = 1
chem_sign[chem_sign < -1] = -1

atlas_ids = list(watlas.neuron_ids)
atlas_ids[atlas_ids.index('AWCON')] = 'AWCL'
atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCR'
cell_inds = np.array([atlas_ids.index(i) for i in cell_ids])
chem_sign = chem_sign[np.ix_(cell_inds, cell_inds)]
chem_sign[nan_mask] = np.nan
#TODO: decide if there should be a negative sign here
chem_sign = -chem_sign[chem_mask]

mss = []
mss_ci = []
for mwc in range(len(model_weights_chem)):
    for m in range(len(model_weights_chem[mwc])):
        model_weights_chem[mwc][m] = (model_weights_chem[mwc][m] > 0).astype(float) - (model_weights_chem[mwc][m] < 0).astype(float)

    model_sign_sim, model_sign_sim_ci = \
        au.compare_matrix_sets(chem_sign, model_weights_chem[mwc], positive_weights=False)[:2]

    mss.append(model_sign_sim)
    mss_ci.append(model_sign_sim_ci)

irm_sign, irm_sign_ci = au.nan_corr(chem_sign, data_irms[chem_mask])
mss.append(irm_sign)
mss_ci.append(irm_sign_ci)

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
plt.figure()
y_val = np.array(model_irms_score)
y_val_ci = np.stack(model_irms_score_ci).T
plot_x = np.arange(y_val.shape[0])
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
plt.ylabel('similarity to measured IRMs')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.savefig(fig_save_path / 'sim_to_irms.pdf')

plt.figure()
y_val = np.array(mss)
y_val_ci = np.stack(mss_ci).T
plot_x = np.arange(y_val.shape[0])
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\nunconstrained', 'data IRMs'])
plt.ylabel('similarity to known synapse sign')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.savefig(fig_save_path / 'sim_to_synapse_sign.pdf')

# plot model vs unconstrained model IRMs
plt.figure()
model_irm_similarity = au.nan_corr(model_irms[0].reshape(-1), model_irms[1].reshape(-1))[0]
plt.scatter(model_irms[0].reshape(-1), model_irms[1].reshape(-1))
plt.xlabel('model IRMs')
plt.ylabel('unconstrained model IRMs')
plt.title('similarity = ' + str(model_irm_similarity))
plt.savefig(fig_save_path / 'model_to_model_irm_scatter.pdf')

# plot model vs unconstrained model weights
plt.figure()
plt.scatter(constrained_weights, unconstrained_weights)
plt.xlabel('model weights')
plt.ylabel('unconstrained model weights')
plt.title('model weights where there are anatomical connections\nsimilarity = ' + str(model_to_model_sim))
plt.savefig(fig_save_path / 'model_to_model_weights_scatter.pdf')

plt.figure()
y_val = np.array(weights_to_anat)
y_val_ci = np.stack(weights_to_anat_ci).T
plot_x = np.arange(y_val.shape[0])
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
plt.ylabel('similarity to anatomical connections')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.savefig(fig_save_path / 'sim_to_conn.pdf')

plt.figure()
y_val = np.array(weights_to_sc)
y_val_ci = np.stack(weights_to_sc_ci).T
plot_x = np.arange(y_val.shape[0])
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
plt.ylabel('similarity to synapse count')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.savefig(fig_save_path / 'sim_to_synap_count.pdf')

for i in range(num_models):
    plt.figure()
    plt.scatter(anatomy_comb[i], weights_comb[i])
    plt.xlabel('synapse count')
    plt.ylabel('|model weights|')

plt.show()




