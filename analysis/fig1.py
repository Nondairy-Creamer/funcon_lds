from pathlib import Path
from matplotlib import pyplot as plt
import analysis_utilities as au
import analysis_methods as am
import numpy as np
import pickle


required_num_stim = 5
force_calc = False
sub_pre_stim = True
window = [-60, 120]
num_traces_plot = 5
cell_ids_chosen = ['AVER', 'AVJL', 'AVJR', 'M3R', 'RMDVL', 'RMDVR', 'RMEL', 'RMER', 'URXL', 'AVDR']
fig_save_path = Path('/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/fig1')

# paths for the full fit, with randomized IDs, and with randomized anatomy
model_path = [Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap/20231030_200902'),
              Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap_randCellID/20231120_122606'),
              Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_synap_rand/20231114_125113')]

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

# pull out the model weights
# get the model weights
model_weights = []
model_weights_list = []
for m in models:
    this_model_weights = au.stack_weights(m.dynamics_weights[:m.dynamics_dim, :], m.dynamics_lags)
    this_model_weights[:, np.eye(m.dynamics_dim, dtype=bool)] = np.nan
    model_weights.append(this_model_weights)

    this_model_weights_list = np.split(this_model_weights, this_model_weights.shape[0], axis=0)
    this_model_weights_list = [i[0, :, :] for i in this_model_weights_list]
    model_weights_list.append(this_model_weights_list)

# test data
emissions_test = data_test['emissions']
inputs_test = data_test['inputs']
cell_ids = data_test['cell_ids']
data_irms, data_irfs, data_irfs_sem = \
    au.simple_get_irms(emissions_test, inputs_test, required_num_stim=required_num_stim,
                       window=window, sub_pre_stim=sub_pre_stim)

# get anatomical data
chem_conn, gap_conn, pep_conn = au.load_anatomical_data(cell_ids=cell_ids)

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

# predict data correlation
model_corrs = []

for m in models:
    model_corrs.append(m.dynamics_weights @ m.dynamics_weights.T + m.dynamics_cov)
    for i in range(10):
        model_corrs[-1] = m.dynamics_weights @ model_corrs[-1] @ m.dynamics_weights.T + m.dynamics_cov
    model_corrs[-1] = model_corrs[-1][:m.dynamics_dim, :m.dynamics_dim]

# find nan locations across the irms and correlations and set them all to nan so we compare apples-to-apples
nan_mask = np.isnan(model_irms[0])
for i in model_irms[1:]:
    nan_mask = nan_mask | np.isnan(i)

nan_mask = nan_mask | np.isnan(data_irms)
nan_mask = nan_mask | np.isnan(data_corr)

for i in range(len(model_irms)):
    model_irms[i][nan_mask] = np.nan
    model_corrs[i][nan_mask] = np.nan

data_irms[nan_mask] = np.nan
data_corr[nan_mask] = np.nan
for i in range(len(model_weights)):
    model_weights[i][:, nan_mask] = np.nan

# compare data corr and data IRM to connectome
anatomy = chem_conn + gap_conn
weights_mag = np.sum(np.abs(model_weights[0]), axis=0)
anatomy_bin = anatomy > 0
std_factor = 3
data_corr_bin = (np.abs(data_corr) > (np.nanstd(data_corr) / std_factor)).astype(float)
data_irms_bin = (np.abs(data_irms) > (np.nanstd(data_irms) / std_factor)).astype(float)
model_weights_bin = (np.abs(weights_mag) > (np.nanstd(weights_mag) / std_factor)).astype(float)

data_corr_bin[nan_mask] = np.nan
data_irms_bin[nan_mask] = np.nan
model_weights_bin[nan_mask] = np.nan

data_corr_conn, data_corr_conn_ci = au.nan_corr(anatomy_bin, data_corr)
data_irm_conn, data_irm_conn_ci = au.nan_corr(anatomy_bin, data_irms)
model_weights_conn, model_weights_conn_ci = au.nan_corr(anatomy_bin, model_weights_bin)

# compare model corr to measured corr
model_corr_score = []
model_corr_score_ci = []
for mc in model_corrs:
    model_corr_to_measured_corr, model_corr_to_measured_corr_ci = au.nan_corr(mc, data_corr)
    model_corr_score.append(model_corr_to_measured_corr)
    model_corr_score_ci.append(model_corr_to_measured_corr_ci)

# compare model IRMs to measured IRMs
model_irms_score = []
model_irms_score_ci = []
for mi in model_irms:
    model_irms_to_measured_irms, model_irms_to_measured_irms_ci = au.nan_corr(mi, data_irms)
    model_irms_score.append(model_irms_to_measured_irms)
    model_irms_score_ci.append(model_irms_to_measured_irms_ci)

# plotting
am.plot_irf(measured_irf=data_irfs, measured_irf_sem=data_irfs_sem, model_irf=model_irfs[0],
            cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen, window=window, num_plot=num_traces_plot,
            fig_save_path=fig_save_path)
am.plot_irm(model_weights=model_weights_list[0], measured_irm=data_irms, model_irm=model_irms[0], data_corr=data_corr,
            cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen, fig_save_path=fig_save_path)

plt.figure()
y_val = np.array([model_weights_conn, data_corr_conn, data_irm_conn, ])
y_val_ci = np.stack([model_weights_conn_ci, data_corr_conn_ci, data_irm_conn_ci]).T
plot_x = np.arange(y_val.shape[0])
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model weights', 'data correlation', 'data IRMs'])
plt.ylabel('similarity to connectome')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.savefig(fig_save_path / 'sim_to_conn.pdf')

plt.figure()
y_val = np.array(model_corr_score)
y_val_ci = np.stack(model_corr_score_ci).T
plot_x = np.arange(y_val.shape[0])
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
plt.ylabel('similarity to measured correlation')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.savefig(fig_save_path / 'sim_to_corr.pdf')

plt.figure()
y_val = np.array(model_irms_score)
y_val_ci = np.stack(model_irms_score_ci).T
plot_x = np.arange(y_val.shape[0])
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
plt.ylabel('similarity to measured IRMs')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.savefig(fig_save_path / 'sim_to_irm.pdf')

plt.show()




