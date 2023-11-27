from pathlib import Path
from matplotlib import pyplot as plt
import analysis_utilities as au
import analysis_methods as am
import numpy as np
import pickle


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

    return irms, irfs, irfs_sem


required_num_stim = 5
force_calc = False
sub_pre_stim = True
window = [-60, 120]
cell_ids_chosen = ['AVER', 'AVJL', 'AVJR', 'M3R', 'RMDVL', 'RMDVR', 'RMEL', 'RMER', 'URXL', 'AVDR']

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

# test data
emissions_test = data_test['emissions']
inputs_test = data_test['inputs']
cell_ids = data_test['cell_ids']
data_irms, data_irfs, data_irfs_sem = get_irms(emissions_test)

# get anatomical data
chem_conn, gap_conn, pep_conn = au.load_anatomical_data(cell_ids=cell_ids)

# get the IRMs of the models and data
# dense
model_irms = []
model_irfs = []
model_irfs_sem = []
for pd in posterior_dicts:
    this_model_irms, this_model_irfs, this_model_irfs_sem = get_irms(pd['model_sampled'])
    model_irms.append(this_model_irms)
    model_irfs.append(this_model_irfs)
    model_irfs_sem.append(this_model_irfs_sem)

# predict data correlation
model_corrs = []

for m in models:
    model_corrs.append(m.dynamics_weights @ m.dynamics_weights.T + m.dynamics_cov)
    for i in range(1000):
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

# compare data corr and data IRM to connectome
anatomy_list = [chem_conn, gap_conn]
data_corr_conn, data_corr_conn_ci = au.compare_matrix_sets(anatomy_list, data_corr)[:2]
data_irm_conn, data_irm_conn_ci = au.compare_matrix_sets(anatomy_list, data_irms)[:2]

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
            cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen, window=window, num_plot=10)
example_weights = models[0].dynamics_weights[:models[0].dynamics_dim, :models[0].dynamics_dim]
example_weights[np.eye(example_weights.shape[0], dtype=bool)] = np.nan
am.plot_irm(model_weights=example_weights, measured_irm=data_irms, model_irm=model_irms[0], data_corr=data_corr,
            cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

plt.figure()
plot_x = np.arange(2)
y_val = np.array([data_corr_conn, data_irm_conn])
y_val_ci = np.stack([data_corr_conn_ci, data_irm_conn_ci]).T
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['data correlation', 'data IRMs'])
plt.ylabel('similarity to connectome')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

plt.figure()
plot_x = np.arange(3)
y_val = np.array(model_corr_score)
y_val_ci = np.stack(model_corr_score_ci).T
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
plt.ylabel('similarity to measured correlation')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

plt.figure()
plot_x = np.arange(3)
y_val = np.array(model_irms_score)
y_val_ci = np.stack(model_irms_score_ci).T
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
plt.ylabel('similarity to measured IRMs')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()

plt.show()




