import pickle
import numpy as np
import analysis_methods as am
import analysis_utilities as au
import loading_utilities as lu
from pathlib import Path
from matplotlib import pyplot as plt

# in this analysis we are going to load two models: one trained on data with stimulations and one trained on
# data without stimulations. We're then going to take the dynamics input weights from the stimulated model and put
# them into the model that was trained without stimulations. Then we're going to stimulate the model that was trained
# without stimulations and see if it can predict the observed IRMs

cell_ids_chosen = ['AVER', 'AVJL', 'AVJR', 'M3R', 'RMDVL', 'RMDVR', 'RMEL', 'RMER', 'URXL', 'AVDR']
window = (-60, 120)
sub_pre_stim = True
num_stim_cutoff = 1

path_stim = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0/20231012_134557')
path_unstim = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_nostim/20231107_194042/')

# load in the models
model_stim_file = open(path_stim / 'models' / 'model_trained.pkl', 'rb')
model_stim = pickle.load(model_stim_file)
model_stim_file.close()

model_unstim_file = open(path_unstim / 'models' / 'model_trained.pkl', 'rb')
model_unstim = pickle.load(model_unstim_file)
model_unstim_file.close()

# load in the data to compare the models on
data_stim_test_file = open(path_stim / 'data_test.pkl', 'rb')
data_stim_test = pickle.load(data_stim_test_file)
data_stim_test_file.close()

posterior_stim_test_file = open(path_stim / 'posterior_test.pkl', 'rb')
posterior_stim_test = pickle.load(posterior_stim_test_file)
posterior_stim_test_file.close()

# choose which cells to focus on
if cell_ids_chosen is None:
    cell_ids_chosen = au.auto_select_ids(data_stim_test['inputs'], data_stim_test['cell_ids'], num_neurons=10)

cell_ids_chosen = list(np.sort(cell_ids_chosen))
emissions_stim_test = data_stim_test['emissions']
inputs_stim_test = data_stim_test['inputs']
cell_ids = data_stim_test['cell_ids']
model_stim_sampled = posterior_stim_test['model_sampled']

# get the data correlation for stimulated data
data_stim_train_file = open(path_stim / 'data_train.pkl', 'rb')
data_stim_train_full = pickle.load(data_stim_train_file)
data_stim_train_file.close()

if 'data_corr' in data_stim_train_full.keys():
    data_stim_corr = data_stim_train_full['data_corr']
else:
    data_stim_corr = au.nan_corr_data(data_stim_train_full['emissions'])

    data_stim_train_full['data_corr'] = data_stim_corr
    data_stim_train_file = open(path_stim / 'data_train.pkl', 'wb')
    pickle.dump(data_stim_train_full, data_stim_train_file)
    data_stim_train_file.close()

# get the data correlation for unstimulated data
data_unstim_file = open(path_unstim / 'data_train.pkl', 'rb')
data_unstim = pickle.load(data_unstim_file)
data_unstim_file.close()

unstim_model_inds = np.array([cell_ids.index(i) for i in model_unstim.cell_ids])

if 'data_corr' in data_unstim.keys():
    data_unstim_corr_in = data_unstim['data_corr']
else:
    data_unstim_corr_in = au.nan_corr_data(data_unstim['emissions'])

    data_unstim['data_corr'] = data_unstim_corr_in
    data_unstim_file = open(path_unstim / 'data_train.pkl', 'wb')
    pickle.dump(data_unstim, data_unstim_file)
    data_unstim_file.close()

# get the correlations in the unstimulated data. Expand them to all the cell IDs in the original data
data_unstim_corr = np.zeros((len(cell_ids), len(cell_ids)))
data_unstim_corr[:] = np.nan
data_unstim_corr[np.ix_(unstim_model_inds, unstim_model_inds)] = data_unstim_corr_in

# if we have saved the unstimmed model's sampling grab it
new_inputs = [i[:, unstim_model_inds] for i in inputs_stim_test]

if 'model_sampled_stimmed' in data_unstim.keys():
    model_unstim_sampled = data_unstim['model_sampled_stimmed']
else:
    # now that we have the models, put the input weights from the stimulated model into the unstimulated model
    # split the input weights into their lags
    stacked_input_weights = np.split(model_stim.dynamics_input_weights[:model_stim.dynamics_dim, :], model_stim.dynamics_input_lags, axis=1)
    # for each lag, take only the cells that the unstimulated model observed
    stacked_input_weights_trimmed = np.stack([i[np.ix_(unstim_model_inds, unstim_model_inds)] for i in stacked_input_weights])
    # put the weights back into lagged form
    new_input_weights = model_unstim._get_lagged_weights(stacked_input_weights_trimmed, model_unstim.dynamics_lags, fill='zeros')
    model_unstim.dynamics_input_weights = new_input_weights
    model_unstim.dynamics_input_lags = model_stim.dynamics_input_lags
    # get the inputs for the model subsampled for the observed cells

    # sample the unstimulated model using these inputs
    model_unstim_sampled = []
    for i in new_inputs:
        model_unstim_sampled.append(model_unstim.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])

    data_unstim['model_sampled_stimmed'] = model_unstim_sampled
    data_unstim_file = open(path_unstim / 'data_train.pkl', 'wb')
    pickle.dump(data_unstim, data_unstim_file)
    data_unstim_file.close()

# put this sampled data in the same format at the original stimulated data
model_unstim_sampled, _, cell_ids = lu.align_data_cell_ids(model_unstim_sampled, new_inputs, [model_unstim.cell_ids] * len(model_unstim_sampled), cell_ids_unique=cell_ids)

# get the impulse response functions (IRF)
measured_irf, measured_irf_sem, measured_irf_all = au.get_impulse_response_function(emissions_stim_test, inputs_stim_test, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)
model_stim_irf, model_stim_irf_sem, model_stim_irf_all = au.get_impulse_response_function(model_stim_sampled, inputs_stim_test, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)
model_unstim_irf, model_unstim_irf_sem, model_unstim_irf_all = au.get_impulse_response_function(model_unstim_sampled, inputs_stim_test, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

# get the impulse response magnitudes (IRM)
measured_irm = au.ave_fun(measured_irf[-window[0]:], axis=0)
model_stim_irm = au.ave_fun(model_stim_irf[-window[0]:], axis=0)
model_unstim_irm = au.ave_fun(model_unstim_irf[-window[0]:], axis=0)

# pull out the model weights
# for the unstimulated weights fill in what was measured and nan neurons that weren't observed
model_unstim_weights = np.zeros((model_unstim.dynamics_lags, len(cell_ids), len(cell_ids)))
model_unstim_weights[:] = np.nan
model_unstim_weights_stacked = au.stack_weights(model_unstim.dynamics_weights[:model_unstim.dynamics_dim, :], model_unstim.dynamics_lags, axis=1)

for i in range(model_unstim_weights.shape[0]):
    model_unstim_weights[i][np.ix_(unstim_model_inds, unstim_model_inds)] = model_unstim_weights_stacked[i]

model_unstim_weights = np.split(model_unstim_weights, model_unstim_weights.shape[0], axis=0)
model_unstim_weights = [i[0, :, :] for i in model_unstim_weights]

# pull out the model weights for the stimulated model
model_stim_weights = model_stim.dynamics_weights
model_stim_weights = au.stack_weights(model_stim_weights[:model_stim.dynamics_dim, :], model_stim.dynamics_lags, axis=1)
model_stim_weights = np.split(model_stim_weights, model_stim_weights.shape[0], axis=0)
model_stim_weights = [i[0, :, :] for i in model_stim_weights]

# remove IRMs that were measured fewer than run_params['num_stim_cutoff'] times
# calculate the number of observed stimulations
num_neurons = len(cell_ids)
num_stim = np.zeros((num_neurons, num_neurons))
for ni in range(num_neurons):
    for nj in range(num_neurons):
        resp_to_stim = measured_irf_all[ni][:, -window[0]:, nj]
        num_obs_when_stim = np.sum(np.mean(~np.isnan(resp_to_stim), axis=1) > 0.5)
        num_stim[nj, ni] += num_obs_when_stim

# set measurements below threshold and diagonals to nan
measured_irm[num_stim < num_stim_cutoff] = np.nan
measured_irm[np.eye(measured_irm.shape[0], dtype=bool)] = np.nan

model_stim_irm[num_stim < num_stim_cutoff] = np.nan
for i in range(len(model_stim_weights)):
    model_stim_weights[i][np.eye(model_stim_weights[i].shape[0], dtype=bool)] = np.nan

model_unstim_irm[num_stim < num_stim_cutoff] = np.nan
for i in range(len(model_unstim_weights)):
    model_unstim_weights[i][np.eye(model_unstim_weights[i].shape[0], dtype=bool)] = np.nan

data_stim_corr[num_stim < num_stim_cutoff] = np.nan
data_stim_corr[np.eye(data_stim_corr.shape[0], dtype=bool)] = np.nan

data_unstim_corr[num_stim < num_stim_cutoff] = np.nan
data_unstim_corr[np.eye(data_unstim_corr.shape[0], dtype=bool)] = np.nan

# make sure that all the matricies are nan in the same place so it is an apples to apples comparison
nan_mask = np.isnan(measured_irm) | np.isnan(model_stim_irm) | np.isnan(model_unstim_irm) \
           | np.isnan(data_stim_corr) | np.isnan(data_unstim_corr)
measured_irm[nan_mask] = np.nan

model_stim_irm[nan_mask] = np.nan
model_unstim_irm[nan_mask] = np.nan

data_stim_corr[nan_mask] = np.nan
data_unstim_corr[nan_mask] = np.nan

for i in range(len(model_unstim_weights)):
    model_unstim_weights[i][nan_mask] = np.nan
    model_stim_weights[i][nan_mask] = np.nan

# run analysis methods on the data
# am.plot_model_params(model=model_unstim, model_true=None, cell_ids_chosen=cell_ids_chosen)
# am.plot_dynamics_eigs(model=model_unstim)
# am.plot_irf_norm(model_weights=model_weights, measured_irf=measured_irf_ave, model_irf=model_irf_ave,
#                  data_corr=data_corr, cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

# am.plot_irf_traces(measured_irf=measured_irf, measured_irf_sem=measured_irf_sem,
#                    model_irf=model_irf, cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen,
#                    window=window, sample_rate=model_unstim.sample_rate, num_plot=10)
unstim_scores = am.compare_measured_and_model_irm(model_weights=model_unstim_weights, measured_irm=measured_irm,
                                                  model_irm=model_unstim_irm, data_corr=data_unstim_corr,
                                                  cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)
stim_scores = am.compare_measured_and_model_irm(model_weights=model_stim_weights, measured_irm=measured_irm,
                                                model_irm=model_stim_irm, data_corr=data_stim_corr,
                                                cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

plt.figure()
plot_x = np.arange(5)
y_val = np.array([stim_scores['data_corr'][0], stim_scores['model_sampled'][0], stim_scores['model_weights'][0], unstim_scores['data_corr'][0], unstim_scores['model_sampled'][0]])
y_val_ci = np.abs(np.stack([stim_scores['data_corr'][1], stim_scores['model_sampled'][1], stim_scores['model_weights'][1], unstim_scores['data_corr'][1], unstim_scores['model_sampled'][1]]).T - y_val)
plt.bar(plot_x, y_val)
plt.errorbar(plot_x, y_val, yerr=y_val_ci, fmt='.', color='k')
plt.xticks(plot_x, ['stim data corr', 'stim model IRMs', 'stim model weights', 'unstim data corr', 'unstim model IRMs'])
plt.ylabel('correlation to measured IRMs')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_rotation(45)
plt.tight_layout()
plt.show()

# if the data is not synthetic compare with the anatomy
# am.compare_irf_w_anatomy(model_weights=model_weights, measured_irf=measured_irf_ave,
#                          model_irf=model_irf_ave, data_corr=data_corr,
#                          cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)
