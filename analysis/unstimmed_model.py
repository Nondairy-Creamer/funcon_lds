import pickle
import numpy as np
import analysis_methods as am
import analysis_utilities as au
import loading_utilities as lu
from pathlib import Path

cell_ids_chosen = ['AVER', 'AVJL', 'AVJR', 'M3R', 'RMDVL', 'RMDVR', 'RMEL', 'RMER', 'URXL', 'AVDR']
window = (-60, 120)
sub_pre_stim = True
num_stim_cutoff = 5

full_model_path = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0/20231012_134557')
full_model_file = open(full_model_path / 'models' / 'model_trained.pkl', 'rb')
full_model = pickle.load(full_model_file)
full_model_file.close()

full_data_file = open(full_model_path / 'data_train.pkl', 'rb')
full_data = pickle.load(full_data_file)
full_data_file.close()

unstimulated_model_path = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_nostim/20231026_190640/')
unstim_model_file = open(unstimulated_model_path / 'models' / 'model_trained.pkl', 'rb')
unstim_model = pickle.load(unstim_model_file)
unstim_model_file.close()

unstim_data_file = open(unstimulated_model_path / 'data_train.pkl', 'rb')
unstim_data = pickle.load(unstim_data_file)
unstim_data_file.close()

# choose which cells to focus on
if cell_ids_chosen is None:
    cell_ids_chosen = au.auto_select_ids(full_data['inputs'], full_data['cell_ids'], num_neurons=10)

cell_ids_chosen = list(np.sort(cell_ids_chosen))
emissions = full_data['emissions']
inputs = full_data['inputs']
cell_ids = full_data['cell_ids']

unstim_model.cell_ids[unstim_model.cell_ids.index('AMSOL')] = 'VB1'
unstim_model.cell_ids[unstim_model.cell_ids.index('AMSOR')] = 'VD1'

# update the unstimulated model to use the input weights from the full model
unstim_model_inds = np.array([cell_ids.index(i) for i in unstim_model.cell_ids])
data_corr = np.zeros((len(cell_ids), len(cell_ids)))
data_corr[:] = np.nan
data_corr[np.ix_(unstim_model_inds, unstim_model_inds)] = au.nan_corr_data(unstim_data['emissions'])

stacked_input_weights = np.split(full_model.dynamics_input_weights[:full_model.dynamics_dim, :], full_model.dynamics_input_lags, axis=1)
stacked_input_weights_trimmed = np.stack([i[np.ix_(unstim_model_inds, unstim_model_inds)] for i in stacked_input_weights])
new_input_weights = unstim_model._get_lagged_weights(stacked_input_weights_trimmed, unstim_model.dynamics_lags, fill='zeros')
unstim_model.dynamics_input_weights = new_input_weights
unstim_model.dynamics_input_lags = full_model.dynamics_input_lags
new_inputs = [i[:, unstim_model_inds] for i in inputs]

model_sampled = []
for i in new_inputs:
    model_sampled.append(unstim_model.sample(num_time=i.shape[0], inputs_list=[i], add_noise=False)['emissions'][0])

model_sampled, new_inputs, cell_ids = lu.align_data_cell_ids(model_sampled, new_inputs, [unstim_model.cell_ids]*len(model_sampled), cell_ids_unique=cell_ids)

# get the impulse response functions (IRF)
measured_irf, measured_irf_sem, measured_irf_all = au.get_impulse_response_function(emissions, inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)
model_irf, model_irf_sem, model_irf_all = au.get_impulse_response_function(model_sampled, new_inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

model_weights = np.zeros((unstim_model.dynamics_lags, len(cell_ids), len(cell_ids)))
model_weights[:] = np.nan
stacked_weights = au.stack_weights(unstim_model.dynamics_weights[:unstim_model.dynamics_dim, :], unstim_model.dynamics_lags, axis=1)

for i in range(model_weights.shape[0]):
    model_weights[i][np.ix_(unstim_model_inds, unstim_model_inds)] = stacked_weights[i]

model_weights = np.split(model_weights, model_weights.shape[0], axis=0)
model_weights = [i[0, :, :] for i in model_weights]

measured_irf_ave = au.ave_fun(measured_irf[-window[0]:], axis=0)
model_irf_ave = au.ave_fun(model_irf[-window[0]:], axis=0)

# remove IRFs that were measured fewer than run_params['num_stim_cutoff'] times
num_neurons = len(cell_ids)
num_stim = np.zeros((num_neurons, num_neurons))
for ni in range(num_neurons):
    for nj in range(num_neurons):
        resp_to_stim = measured_irf_all[ni][:, -window[0]:, nj]
        num_obs_when_stim = np.sum(np.mean(~np.isnan(resp_to_stim), axis=1) > 0.5)
        num_stim[nj, ni] += num_obs_when_stim

measured_irf_ave[num_stim < num_stim_cutoff] = np.nan
model_irf_ave[num_stim < num_stim_cutoff] = np.nan
data_corr[num_stim < num_stim_cutoff] = np.nan

# set diagonals to nan because we won't be analyzing the diagonals
data_corr[np.eye(data_corr.shape[0], dtype=bool)] = np.nan
measured_irf_ave[np.eye(measured_irf_ave.shape[0], dtype=bool)] = np.nan
model_irf_ave[np.eye(model_irf_ave.shape[0], dtype=bool)] = np.nan
for i in range(len(model_weights)):
    model_weights[i][np.eye(model_weights[i].shape[0], dtype=bool)] = np.nan

# make sure that all the matricies are nan in the same place so its an apples to apples comparison
nan_mask = np.isnan(measured_irf_ave) | np.isnan(model_irf_ave) | np.isnan(data_corr)
measured_irf_ave[nan_mask] = np.nan
model_irf_ave[nan_mask] = np.nan
data_corr[nan_mask] = np.nan
for i in range(len(model_weights)):
    model_weights[i][nan_mask] = np.nan

# run analysis methods on the data
am.plot_model_params(model=unstim_model, model_true=None, cell_ids_chosen=cell_ids_chosen)
am.plot_dynamics_eigs(model=unstim_model)
am.plot_irf_norm(model_weights=model_weights, measured_irf=measured_irf_ave, model_irf=model_irf_ave,
                 data_corr=data_corr, cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

am.plot_irf_traces(measured_irf=measured_irf, measured_irf_sem=measured_irf_sem,
                   model_irf=model_irf, cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen,
                   window=window, sample_rate=unstim_model.sample_rate, num_plot=10)
am.compare_irf_w_prediction(model_weights=model_weights, measured_irf=measured_irf_ave,
                            model_irf=model_irf_ave, data_corr=data_corr,
                            cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)
# if the data is not synthetic compare with the anatomy
am.compare_irf_w_anatomy(model_weights=model_weights, measured_irf=measured_irf_ave,
                         model_irf=model_irf_ave, data_corr=data_corr,
                         cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

