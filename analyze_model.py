import pickle
import numpy as np
import analysis_methods as am
import analysis_utilities as au
import loading_utilities as lu
from pathlib import Path

# run_params = lu.get_run_params(param_name='analysis_params/ana_test.yml')
# run_params = lu.get_run_params(param_name='analysis_params/ana_syn_test_analysis.yml')
run_params = lu.get_run_params(param_name='analysis_params/ana_exp_DL.yml')
# run_params = lu.get_run_params(param_name='analysis_params/ana_exp_DL_old.yml')
# run_params = lu.get_run_params(param_name='analysis_params/ana_exp_DL_fullq.yml')
# run_params = lu.get_run_params(param_name='analysis_params/ana_syn_ridge_sweep.yml')

window = run_params['window']
sub_pre_stim = run_params['sub_pre_stim']
model_folders = [Path(i) for i in run_params['model_folders']]

# choose the model that has the highest test log likelihood
model, model_true, data, posterior_dict, data_path, posterior_path, data_corr = \
    am.get_best_model(model_folders, run_params['sorting_param'], use_test_data=run_params['use_test_data'],
                      plot_figs=run_params['plot_model_comparison'], best_model_ind=run_params['best_model_ind'])

is_synth = '0' in data['cell_ids']

# choose which cells to focus on
if run_params['auto_select_ids']:
    cell_ids_chosen, neuron_to_remove, neuron_to_stim = au.auto_select_ids(data['inputs'], data['cell_ids'], num_neurons=run_params['num_select_ids'])
else:
    # check if the data is synthetic
    if is_synth:
        cell_ids_chosen = [str(i) for i in np.arange(run_params['num_select_ids'])]
        neuron_to_remove = '0'
        neuron_to_stim = '1'
    else:
        cell_ids_chosen = run_params['cell_ids_chosen']
        neuron_to_remove = run_params['neuron_to_remove']
        neuron_to_stim = run_params['neuron_to_stim']

cell_ids_chosen = list(np.sort(cell_ids_chosen))
emissions = data['emissions']
inputs = data['inputs']
model_sampled = posterior_dict['model_sampled']
cell_ids = data['cell_ids']
posterior = posterior_dict['posterior']

# get the impulse response functions (IRF)
measured_irf, measured_irf_sem, measured_irf_all = au.get_impulse_response_function(emissions, inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)
model_irf, model_irf_sem, model_irf_all = au.get_impulse_response_function(model_sampled, inputs, window=window,
                                                                           sub_pre_stim=sub_pre_stim,
                                                                           return_pre=True)

model_weights = model.dynamics_weights
model_weights = au.stack_weights(model_weights[:model.dynamics_dim, :], model.dynamics_lags, axis=1)
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

measured_irf_ave[num_stim < run_params['num_stim_cutoff']] = np.nan
model_irf_ave[num_stim < run_params['num_stim_cutoff']] = np.nan
data_corr[num_stim < run_params['num_stim_cutoff']] = np.nan

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
# am.plot_model_params(model=model, model_true=model_true, cell_ids_chosen=cell_ids_chosen)
# am.plot_dynamics_eigs(model=model)
# am.plot_posterior(data=data, posterior_dict=posterior_dict, cell_ids_chosen=cell_ids_chosen, sample_rate=model.sample_rate)
# am.plot_irf_norm(model_weights=model_weights, measured_irf=measured_irf_ave, model_irf=model_irf_ave,
#                  data_corr=data_corr, cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

# am.plot_irf_traces(measured_irf=measured_irf, measured_irf_sem=measured_irf_sem,
#                    model_irf=model_irf, cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen,
#                    window=window, sample_rate=model.sample_rate, num_plot=5)
am.compare_irf_w_prediction(model_weights=model_weights, measured_irf=measured_irf_ave,
                            model_irf=model_irf_ave, data_corr=data_corr,
                            cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

# if the data is not synthetic compare with the anatomy
if not is_synth:
    am.compare_irf_w_anatomy(model_weights=model_weights, measured_irf=measured_irf_ave,
                             model_irf=model_irf_ave, data_corr=data_corr,
                             cell_ids=cell_ids, cell_ids_chosen=cell_ids_chosen)

if 'posterior_missing' in posterior_dict.keys():
    am.plot_missing_neuron(data=data, posterior_dict=posterior_dict, sample_rate=model.sample_rate)

a=1
