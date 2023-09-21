import pickle
import numpy as np
import analysis_methods as am
import analysis_utilities as au
import loading_utilities as lu
from pathlib import Path

# run_params = lu.get_run_params(param_name='analysis_params/ana_test.yml')
# run_params = lu.get_run_params(param_name='analysis_params/ana_syn_test_analysis.yml')
run_params = lu.get_run_params(param_name='analysis_params/ana_exp_DL.yml')
# run_params = lu.get_run_params(param_name='analysis_params/ana_syn_ridge_sweep.yml')
window = run_params['window']
sub_pre_stim = run_params['sub_pre_stim']
model_folders = [Path(i) for i in run_params['model_folders']]

model_list = []
model_true_list = []
posterior_train_list = []
data_train_list = []
posterior_test_list = []
data_test_list = []

for m in model_folders:
    m = 'trained_models' / m
    # load in the model and the data
    model_file = open(m / 'models' / 'model_trained.pkl', 'rb')
    model_list.append(pickle.load(model_file))
    model_file.close()

    model_true_path = m / 'models' / 'model_true.pkl'
    if model_true_path.exists():
        model_true_file = open(m / 'models' / 'model_true.pkl', 'rb')
        model_true_list.append(pickle.load(model_true_file))
        model_true_file.close()
    else:
        model_true_list.append(None)

    posterior_train_file = open(m / 'posterior_train.pkl', 'rb')
    posterior_train_list.append(pickle.load(posterior_train_file))
    posterior_train_file.close()

    data_train_file = open(m / 'data_train.pkl', 'rb')
    data_train_list.append(pickle.load(data_train_file))
    data_train_file.close()

    posterior_test_file = open(m / 'posterior_test.pkl', 'rb')
    posterior_test_list.append(pickle.load(posterior_test_file))
    posterior_test_file.close()

    data_test_file = open(m / 'data_test.pkl', 'rb')
    data_test_list.append(pickle.load(data_test_file))
    data_test_file.close()

best_model_ind = am.plot_model_comparison(run_params['sorting_param'], model_list, posterior_train_list, data_train_list,
                                          posterior_test_list, data_test_list)

model = model_list[best_model_ind]
model_true = model_true_list[best_model_ind]

if run_params['use_test_data']:
    data = data_test_list[best_model_ind]
    posterior_dict = posterior_test_list[best_model_ind]
    posterior_path = 'trained_models' / model_folders[best_model_ind] / 'posterior_test.yml'
else:
    data = data_train_list[best_model_ind]
    posterior_dict = posterior_train_list[best_model_ind]
    posterior_path = 'trained_models' / model_folders[best_model_ind] / 'posterior_train.yml'

is_synth = '0' in data['cell_ids']

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

emissions = data['emissions']
inputs = data['inputs']
post_pred = posterior_dict['post_pred']
cell_ids = data['cell_ids']
posterior = posterior_dict['posterior']

# get the impulse response functions (IRF)
measured_irf, measured_irf_sem, measured_irf_all = au.get_impulse_response_function(emissions, inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)
measured_irf_rms = au.p_norm(measured_irf[-window[0]:], axis=0)
posterior_irf = au.get_impulse_response_function(posterior, inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)[0]
post_pred_irf = au.get_impulse_response_function(post_pred, inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)[0]
post_pred_irf_rms = au.p_norm(post_pred_irf[-window[0]:], axis=0)
# data_corr = np.abs(au.nan_corr_data(emissions))
data_corr = au.nan_corr_data(emissions)

# TODO: remove testing
# from matplotlib import pyplot as plt
# measured_irf_neg = np.where(post_pred_irf_rms > 0.5)
#
# for i in range(len(measured_irf_neg[0])):
#     if np.any(~np.isnan(measured_irf[:, measured_irf_neg[0][i], measured_irf_neg[1][i]])):
#         plt.figure()
#         plt.plot(measured_irf[:, measured_irf_neg[0][i], measured_irf_neg[1][i]])
#         plt.plot(post_pred_irf[:, measured_irf_neg[0][i], measured_irf_neg[1][i]])
#
# plt.show()

num_neurons = len(cell_ids)
num_stim = np.zeros((num_neurons, num_neurons))
for ni in range(num_neurons):
    for nj in range(num_neurons):
        resp_to_stim = measured_irf_all[ni][:, -window[0]:, nj]
        num_obs_when_stim = np.sum(np.any(~np.isnan(resp_to_stim), axis=1))
        num_stim[nj, ni] += num_obs_when_stim

stim_cutoff = 5
measured_irf_rms[num_stim < stim_cutoff] = np.nan
post_pred_irf_rms[num_stim < stim_cutoff] = np.nan
data_corr[num_stim < stim_cutoff] = np.nan

am.plot_model_params(model, model_true=model_true, cell_ids_chosen=cell_ids_chosen)
am.plot_dynamics_eigs(model)

am.plot_posterior(data, posterior_dict, cell_ids_chosen, sample_rate=model.sample_rate)
am.plot_irf_norm(model, measured_irf_rms, post_pred_irf_rms, data_corr, cell_ids_chosen)
am.compare_irf_w_anatomy(model, measured_irf_rms, post_pred_irf_rms, data_corr)
am.plot_irf(measured_irf, measured_irf_sem, posterior_irf, post_pred_irf, cell_ids, cell_ids_chosen, window,
            sample_rate=model.sample_rate, num_plot=5)
posterior_dict['posterior_missing'] = am.plot_missing_neuron(model, data, posterior_dict, cell_ids_chosen, neuron_to_remove, force_calc=run_params['force_calc_missing_posterior'])

posterior_file = open(posterior_path.with_suffix('.pkl'), 'wb')
pickle.dump(posterior_dict, posterior_file)
posterior_file.close()

a=1
