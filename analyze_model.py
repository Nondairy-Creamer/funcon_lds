import numpy as np
import pickle
import analysis_utilities as au
import matplotlib as mpl
from pathlib import Path

use_synth = False
run_test = True

if not use_synth:
    # load in the model and training data
    model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_test/20230809_141820/')
    cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
    cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
    neuron_to_remove = 'AVJR'
    neuron_to_stim = 'AVJR'
else:
    model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_test/20230808_185310/')
    # model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/syn_test/20230808_163745/')
    cell_ids_chosen = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    neuron_to_remove = '4'
    neuron_to_stim = '4'

window_size = 1000
colormap = mpl.colormaps['coolwarm']

# load in the model and the data
model_path = model_folder / 'model_trained.pkl'

if run_test:
    data_test_path = model_folder / 'data_test.pkl'
    posterior_test_path = model_folder / 'posterior_test.pkl'
else:
    data_test_path = model_folder / 'data_train.pkl'
    posterior_test_path = model_folder / 'posterior_train.pkl'

model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()

data_test_file = open(data_test_path, 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

posterior_test_file = open(posterior_test_path, 'rb')
posterior_test = pickle.load(posterior_test_file)
posterior_test_file.close()

emissions = data_test['emissions']
inputs = data_test['inputs']
cell_ids = data_test['cell_ids']

post_pred = posterior_test['post_pred']
post_pred_noise = posterior_test['post_pred_noise']
posterior = posterior_test['posterior']

neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])

# the inputs matrix doesn't match the emissions matrix size because some neurons were not stimulated
# upscale the inputs here so that the location of the stimulation in the input matrix
# matches the neuron that was stimulated
dynamics_input_weights_mask = model.param_props['mask']['dynamics_input_weights']
weights_loc = ~np.all(dynamics_input_weights_mask == 0, axis=1)
inputs_full = []

for i in inputs:
    inputs_full.append(np.zeros((i.shape[0], model.dynamics_dim)))
    inputs_full[-1][:, weights_loc] = i

# get all the inputs but with only the chosen neurons
inputs_truncated = [i[:, neuron_inds_chosen] for i in inputs_full]
data_ind_chosen, time_window = au.find_stim_events(inputs_truncated, window_size=window_size)

emissions_chosen = emissions[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
inputs_chosen = inputs_full[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
posterior_chosen = posterior[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
post_pred_chosen = post_pred_noise[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]

au.plot_log_likelihood(model)
au.plot_model_params(model, cell_ids, cell_ids_chosen=cell_ids_chosen)
au.plot_dynamics_eigs(model)
au.plot_posterior(emissions_chosen, inputs_chosen, posterior_chosen, post_pred_chosen, cell_ids_chosen, sample_rate=model.sample_rate)
au.plot_stim_l2_norm(model, emissions, inputs_full, posterior, post_pred, cell_ids, cell_ids_chosen, window=(0, 120))
au.plot_stim_response(emissions, inputs_full, posterior, post_pred, cell_ids, cell_ids_chosen, neuron_to_stim, window=(-60, 120), sample_rate=model.sample_rate)

# au.plot_missing_neuron(model, emissions[data_ind_chosen], inputs[data_ind_chosen], posterior[data_ind_chosen], cell_ids, neuron_to_remove, time_window, sample_rate=model.sample_rate)

