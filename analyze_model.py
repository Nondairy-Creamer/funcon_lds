import numpy as np
import pickle
import analysis_utilities as au
from pathlib import Path

use_synth = False
use_test_data = True
auto_select_ids = True

if use_synth:
    model_folder = Path('C:/Users/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL1_IL45_N80_R0/20230819_092054')
    cell_ids_chosen = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    neuron_to_remove = '4'
    neuron_to_stim = '4'
else:
    # load in the model and training data
    model_folder = Path('C:/Users/mcreamer/Documents/python/funcon_lds/trained_models\exp_DL1_IL45_N80_R0/20230819_201050')
    cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
    # cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
    # cell_ids_chosen = None
    neuron_to_remove = 'AFDR'
    neuron_to_stim = 'AFDR'

# load in the model and the data
model_path = model_folder / 'models' / 'model_trained.pkl'

if use_test_data:
    data_path = model_folder / 'data_test.pkl'
    posterior_path = model_folder / 'posterior_test.pkl'
else:
    data_path = model_folder / 'data_train.pkl'
    posterior_path = model_folder / 'posterior_train.pkl'

model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()
cell_ids = model.cell_ids

if data_path.exists():
    data_file = open(data_path, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    emissions = data['emissions']
    inputs = data['inputs']

    has_data = True
else:
    data = None

if posterior_path.exists():
    posterior_file = open(posterior_path, 'rb')
    posterior_dict = pickle.load(posterior_file)
    posterior_file.close()

    has_post = True
else:
    posterior_dict = None

if cell_ids_chosen is None:
    cell_ids_chosen = cell_ids.copy()

if auto_select_ids and has_data and has_post:
    num_neurons = 10
    measured_stim_responses_ave, measured_stim_responses = au.get_stim_response(data['emissions'], data['inputs'], window=(-60, 120))
    measured_stim_responses_ave_l2 = au.rms(measured_stim_responses_ave, axis=0)
    num_stim = [i.shape[0] for i in measured_stim_responses]

    most_measured_neurons = np.argsort(num_stim, axis=0)[-num_neurons:]
    cell_ids_chosen = [cell_ids[i] for i in most_measured_neurons]

    measured_stim_responses_ave_l2_chosen = measured_stim_responses_ave_l2[np.ix_(most_measured_neurons, most_measured_neurons)]

    measured_stim_responses_ave_l2_chosen[np.eye(len(most_measured_neurons), dtype=bool)] = 0
    ave_response_to_stim = np.nanmean(np.exp(measured_stim_responses_ave_l2_chosen), axis=0)
    neuron_to_remove = cell_ids_chosen[np.nanargmax(ave_response_to_stim)]
    neuron_to_stim = neuron_to_remove

# au.plot_log_likelihood(model)
au.plot_model_params(model, cell_ids_chosen=cell_ids_chosen)
# au.plot_dynamics_eigs(model)

if has_data and has_post:
    a=0
    # au.plot_posterior(data, posterior_dict, cell_ids_chosen, sample_rate=model.sample_rate)
    # responses = au.plot_stim_l2_norm(model, data, posterior_dict, cell_ids_chosen, window=(0, 120))
    # au.plot_stim_response(data, posterior_dict, cell_ids_chosen, neuron_to_stim, window=(-60, 120), sample_rate=model.sample_rate)

    # au.plot_missing_neuron(model, emissions[data_ind_chosen], inputs[data_ind_chosen], posterior[data_ind_chosen], cell_ids, neuron_to_remove, time_window, sample_rate=model.sample_rate)

