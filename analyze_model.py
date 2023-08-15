import numpy as np
import pickle
import analysis_utilities as au
import matplotlib as mpl
from pathlib import Path

colormap = mpl.colormaps['coolwarm']
use_synth = False
use_test_data = False

if not use_synth:
    # load in the model and training data
    model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_test/20230815_150126/')
    cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
    # cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
    # cell_ids_chosen = None
    neuron_to_remove = 'AVJR'
    neuron_to_stim = 'AVJR'
else:
    model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/syn_test/20230809_161257/')
    # model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/syn_test/20230808_163745/')
    cell_ids_chosen = ['0', '1', '2', '3', '4']
    neuron_to_remove = '4'
    neuron_to_stim = '4'

# load in the model and the data
model_path = model_folder / 'model_trained.pkl'

if use_test_data:
    data_path = model_folder / 'data_test.pkl'
    posterior_path = model_folder / 'posterior_test.pkl'
else:
    data_path = model_folder / 'data_train.pkl'
    posterior_path = model_folder / 'posterior_train.pkl'

model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()

data_file = open(data_path, 'rb')
data = pickle.load(data_file)
data_file.close()

posterior_file = open(posterior_path, 'rb')
posterior_dict = pickle.load(posterior_file)
posterior_file.close()

emissions = data['emissions']
inputs = data['inputs']
cell_ids = data['cell_ids']

if cell_ids_chosen is None:
    cell_ids_chosen = cell_ids.copy()

post_pred = posterior_dict['post_pred']
post_pred_noise = posterior_dict['post_pred_noise']
posterior = posterior_dict['posterior']

# au.plot_data(data)
# au.plot_log_likelihood(model)
# au.plot_model_params(model, cell_ids_chosen=cell_ids_chosen)
# au.plot_dynamics_eigs(model)
# au.plot_posterior(data, posterior_dict, cell_ids_chosen, sample_rate=model.sample_rate)
au.plot_stim_l2_norm(model, data, posterior_dict, cell_ids_chosen, window=(0, 120))
au.plot_stim_response(data, posterior_dict, cell_ids_chosen, neuron_to_stim, window=(-60, 120), sample_rate=model.sample_rate)

# au.plot_missing_neuron(model, emissions[data_ind_chosen], inputs[data_ind_chosen], posterior[data_ind_chosen], cell_ids, neuron_to_remove, time_window, sample_rate=model.sample_rate)

