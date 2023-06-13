import numpy as np
import pickle
import analysis_utilities as au
import matplotlib as mpl
from pathlib import Path


# load in the model and training data
# model_folder = Path('/home/mcreamer/Documents/data_sets/fun_con_models/48203609_DL5_IL60/')
model_folder = Path('/home/mcreamer/Documents/data_sets/fun_con_models/48196084_DL3_IL45/')
cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
# cell_ids_chosen = ['AVAL', 'AVAR', 'ADAL', 'ADAR', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'ADLL', 'ADLR']
# cell_ids_chosen = ['AVDR', 'AVER', 'AVJR', 'RMDL', 'SAADL']
neuron_to_remove = 'AVDL'
neuron_to_stim = 'AVER'
sample_rate = 0.5

window_size = 1000
colormap = mpl.colormaps['coolwarm']

# load in the model and the data
model_path = model_folder / 'model_trained.pkl'
data_path = model_folder / 'data.pkl'
posterior_path = model_folder / 'posterior.pkl'
prior_path = model_folder / 'prior.pkl'
prior_w_noise_path = model_folder / 'prior_w_noise.pkl'

model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()

data_file = open(data_path, 'rb')
data = pickle.load(data_file)
data_file.close()

emissions = data['emissions']
inputs = data['inputs']
cell_ids = data['cell_ids']
cell_ids = [i for i in cell_ids]

# check if prior has been generated yet. if not generate it
if prior_path.exists():
    prior_file = open(prior_path, 'rb')
    prior = pickle.load(prior_file)
    prior_file.close()

    prior_w_noise_file = open(prior_w_noise_path, 'rb')
    prior_w_noise = pickle.load(prior_w_noise_file)
    prior_w_noise_file.close()

else:
    print('Prior has not been sampled yet, this will take some time')
    print('Output will be saved and will not need to be generated again')
    prior = model.sample([i.shape[0] for i in emissions], num_data_sets=len(inputs),
                         inputs_list=inputs, add_noise=False)['emissions']
    prior_w_noise = model.sample([i.shape[0] for i in emissions], num_data_sets=len(inputs),
                                 inputs_list=inputs, add_noise=True)['emissions']

    prior_file = open(prior_path, 'wb')
    pickle.dump(prior, prior_file)
    prior_file.close()

    prior_w_noise_file = open(prior_w_noise_path, 'wb')
    pickle.dump(prior_w_noise, prior_w_noise_file)
    prior_w_noise_file.close()

posterior_file = open(posterior_path, 'rb')
posterior = pickle.load(posterior_file)
posterior = [(i @ model.emissions_weights.T).detach().cpu().numpy() for i in posterior]
posterior_file.close()

# pull out specific data sets to show
if cell_ids_chosen is not None:
    neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])
else:
    neuron_inds_chosen = np.arange(len(cell_ids))

# the inputs matrix doesn't match the emissions matrix size because some neurons were not stimulated
# upscale the inputs here so that the location of the stimulation in the input matrix
# matches the neuron that was stimulated
dynamics_input_weights_mask = model.param_props['mask']['dynamics_input_weights'].numpy()
weights_loc = ~np.all(dynamics_input_weights_mask == 0, axis=1)
inputs_full = []

for i in inputs:
    inputs_full.append(np.zeros((i.shape[0], model.dynamics_dim)))
    inputs_full[-1][:, weights_loc] = i

# get all the inputs but with only the chosen neurons
inputs_truncated = [i[:, neuron_inds_chosen] for i in inputs_full]
inputs_truncated.pop(3)
data_ind_chosen, time_window = au.find_stim_events(inputs_truncated, window_size=window_size)
data_ind_chosen += 1

emissions_chosen = emissions[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
inputs_chosen = inputs_full[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
posterior_chosen = posterior[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
prior_chosen = prior_w_noise[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]

# au.plot_log_likelihood(model)
# au.plot_model_params(model, cell_ids, cell_ids_chosen=cell_ids_chosen)
# au.plot_dynamics_eigs(model)
au.plot_posterior(emissions_chosen, inputs_chosen, posterior_chosen, prior_chosen, cell_ids_chosen)
# au.plot_missing_neuron(model, emissions[data_ind_chosen], inputs[data_ind_chosen], posterior[data_ind_chosen], cell_ids, neuron_to_remove, time_window)
# au.plot_stim_l2_norm(model, emissions, inputs_full, posterior, prior, cell_ids, cell_ids_chosen, window=(0, 120))
# au.plot_stim_response(emissions, inputs_full, posterior, prior, cell_ids, cell_ids_chosen, neuron_to_stim, window=(-60, 120))


