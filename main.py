import numpy as np
import torch
import preprocessing as pp
from ssm_classes import Lgssm
import plotting

from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
# from torch.utils.tensorboard import SummaryWriter

# the goal of this function is to take the pairwise stimulation and response data from
# https://arxiv.org/abs/2208.04790
# this data is a collection of calcium recordings of ~100 neurons over ~5-15 minutes where individual neurons are
# randomly targets and stimulated optogenetically
# We want to fit a linear dynamical system to the data in order to infer the connection weights between neurons
# The model is of the form
# x_t = A @ x_(t-1) + V @ u_t + w_t
# y_t = x_t + v_t

# the mapping of x_t to y_t is the identity
# V is diagonal
# w_t, v_t are gaussian with 0 mean, and diagonal covariance

# get run parameters, yaml file contains descriptions of the parameters
run_params = pp.get_params(param_name='params')

device = run_params['device']
dtype = getattr(torch, run_params['dtype'])

# load all the recordings
emissions_unaligned, cell_ids_unaligned, q, q_labels, stim_cell_ids, inputs_unaligned = \
    pp.load_data(run_params['data_path'])

# remove recordings that are noisy
data_sets_to_remove = np.sort(run_params['bad_data_sets'])[::-1]
for bd in data_sets_to_remove:
    emissions_unaligned.pop(bd)
    cell_ids_unaligned.pop(bd)
    inputs_unaligned.pop(bd)
    stim_cell_ids.pop(bd)

emissions_unaligned = emissions_unaligned[:2]
cell_ids_unaligned = cell_ids_unaligned[:2]
inputs_unaligned = inputs_unaligned[:2]
stim_cell_ids = stim_cell_ids[:2]

# choose a subset of the data sets to maximize the number of recordings * the number of neurons included
cell_ids, emissions, best_runs, inputs = \
    pp.get_combined_dataset(emissions_unaligned, cell_ids_unaligned, stim_cell_ids, inputs_unaligned,
                            frac_neuron_coverage=run_params['frac_neuron_coverage'],
                            minimum_freq=run_params['minimum_frac_measured'])

num_data = len(emissions)
num_epochs = int(np.ceil(run_params['num_grad_steps'] * run_params['batch_size'] / num_data))
num_neurons = emissions[0].shape[1]

# remove the beginning of the recording which contains artifacts and mean subtract
for ri in range(num_data):
    emissions[ri] = emissions[ri][run_params['index_start']:, :]
    emissions[ri] = emissions[ri] - np.mean(emissions[ri], axis=0, keepdims=True)
    inputs[ri] = inputs[ri][run_params['index_start']:, :]

# initialize a linear gaussian ssm model and train
param_props = {'update': {'dynamics_offset': False,
                          'emissions_weights': False,
                          'emissions_offset': False,
                          },
               'shape': {'dynamics_input_weights': 'diag',
                         'dynamics_cov': 'diag',
                         'emissions_cov': 'diag'}}

model_trained = Lgssm(num_neurons, num_neurons, num_neurons,
                      dtype=dtype, device=device, verbose=run_params['verbose'], param_props=param_props)

model_trained.emissions_weights = torch.eye(model_trained.dynamics_dim,
                                            device=model_trained.device,
                                            dtype=model_trained.dtype)

model_trained.fit_gd(emissions, inputs,
                     learning_rate=run_params['learning_rate'],
                     num_steps=run_params['num_grad_steps'])

if run_params['save_model']:
    model_trained.save(path=run_params['save_folder'] + '/model_trained.pkl')

if run_params['plot_figures']:
    plotting.trained_on_real(model_trained)

