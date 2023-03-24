import numpy as np
import torch
import preprocessing as pp
from ssm_classes import LgssmSimple
import plotting
from ssm_jax import LgssmJax
import jax.random as jr
from jax import numpy as jnp
import optax


def nan_stack(emissions_list, inputs_list):
    data_set_time = [i.shape[0] for i in emissions_list]
    max_time = np.max(data_set_time)
    num_data_sets = len(emissions_list)
    num_neurons = emissions_list[0].shape[1]

    emissions = np.empty((num_data_sets, max_time, num_neurons))
    emissions[:] = np.nan
    emissions = np.array(emissions)
    inputs = np.zeros((num_data_sets, max_time, num_neurons))

    for d in range(num_data_sets):
        emissions[d, :data_set_time[d], :] = emissions_list[d]
        inputs[d, :data_set_time[d], :] = inputs_list[d]

    return jnp.array(emissions), jnp.array(inputs)

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
params = pp.get_params(param_name='params')

device = params['device']
dtype = getattr(torch, params['dtype'])

# load all the recordings
emissions_unaligned, cell_ids_unaligned, q, q_labels, stim_cell_ids, inputs_unaligned = \
    pp.load_data(params['data_path'])

# remove recordings that are noisy
data_sets_to_remove = np.sort(params['bad_data_sets'])[::-1]
for bd in data_sets_to_remove:
    emissions_unaligned.pop(bd)
    cell_ids_unaligned.pop(bd)
    inputs_unaligned.pop(bd)
    stim_cell_ids.pop(bd)

emissions_unaligned = emissions_unaligned[0:2]
cell_ids_unaligned = cell_ids_unaligned[0:2]
inputs_unaligned = inputs_unaligned[0:2]
stim_cell_ids = stim_cell_ids[0:2]

# choose a subset of the data sets to maximize the number of recordings * the number of neurons included
cell_ids, emissions, best_runs, inputs = \
    pp.get_combined_dataset(emissions_unaligned, cell_ids_unaligned, stim_cell_ids, inputs_unaligned,
                            frac_neuron_coverage=params['frac_neuron_coverage'],
                            minimum_freq=params['minimum_frac_measured'])

num_data = len(emissions)
num_epochs = int(np.ceil(params['num_grad_steps'] * params['batch_size'] / num_data))

# remove the beginning of the recording which contains artifacts and mean subtract
for ri in range(num_data):
    emissions[ri] = emissions[ri][params['index_start']:, :]
    emissions[ri] = emissions[ri] - np.mean(emissions[ri], axis=0, keepdims=True)
    inputs[ri] = inputs[ri][params['index_start']:, :]

# initialize and train model
latent_dim = len(cell_ids)
test_model = LgssmJax(latent_dim, latent_dim, input_dim=latent_dim)
key = jr.PRNGKey(42)
init_params, param_props = test_model.initialize(key,
                                                 dynamics_bias=jnp.zeros(latent_dim),
                                                 emission_weights=jnp.eye(latent_dim),
                                                 emission_input_weights=jnp.zeros((latent_dim, latent_dim)),
                                                 emission_bias=jnp.zeros(latent_dim),
                                                 )


param_props.dynamics.bias.trainable = False
param_props.emissions.bias.trainable = False
param_props.emissions.weights.trainable = False

num_iters = 20000
alpha = 1e-4
optimizer = optax.adam(alpha)
emissions, inputs = nan_stack(emissions, inputs)

emissions = emissions[:, :100, :]
inputs = inputs[:, :100, :]

test_params, marginal_lls = test_model.fit_sgd(init_params, param_props, emissions,
                                               inputs=inputs, num_epochs=num_iters,
                                               optimizer=optimizer)

if params['plot_figures']:
    plotting.trained_on_real(marginal_lls, test_params, init_params)
