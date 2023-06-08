import numpy as np
import torch
import loading_utilities as lu
from matplotlib import pyplot as plt
import analysis_utilities as au

# the goal of this function is to take the pairwise stimulation and response data from
# https://arxiv.org/abs/2208.04790
# this data is a collection of calcium recordings of ~200 neurons over ~5-15 minutes where individual neurons are
# randomly targets and stimulated optogenetically
# We want to fit a linear dynamical system to the data in order to infer the connection weights between neurons
# The model is of the form
# x_t = A @ x_(t-1) + B @ u_t + w_t
# y_t = C @ x_t + D @ u_t + v_t

# The code should work with different parameters, but for my normal use case
# C is the identity
# B is diagonal
# D is the zero matrix
# w_t, v_t are gaussian with 0 mean

# get run parameters, yaml file contains descriptions of the parameters
run_params = lu.get_run_params(param_name='params')

# rank 0 is the parent node which will send out the data to the children nodes
# set the device (cpu / gpu) and data type
device = run_params['device']
dtype = getattr(torch, run_params['dtype'])

# load in the data for the model and do any preprocessing here
emissions, inputs, cell_ids = \
    lu.load_and_preprocess_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                                bad_data_sets=run_params['bad_data_sets'],
                                frac_neuron_coverage=run_params['frac_neuron_coverage'],
                                minimum_frac_measured=run_params['minimum_frac_measured'],
                                start_index=run_params['start_index'])

chosen_neuron = 'RMDL'
cell_ids = list(cell_ids)
chosen_neuron_ind = cell_ids.index(chosen_neuron)
window = (-10, 20)
measured_stim_responses = au.get_stim_response(emissions, inputs, window=window)

response = measured_stim_responses[:, chosen_neuron_ind, chosen_neuron_ind]

plot_x = np.arange(window[0], window[1])
plt.figure()
plt.scatter(plot_x, response)
plt.axvline(0, color='k', linestyle='--')
plt.axhline(0, color='k', linestyle='--')
plt.show()
a=1
