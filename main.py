import numpy as np
import torch
import loading_utilities as lu
import plotting
from ssm_classes import Lgssm
from mpi4py import MPI
import inference_utilities as iu


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

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_parallel = size > 1

# get run parameters, yaml file contains descriptions of the parameters
run_params = lu.get_params(param_name='params')

if rank == 0:
    device = run_params['device']
    dtype = getattr(torch, run_params['dtype'])

    emissions, inputs, cell_ids = \
        lu.get_model_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                          bad_data_sets=run_params['bad_data_sets'],
                          frac_neuron_coverage=run_params['frac_neuron_coverage'],
                          minimum_frac_measured=run_params['minimum_frac_measured'],
                          start_index=run_params['start_index'])

    # lag the inputs
    num_neurons = emissions[0].shape[1]
    inputs = [Lgssm._get_lagged_data(i, run_params['dynamics_input_lags']) for i in inputs]

    # create a mask for the dynamics_input_weights
    input_mask = torch.eye(num_neurons, dtype=dtype, device=device)
    has_stims = np.any(np.concatenate(inputs, axis=0), axis=0)
    inputs = [i[:, has_stims] for i in inputs]
    input_mask = input_mask[:, has_stims]
    run_params['param_props']['mask']['dynamics_input_weights'] = input_mask
    input_dim = inputs[0].shape[1]

    model_trained = Lgssm(num_neurons, num_neurons, input_dim,
                          dynamics_lags=run_params['dynamics_lags'],
                          dynamics_input_lags=run_params['dynamics_input_lags'],
                          dtype=dtype, device=device, verbose=run_params['verbose'],
                          param_props=run_params['param_props'])

    model_trained.emissions_weights = torch.eye(model_trained.emissions_dim, model_trained.dynamics_dim_full, device=device, dtype=dtype)
    model_trained.emissions_input_weights = torch.zeros((model_trained.emissions_dim, model_trained.input_dim_full), device=device, dtype=dtype)
else:
    emissions = None
    inputs = None
    cell_ids = None
    model_trained = None

model_trained = iu.fit_em(model_trained, emissions, inputs, num_steps=run_params['num_train_steps'],
                          is_parallel=is_parallel, save_folder=run_params['model_save_folder'])

if rank == 0:
    lu.save_run(run_params['model_save_folder'], model_trained,
                data={'emissions': emissions, 'inputs': inputs, 'cell_ids': cell_ids}, run_params=run_params)

    if run_params['plot_figures']:
        plotting.plot_model_params(model_trained)

