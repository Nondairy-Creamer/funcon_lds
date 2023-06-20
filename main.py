import numpy as np
import torch
import loading_utilities as lu
import plotting
from ssm_classes import Lgssm
from mpi4py import MPI
from mpi4py.util import pkl5
import inference_utilities as iu


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

# set up the option to parallelize the model fitting over CPUs
comm = pkl5.Intracomm(MPI.COMM_WORLD)
size = comm.Get_size()
rank = comm.Get_rank()
is_parallel = size > 1

# get run parameters, yaml file contains descriptions of the parameters
run_params = lu.get_run_params(param_name='params')

# rank 0 is the parent node which will send out the data to the children nodes
if rank == 0:
    # set the device (cpu / gpu) and data type
    device = run_params['device']
    dtype = getattr(torch, run_params['dtype'])

    # load in the data for the model and do any preprocessing here
    emissions, inputs, cell_ids = \
        lu.load_and_align_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                               bad_data_sets=run_params['bad_data_sets'],
                               start_index=run_params['start_index'],
                               force_preprocess=run_params['force_preprocess'],
                               correct_photobleach=run_params['correct_photobleach'],
                               interpolate_nans=run_params['interpolate_nans'])

    num_neurons = emissions[0].shape[1]
    # create a mask for the dynamics_input_weights. This allows us to fit dynamics weights that are diagonal
    input_mask = torch.eye(num_neurons, dtype=dtype, device=device)
    # get rid of any inputs that never receive stimulation
    has_stims = np.any(np.concatenate(inputs, axis=0), axis=0)
    inputs = [i[:, has_stims] for i in inputs]
    input_mask = input_mask[:, has_stims]
    # set the model properties so the model fits with this mask
    run_params['param_props']['mask']['dynamics_input_weights'] = input_mask
    # get the input dimension after removing the neurons that were never stimulated
    input_dim = inputs[0].shape[1]

    # initialize the model and set model weights
    model_trained = Lgssm(num_neurons, num_neurons, input_dim,
                          dynamics_lags=run_params['dynamics_lags'],
                          dynamics_input_lags=run_params['dynamics_input_lags'],
                          dtype=dtype, device=device, verbose=run_params['verbose'],
                          param_props=run_params['param_props'])

    model_trained.emissions_weights = torch.eye(model_trained.emissions_dim, model_trained.dynamics_dim_full, device=device, dtype=dtype)
    model_trained.emissions_input_weights = torch.zeros((model_trained.emissions_dim, model_trained.input_dim_full), device=device, dtype=dtype)
    model_trained.cell_ids = cell_ids

    lu.save_run(run_params['model_save_folder'], model_trained, remove_old=True,
                data={'emissions': emissions, 'inputs': inputs, 'cell_ids': cell_ids}, run_params=run_params)

else:
    # if you are a child node, just set everything to None and only calculate your sufficient statistics
    emissions = None
    inputs = None
    cell_ids = None
    model_trained = None

# fit the model using expectation maximization
model_trained, smoothed_means = iu.fit_em(model_trained, emissions, inputs, num_steps=run_params['num_train_steps'],
                                          save_folder=run_params['model_save_folder'])

if rank == 0:
    lu.save_run(run_params['model_save_folder'], model_trained, posterior=smoothed_means)

    if not is_parallel and run_params['plot_figures']:
        plotting.plot_model_params(model_trained)

