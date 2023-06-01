import torch
from ssm_classes import Lgssm
import loading_utilities as lu
import numpy as np
import time
from mpi4py import MPI
import inference_utilities as iu
import plotting


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

run_params = lu.get_run_params(param_name='params_synth')
is_parallel = size > 1

if rank == 0:
    device = run_params['device']
    dtype = getattr(torch, run_params['dtype'])
    rng = np.random.default_rng(run_params['random_seed'])

    # define the model, setting specific parameters
    model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                       dtype=dtype, device=device, param_props=run_params['param_props'],
                       dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'])
    model_true.randomize_weights(rng=rng)
    model_true.emissions_weights_init = np.eye(model_true.emissions_dim, model_true.dynamics_dim_full)
    model_true.emissions_input_weights_init = np.zeros((model_true.emissions_dim, model_true.input_dim_full))
    model_true.set_to_init()

    start = time.time()
    # sample from the randomized model
    data_dict = \
        model_true.sample(num_time=run_params['num_time'],
                          num_data_sets=run_params['num_data_sets'],
                          scattered_nan_freq=run_params['scattered_nan_freq'],
                          lost_emission_freq=run_params['lost_emission_freq'],
                          input_time_scale=run_params['input_time_scale'],
                          rng=rng)
    print('Time to sample:', time.time() - start, 's')

    emissions = data_dict['emissions']
    inputs = data_dict['inputs']
    latents_true = data_dict['latents']
    init_mean_true = data_dict['init_mean']
    init_cov_true = data_dict['init_cov']

    # get the log likelihood of the true data
    ll_true_params = model_true.get_ll(emissions, inputs, init_mean_true, init_cov_true)
    model_true.log_likelihood = [ll_true_params]

    # make a new model to fit to the random model
    model_trained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                          dtype=dtype, device=device, verbose=run_params['verbose'], param_props=run_params['param_props'],
                          dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'])
    for k in model_trained.param_props['update'].keys():
        if not model_trained.param_props['update'][k]:
            init_key = k + '_init'
            setattr(model_trained, init_key, getattr(model_true, init_key))
    model_trained.set_to_init()

    lu.save_run(run_params['model_save_folder'], model_trained, model_true=model_true,
                data={'emissions': emissions, 'inputs': inputs, 'cell_ids': model_true.cell_ids}, run_params=run_params,
                remove_old=True)
else:
    emissions = None
    inputs = None
    model_trained = None
    model_true = None

model_trained, smoothed_means = iu.fit_em(model_trained, emissions, inputs, num_steps=run_params['num_train_steps'],
                                          save_folder=run_params['model_save_folder'])

if rank == 0:
    if not is_parallel and run_params['plot_figures']:
        plotting.plot_model_params(model_trained, model_true=model_true)

