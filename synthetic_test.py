import torch
from ssm_classes import Lgssm
import preprocessing as pp
import plotting
import pickle
import numpy as np
import time
from mpi4py import MPI
import utilities as utils
import scipy.io as sio


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

run_params = pp.get_params(param_name='params_synth')
is_parallel = size > 1

if rank == 0:
    device = run_params['device']
    dtype = getattr(torch, run_params['dtype'])
    rng = np.random.default_rng(run_params['random_seed'])

    # define the model, setting specific parameters
    model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                       dtype=dtype, device=device, param_props=run_params['param_props'],
                       num_lags=run_params['num_lags'])
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
                          num_lags=run_params['num_lags'])
    for k in model_trained.param_props['update'].keys():
        if not model_trained.param_props['update'][k]:
            init_key = k + '_init'
            setattr(model_trained, init_key, getattr(model_true, init_key))
    model_trained.set_to_init()

    # save the data
    data_dict['params_init'] = model_trained.get_params()
    data_dict['params_init']['init_mean'] = init_mean_true
    data_dict['params_init']['init_cov'] = init_cov_true
    data_dict['params_true'] = model_true.get_params()
    data_dict['params_true']['init_mean'] = init_mean_true
    data_dict['params_true']['init_cov'] = init_cov_true

    save_file = open(run_params['synth_data_save_folder'] + '/data.pkl', 'wb')
    pickle.dump(data_dict, save_file)
    save_file.close()

    sio.savemat(run_params['synth_data_save_folder'] + '/data.mat', data_dict)

else:
    emissions = None
    inputs = None
    model_trained = None

model_trained = utils.fit_em(model_trained, emissions, inputs, num_steps=run_params['num_grad_steps'], is_parallel=is_parallel)

if rank == 0:
    # save the model
    model_true.save(path=run_params['model_save_folder'] + '/model_true.pkl')
    model_trained.save(path=run_params['model_save_folder'] + '/model_trained.pkl')

    # plotting
    if run_params['plot_figures']:
        plotting.plot_model_params(model_trained, model_true)

