from ssm_classes import Lgssm
import loading_utilities as lu
import numpy as np
import time
from mpi4py import MPI
from mpi4py.util import pkl5
import inference_utilities as iu
import plotting


def fit_synthetic(param_name, save_folder):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    rank = comm.Get_rank()
    is_parallel = size > 1

    run_params = lu.get_run_params(param_name=param_name)

    if rank == 0:
        rng = np.random.default_rng(run_params['random_seed'])

        # define the model, setting specific parameters
        model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                           param_props=run_params['param_props'],
                           dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'])
        model_true.randomize_weights(rng=rng)
        model_true.emissions_weights_init = np.eye(model_true.emissions_dim, model_true.dynamics_dim_full)
        model_true.emissions_input_weights_init = np.zeros((model_true.emissions_dim, model_true.input_dim_full))
        model_true.set_to_init()

        start = time.time()
        # sample from the randomized model
        data_train_dict = \
            model_true.sample(num_time=run_params['num_time'],
                              num_data_sets=run_params['num_data_sets'],
                              scattered_nan_freq=run_params['scattered_nan_freq'],
                              lost_emission_freq=run_params['lost_emission_freq'],
                              input_time_scale=run_params['input_time_scale'],
                              rng=rng)
        data_test_dict = \
            model_true.sample(num_time=run_params['num_time'],
                              num_data_sets=run_params['num_data_sets'],
                              scattered_nan_freq=run_params['scattered_nan_freq'],
                              lost_emission_freq=run_params['lost_emission_freq'],
                              input_time_scale=run_params['input_time_scale'],
                              rng=rng)
        print('Time to sample:', time.time() - start, 's')

        emissions = data_train_dict['emissions']
        inputs = data_train_dict['inputs']
        latents_true = data_train_dict['latents']
        init_mean_true = data_train_dict['init_mean']
        init_cov_true = data_train_dict['init_cov']

        # get the log likelihood of the true data
        ll_true_params = model_true.get_ll(emissions, inputs, init_mean_true, init_cov_true)
        model_true.log_likelihood = [ll_true_params]

        # make a new model to fit to the random model
        model_trained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                              verbose=run_params['verbose'], param_props=run_params['param_props'],
                              dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'],
                              ridge_lambda=run_params['ridge_lambda'])
        for k in model_trained.param_props['update'].keys():
            if not model_trained.param_props['update'][k]:
                init_key = k + '_init'
                setattr(model_trained, init_key, getattr(model_true, init_key))
        model_trained.set_to_init()

        lu.save_run(save_folder, model_trained, model_true=model_true,
                    data_train=data_train_dict, data_test=data_test_dict,
                    run_params=run_params, remove_old=True)
    else:
        emissions = None
        inputs = None
        model_trained = None
        model_true = None

    model_trained, smoothed_means, init_mean, init_cov = \
        iu.fit_em(model_trained, emissions, inputs, num_steps=run_params['num_train_steps'],
                  save_folder=save_folder)

    if rank == 0:
        initial_coniditons = {'init_mean': init_mean, 'init_cov': init_cov}
        lu.save_run(save_folder, model_trained, posterior=smoothed_means,
                    initial_conditions=initial_coniditons)

        if not is_parallel and run_params['plot_figures']:
            plotting.plot_model_params(model_trained, model_true=model_true)


def fit_experimental(param_name, save_folder):
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

    run_params = lu.get_run_params(param_name=param_name)

    # rank 0 is the parent node which will send out the data to the children nodes
    if rank == 0:
        # load in the data for the model and do any preprocessing here
        data_train, data_test = \
            lu.load_and_align_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                                   start_index=run_params['start_index'],
                                   force_preprocess=run_params['force_preprocess'],
                                   correct_photobleach=run_params['correct_photobleach'],
                                   interpolate_nans=run_params['interpolate_nans'],
                                   held_out_data=run_params['held_out_data'])

        emissions = data_train['emissions']
        inputs = data_train['inputs']
        cell_ids = data_train['cell_ids']

        num_neurons = emissions[0].shape[1]
        # create a mask for the dynamics_input_weights. This allows us to fit dynamics weights that are diagonal
        input_mask = np.eye(num_neurons)
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
                              verbose=run_params['verbose'],
                              param_props=run_params['param_props'],
                              ridge_lambda=run_params['ridge_lambda'],
                              cell_ids=cell_ids)

        model_trained.emissions_weights = np.eye(model_trained.emissions_dim, model_trained.dynamics_dim_full)
        model_trained.emissions_input_weights = np.zeros((model_trained.emissions_dim, model_trained.input_dim_full))
        model_trained.cell_ids = cell_ids

        lu.save_run(save_folder, model_trained, remove_old=True,
                    data_train=data_train, data_test=data_test, run_params=run_params)

    else:
        # if you are a child node, just set everything to None and only calculate your sufficient statistics
        emissions = None
        inputs = None
        model_trained = None

    # fit the model using expectation maximization
    model_trained, smoothed_means, init_mean, init_cov = \
        iu.fit_em(model_trained, emissions, inputs, num_steps=run_params['num_train_steps'],
                  save_folder=save_folder)

    if rank == 0:
        initial_coniditons = {'init_mean': init_mean, 'init_cov': init_cov}
        lu.save_run(save_folder, model_trained, posterior=smoothed_means,
                    initial_conditions=initial_coniditons)

        if not is_parallel and run_params['plot_figures']:
            plotting.plot_model_params(model_trained)

