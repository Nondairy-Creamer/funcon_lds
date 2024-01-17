from ssm_classes import Lgssm
import loading_utilities as lu
import numpy as np
import time
from mpi4py import MPI
from mpi4py.util import pkl5
import inference_utilities as iu
import analysis_methods as am
import os
import pickle
from pathlib import Path


def fit_synthetic(param_name, save_folder):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    cpu_id = comm.Get_rank()
    is_parallel = size > 1

    run_params = lu.get_run_params(param_name=param_name)

    if cpu_id == 0:
        rng = np.random.default_rng(run_params['random_seed'])

        # define the model, setting specific parameters
        model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                           dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'],
                           emissions_input_lags=run_params['emissions_input_lags'], param_props=run_params['param_props'],)
        model_true.randomize_weights(rng=rng)
        emission_weights_values = rng.uniform(size=(model_true.emissions_dim, model_true.dynamics_lags))
        emission_weights_values = emission_weights_values / np.sum(emission_weights_values, axis=1, keepdims=True)
        emissions_weights_list = [np.diag(emission_weights_values[:, i]) for i in range(emission_weights_values.shape[1])]
        model_true.emissions_weights_init = np.concatenate(emissions_weights_list, axis=1)
        model_true.emissions_input_weights_init = np.zeros(model_true.emissions_input_weights_init.shape)
        model_true.set_to_init()

        start = time.time()
        # sample from the randomized model
        data_train = \
            model_true.sample(num_time=run_params['num_time'],
                              num_data_sets=run_params['num_data_sets'],
                              scattered_nan_freq=run_params['scattered_nan_freq'],
                              lost_emission_freq=run_params['lost_emission_freq'],
                              input_time_scale=run_params['input_time_scale'],
                              rng=rng)

        data_test = \
            model_true.sample(num_time=run_params['num_time'],
                              num_data_sets=run_params['num_data_sets'],
                              scattered_nan_freq=run_params['scattered_nan_freq'],
                              lost_emission_freq=run_params['lost_emission_freq'],
                              input_time_scale=run_params['input_time_scale'],
                              rng=rng)
        print('Time to sample:', time.time() - start, 's')

        # make a new model to fit to the random model
        model_trained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                              verbose=run_params['verbose'], param_props=run_params['param_props'],
                              dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'],
                              emissions_input_lags=run_params['emissions_input_lags'], ridge_lambda=run_params['ridge_lambda'])

        # for any value that we are not fitting, set it to the true value
        for k in model_trained.param_props['update'].keys():
            if not model_trained.param_props['update'][k]:
                init_key = k + '_init'
                setattr(model_trained, init_key, getattr(model_true, init_key))

        model_trained.set_to_init()

        lu.save_run(save_folder, model_true=model_true, model_trained=model_trained, ep=0, data_train=data_train,
                    data_test=data_test, params=run_params)
    else:
        model_trained = None
        data_train = None
        data_test = None
        model_true = None

    # get the log likelihood of the true data
    ll_true_params = iu.parallel_get_ll(model_true, data_train)

    if cpu_id == 0:
        print('log likelihood of true parameters: ', ll_true_params)

        model_true.log_likelihood = [ll_true_params]
        lu.save_run(save_folder, model_true=model_true)

    run_fitting(run_params, model_trained, data_train, data_test, save_folder, model_true=model_true)


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
    cpu_id = comm.Get_rank()

    run_params = lu.get_run_params(param_name=param_name)

    # cpu_id 0 is the parent node which will send out the data to the children nodes
    if cpu_id == 0:
        # load in the data for the model and do any preprocessing here
        data_train, data_test = \
            lu.load_and_preprocess_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                                        start_index=run_params['start_index'],
                                        force_preprocess=run_params['force_preprocess'],
                                        correct_photobleach=run_params['correct_photobleach'],
                                        interpolate_nans=run_params['interpolate_nans'],
                                        held_out_data=run_params['held_out_data'],
                                        neuron_freq=run_params['neuron_freq'],
                                        filter_size=run_params['filter_size'],
                                        hold_out=run_params['hold_out'])

        # initialize the model and set model weights
        num_neurons = data_train['emissions'][0].shape[1]
        model_trained = Lgssm(num_neurons, num_neurons, num_neurons,
                              dynamics_lags=run_params['dynamics_lags'],
                              dynamics_input_lags=run_params['dynamics_input_lags'],
                              emissions_input_lags=run_params['emissions_input_lags'],
                              verbose=run_params['verbose'],
                              param_props=run_params['param_props'],
                              ridge_lambda=run_params['ridge_lambda'],
                              cell_ids=data_train['cell_ids'])

        # model_trained.emissions_weights = np.eye(model_trained.emissions_dim, model_trained.dynamics_dim_full)
        model_trained.emissions_input_weights = np.zeros(model_trained.emissions_input_weights.shape)

        # permute the mask for the dynamics weights so that it is a randomized version
        if 'permute_mask' in run_params:
            if run_params['permute_mask']:
                rng = np.random.default_rng(run_params['random_seed'])

                old_mask = model_trained.param_props['mask']['dynamics_weights']
                new_inds_row = rng.permutation(model_trained.dynamics_dim)
                new_inds_col = [i * model_trained.dynamics_dim + new_inds_row for i in range(model_trained.dynamics_lags)]
                new_inds_col = np.concatenate(new_inds_col)
                new_mask = old_mask[np.ix_(new_inds_row, new_inds_col)]
                model_trained.param_props['mask']['dynamics_weights'] = new_mask

        lu.save_run(save_folder, model_trained=model_trained, ep=0, data_train=data_train, data_test=data_test, params=run_params)

    else:
        # if you are a child node, just set everything to None and only calculate your sufficient statistics
        model_trained = None
        data_train = None
        data_test = None

    run_fitting(run_params, model_trained, data_train, data_test, save_folder)


def infer_posterior(param_name, data_folder, infer_missing=False):
    # fit a posterior to test data
    # set up the option to parallelize the model fitting over CPUs
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()

    run_params = lu.get_run_params(param_name=param_name)

    if run_params['use_memmap']:
        memmap_cpu_id = cpu_id
    else:
        memmap_cpu_id = None

    # cpu_id 0 is the parent node which will send out the data to the children nodes
    if cpu_id == 0:
        data_folder = Path(data_folder)
        model_path = data_folder / 'models' / 'model_trained.pkl'
        data_train_path = data_folder / 'data_train.pkl'
        data_test_path = data_folder / 'data_test.pkl'

        # load in the model
        model_file = open(model_path, 'rb')
        model = pickle.load(model_file)
        model_file.close()

        # load in the data
        data_train_file = open(data_train_path, 'rb')
        data_train = pickle.load(data_train_file)
        data_train_file.close()

        data_test_file = open(data_test_path, 'rb')
        data_test = pickle.load(data_test_file)
        data_test_file.close()

        posterior_train_path = data_folder / 'posterior_train.pkl'
        if posterior_train_path.exists():
            posterior_train_file = open(posterior_train_path, 'rb')
            posterior_train = pickle.load(posterior_train_file)
            posterior_train_file.close()
            emissions_offset_train = posterior_train['emissions_offset']
            init_mean_train = posterior_train['init_mean']
            init_cov_train = posterior_train['init_cov']
        else:
            emissions_offset_train = None
            init_mean_train = None
            init_cov_train = None

        posterior_test_path = data_folder / 'posterior_test.pkl'
        if posterior_test_path.exists():
            posterior_test_file = open(posterior_test_path, 'rb')
            posterior_test = pickle.load(posterior_test_file)
            posterior_test_file.close()
            emissions_offset_test = posterior_test['emissions_offset']
            init_mean_test = posterior_test['init_mean']
            init_cov_test = posterior_test['init_cov']
        else:
            emissions_offset_test = None
            init_mean_test = None
            init_cov_test = None
    else:
        model = None
        data_train = None
        data_test = None
        emissions_offset_train = None
        init_mean_train = None
        init_cov_train = None
        emissions_offset_test = None
        init_mean_test = None
        init_cov_test = None

    posterior_train = iu.parallel_get_post(model, data_train, max_iter=100, memmap_cpu_id=memmap_cpu_id, time_lim=300,
                                           emissions_offset=emissions_offset_train, init_mean=init_mean_train,
                                           init_cov=init_cov_train, infer_missing=infer_missing)
    posterior_test = iu.parallel_get_post(model, data_test, max_iter=100, memmap_cpu_id=memmap_cpu_id, time_lim=300,
                                          emissions_offset=emissions_offset_test, init_mean=init_mean_test,
                                          init_cov=init_cov_test, infer_missing=infer_missing)

    if cpu_id == 0:
        lu.save_run(data_folder, posterior_train=posterior_train, posterior_test=posterior_test)


def continue_fit(param_name, save_folder, extra_train_steps):
    # set up the option to parallelize the model fitting over CPUs
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    cpu_id = comm.Get_rank()

    run_params = lu.get_run_params(param_name=param_name)
    run_params['num_train_steps'] = extra_train_steps

    # cpu_id 0 is the parent node which will send out the data to the children nodes
    if cpu_id == 0:
        save_folder = Path(save_folder)
        # load in the data for the model and do any preprocessing here
        data_train_path = save_folder / 'data_train.pkl'
        data_train_file = open(data_train_path, 'rb')
        data_train = pickle.load(data_train_file)
        data_train_file.close()

        data_test_path = save_folder / 'data_test.pkl'
        data_test_file = open(data_test_path, 'rb')
        data_test = pickle.load(data_test_file)
        data_test_file.close()

        posterior_train_path = save_folder / 'posterior_train.pkl'
        posterior_train_file = open(posterior_train_path, 'rb')
        posterior_train = pickle.load(posterior_train_file)
        posterior_train_file.close()

        posterior_test_path = save_folder / 'posterior_test.pkl'
        if posterior_test_path.exists():
            posterior_test_file = open(posterior_test_path, 'rb')
            posterior_test = pickle.load(posterior_test_file)
            posterior_test_file.close()
            emissions_offset_test = posterior_test['emissions_offset']
            init_mean_test = posterior_test['init_mean']
            init_cov_test = posterior_test['init_cov']
        else:
            emissions_offset_test = None
            init_mean_test = None
            init_cov_test = None

        model_path = save_folder / 'models' / 'model_trained.pkl'
        model_file = open(model_path, 'rb')
        model_trained = pickle.load(model_file)
        model_file.close()

        emissions_offset_train = posterior_train['emissions_offset']
        init_mean_train = posterior_train['init_mean']
        init_cov_train = posterior_train['init_cov']
        starting_step = len(model_trained.log_likelihood)

    else:
        # if you are a child node, just set everything to None and only calculate your sufficient statistics
        model_trained = None
        data_train = None
        data_test = None
        emissions_offset_train = None
        init_mean_train = None
        init_cov_train = None
        emissions_offset_test = None
        init_mean_test = None
        init_cov_test = None
        starting_step = 0

    run_fitting(run_params, model_trained, data_train, data_test, save_folder, starting_step=starting_step,
                emissions_offset_train=emissions_offset_train,
                init_mean_train=init_mean_train, init_cov_train=init_cov_train,
                emissions_offset_test=emissions_offset_test,
                init_mean_test=init_mean_test, init_cov_test=init_cov_test)


def run_fitting(run_params, model, data_train, data_test, save_folder, model_true=None, starting_step=0,
                emissions_offset_train=None, emissions_offset_test=None,
                init_mean_train=None, init_cov_train=None,
                init_mean_test=None, init_cov_test=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    cpu_id = comm.Get_rank()
    is_parallel = size > 1

    # if memory gets to big, use memmap. Reduces speed but significantly reduces memory
    if run_params['use_memmap']:
        memmap_cpu_id = cpu_id
    else:
        memmap_cpu_id = None

    if cpu_id == 0:
        if emissions_offset_train is None:
            emissions_offset_train = model.estimate_emissions_offset(data_train['emissions'])

        if init_mean_train is None:
            init_mean_train = model.estimate_init_mean(data_train['emissions'])

        if init_cov_train is None:
            init_cov_train = model.estimate_init_cov(data_train['emissions'])

    # fit the model using expectation maximization
    ll, model, emissions_offset_train, init_mean_train, init_cov_train = \
        iu.fit_em(model, data_train, num_steps=run_params['num_train_steps'],
                  emissions_offset=emissions_offset_train, init_mean=init_mean_train, init_cov=init_cov_train,
                  save_folder=save_folder, memmap_cpu_id=memmap_cpu_id, starting_step=starting_step)

    # sample from the model
    if cpu_id == 0:
        print('get posterior for the training data')
    posterior_train = iu.parallel_get_post(model, data_train, emissions_offset=emissions_offset_train,
                                           init_mean=init_mean_train, init_cov=init_cov_train,
                                           max_iter=100, converge_res=1e-2, time_lim=300,
                                           memmap_cpu_id=memmap_cpu_id, infer_missing=False)

    if cpu_id == 0:
        print('get posterior for the test data')
    posterior_test = iu.parallel_get_post(model, data_test, emissions_offset=emissions_offset_test,
                                          init_mean=init_mean_test, init_cov=init_cov_test,
                                          max_iter=100, converge_res=1e-2, time_lim=300, memmap_cpu_id=memmap_cpu_id,
                                          infer_missing=False)

    if cpu_id == 0:
        lu.save_run(save_folder, model_trained=model, ep=-1, posterior_train=posterior_train,
                    posterior_test=posterior_test)

        if run_params['use_memmap']:
            for i in range(size):
                os.remove('/tmp/filtered_covs_' + str(i) + '.tmp')

        if not is_parallel and run_params['plot_figures']:
            am.plot_model_params(model, model_true=model_true)

