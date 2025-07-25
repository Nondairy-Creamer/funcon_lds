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
import lgssm_utilities as lgssmu
import copy
import metrics as met
import shutil
import torch


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
                           emissions_input_lags=run_params['emissions_input_lags'], param_props=run_params['param_props'])

        model_true.randomize_weights(rng=rng)
        if model_true.param_props['update']['emissions_weights']:
            emission_weights_values = rng.uniform(size=(model_true.emissions_dim, model_true.dynamics_lags))
            emission_weights_values = emission_weights_values / np.sum(emission_weights_values, axis=1, keepdims=True)
            emissions_weights_list = [np.diag(emission_weights_values[:, i]) for i in range(emission_weights_values.shape[1])]
            model_true.emissions_weights_init = np.concatenate(emissions_weights_list, axis=1)
        else:
            model_true.emissions_weights_init = np.eye(model_true.emissions_dim, model_true.dynamics_dim_full)
        model_true.emissions_input_weights_init = np.zeros(model_true.emissions_input_weights_init.shape)
        model_true.set_to_init()

        start = time.time()
        # sample from the randomized model
        data_train = \
            model_true.sample_multiple(num_time=run_params['num_time'],
                                       num_data_sets=run_params['num_data_sets'],
                                       scattered_nan_freq=run_params['scattered_nan_freq'],
                                       lost_emission_freq=run_params['lost_emission_freq'],
                                       input_time_scale=run_params['input_time_scale'],
                                       rng=rng)

        data_test = \
            model_true.sample_multiple(num_time=run_params['num_time'],
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
        if 'upsample_factor' in run_params:
            upsample_factor = run_params['upsample_factor']
        else:
            upsample_factor = 1

        # load in the data for the model and do any preprocessing here
        data_train, data_test = \
            lu.load_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                         held_out_data=run_params['held_out_data'],
                         neuron_freq=run_params['neuron_freq'],
                         hold_out=run_params['hold_out'],
                         upsample_factor=upsample_factor,
                         hold_out_start=run_params['hold_out_start'])

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

        # permute the mask for the dynamics weights so that it is a randomized version
        if 'randomize_weights' in run_params:
            if run_params['randomize_weights']:
                if 'myVar' not in locals():
                    rng = np.random.default_rng(run_params['random_seed'])

                model_trained.randomize_weights(rng=rng)

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
                emissions_offset_train=emissions_offset_train, emissions_offset_test=emissions_offset_test,
                init_mean_train=init_mean_train, init_mean_test=init_mean_test,
                init_cov_train=init_cov_train, init_cov_test=init_cov_test)


def prune_model(param_name, save_folder, extra_train_steps, prune_frac):
    # set up the option to parallelize the model fitting over CPUs
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    cpu_id = comm.Get_rank()

    # this code will load in an existing model then prune connections by removing the model weights closest to 0
    error_frac = np.inf
    pruning_method = ['exponential', 'linear']
    pruning_method = pruning_method[1]
    min_score_frac = 0.9
    window = (15, 30)  # window around which to calculate the eIRFs and IRFs
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

        data_irfs = lgssmu.get_impulse_response_functions(
            data_test['emissions'], data_test['inputs'], sample_rate=data_test['sample_rate'],
            window=window, sub_pre_stim=True)[0]
        data_irms = np.sum(data_irfs[window[0]:, :, :], axis=0)
        data_irms[np.eye(data_irms.shape[0], dtype=bool)] = np.nan

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
        model_base = pickle.load(model_file)
        model_file.close()

        emissions_offset_train = posterior_train['emissions_offset']
        init_mean_train = posterior_train['init_mean']
        init_cov_train = posterior_train['init_cov']

        model_irms_base = lgssmu.calculate_irms(model_base, window=window)
        model_base_score = met.nan_corr(data_irms, model_irms_base)[0]

        prune_folder_str = 'pruning_es' + f'{int(extra_train_steps):03d}' + '_pf' + f'{int(prune_frac * 100):03d}'
        if (save_folder / prune_folder_str).exists():
            shutil.rmtree(save_folder / prune_folder_str)
        os.mkdir(save_folder / prune_folder_str)

        dynamics_dim = model_base.dynamics_dim
        dynamics_lags = model_base.dynamics_lags
        model_dict = {'model': copy.deepcopy(model_base),
                      'init_mean_train': init_mean_train.copy(),
                      'init_mean_test': init_mean_test.copy(),
                      'init_cov_train': init_cov_train.copy(),
                      'init_cov_test': init_cov_test.copy(),
                      'emissions_offset_train': emissions_offset_train.copy(),
                      'emissions_offset_test': emissions_offset_test.copy(),
                      }
    else:
        # if you are a child node, just set everything to None and only calculate your sufficient statistics
        data_train = None
        data_test = None
        model_dict = {'model': None,
                      'init_mean_train': None,
                      'init_mean_test': None,
                      'init_cov_train': None,
                      'init_cov_test': None,
                      'emissions_offset_train': None,
                      'emissions_offset_test': None,
                      }


    num_iter = 0

    while (error_frac > min_score_frac):
        if cpu_id == 0:
            # prune the smallest weights
            current_mask = model_dict['model'].param_props['mask']['dynamics_weights'][:, :dynamics_dim]
            model_weights = lgssmu.calculate_eirms(model_dict['model'], window=window)
            model_weights_no_masked = model_weights.copy()

            # set the diagonal to inf so we always fit it
            model_weights_no_masked[np.eye(model_weights_no_masked.shape[0], dtype=bool)] = np.inf

            if pruning_method == 'exponential':
                # find how many weights to remove as a fraction of the remaining values not masked
                num_weights_remove = np.ceil(prune_frac * np.sum(current_mask)).astype(int)
                # set the current masked weights to inf so that they're not counted among the smallest weights
                model_weights_no_masked[~current_mask] = np.inf
            elif pruning_method == 'linear':
                # find the number of weights to remove as a linear fraction of all the weights in the mask
                num_weights_remove = np.ceil((num_iter + 1) * prune_frac * current_mask.size).astype(int)
            else:
                raise Exception('pruning method not recognized')

            # sort the absolute value of the weights and get the num-weights_remove smallest
            cutoff_value = np.sort(np.abs(model_weights_no_masked).reshape(-1))[num_weights_remove - 1]
            # keep all values larger than the cutoff
            new_mask = np.abs(model_weights) > cutoff_value
            new_mask[np.eye(new_mask.shape[0], dtype=bool)] = True

            # set the masked values to 0 and update the mask
            model_dict['model'].dynamics_weights[:dynamics_dim, :][np.tile(~new_mask, (1, model_dict['model'].dynamics_lags))] = 0
            model_dict['model'].param_props['mask']['dynamics_weights'] = np.tile(new_mask, (1, dynamics_lags))

            save_path_iter = save_folder / prune_folder_str / ('model_iter_' + f'{num_iter:03d}')
            os.mkdir(save_path_iter)
        else:
            save_path_iter = None

        # set all the learned data parameters
        init_mean_train = model_dict['init_mean_train']
        init_mean_test = model_dict['init_mean_test']
        init_cov_train = model_dict['init_cov_train']
        init_cov_test = model_dict['init_cov_test']
        emissions_offset_train = model_dict['emissions_offset_train']
        emissions_offset_test = model_dict['emissions_offset_test']

        model_dict = \
            run_fitting(run_params, model_dict['model'], data_train, data_test, save_path_iter,
                        emissions_offset_train=emissions_offset_train, emissions_offset_test=emissions_offset_test,
                        init_mean_train=init_mean_train, init_mean_test=init_mean_test,
                        init_cov_train=init_cov_train, init_cov_test=init_cov_test, plot_figs=False)

        if cpu_id == 0:
            # get the predicted IRFs from the model and compare them to the data
            model_irms = lgssmu.calculate_irms(model_dict['model'], window=window, verbose=False)
            model_score = met.nan_corr(data_irms, model_irms)[0]

            error_frac = model_score / model_base_score

        error_frac = comm.bcast(error_frac, root=0)

        num_iter += 1


def run_fitting(run_params, model, data_train, data_test, save_folder, model_true=None, starting_step=0,
                emissions_offset_train=None, emissions_offset_test=None,
                init_mean_train=None, init_mean_test=None,
                init_cov_train=None, init_cov_test=None, plot_figs=True):
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
                                           max_iter=50, converge_res=1e-2, time_lim=1000,
                                           memmap_cpu_id=memmap_cpu_id, infer_missing=False)

    if cpu_id == 0:
        print('get posterior for the test data')
    posterior_test = iu.parallel_get_post(model, data_test, emissions_offset=emissions_offset_test,
                                          init_mean=init_mean_test, init_cov=init_cov_test,
                                          max_iter=50, converge_res=1e-2, time_lim=1000,
                                          memmap_cpu_id=memmap_cpu_id, infer_missing=False)

    if cpu_id == 0:
        print('Finished posterior for test data')

        lu.save_run(save_folder, model_trained=model, ep=-1, posterior_train=posterior_train,
                    posterior_test=posterior_test)

        print('finished saving')
        if run_params['use_memmap']:
            for i in range(size):
                os.remove('/tmp/filtered_covs_' + str(i) + '.tmp')

        if not is_parallel and run_params['plot_figures'] and plot_figs:
            am.plot_model_params(model, model_true=model_true)

        model_trained = {'model': model,
                         'init_mean_train': posterior_train['init_mean'],
                         'init_mean_test': posterior_test['init_mean'],
                         'init_cov_train': posterior_train['init_cov'],
                         'init_cov_test': posterior_test['init_cov'],
                         'emissions_offset_train': posterior_train['emissions_offset'],
                         'emissions_offset_test': posterior_test['emissions_offset'],
                         }

    else:
        model_trained = {'model': None,
                         'init_mean_train': None,
                         'init_mean_test': None,
                         'init_cov_train': None,
                         'init_cov_test': None,
                         'emissions_offset_train': None,
                         'emissions_offset_test': None,
                         }

    return model_trained


def fit_mismatch(param_name, save_folder):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    cpu_id = comm.Get_rank()
    is_parallel = size > 1

    plot_color = {'data': np.array([217, 95, 2]) / 255,
                  'synap': np.array([27, 158, 119]) / 255,
                  'unconstrained': np.array([117, 112, 179]) / 255,
                  'synap_randA': np.array([231, 41, 138]) / 255,
                  # 'synap_randC': np.array([102, 166, 30]) / 255,
                  'synap_randC': np.array([128, 128, 128]) / 255,
                  'anatomy': np.array([64, 64, 64]) / 255,
                  'true_model': np.array([60, 60, 245]) / 255,
                  }

    run_params = lu.get_run_params(param_name=param_name)

    true_corr = []
    mismatch_corr = []
    uncon_corr = []

    true_corr_ci = []
    mismatch_corr_ci = []
    uncon_corr_ci = []

    # mask_array = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.175, 0.2, 0.4]
    mask_array = [0.2]
    rng = np.random.default_rng(run_params['random_seed'])
    num_repeats = 10

    for i in range(num_repeats):
        for true_mask_prob in mask_array:
            if cpu_id == 0:
                # rng = np.random.default_rng(run_params['random_seed'])

                mismatch_mask_prob = 0.1
                true_mask = rng.random((run_params['dynamics_dim'], run_params['dynamics_dim'])) < true_mask_prob
                true_mask[np.eye(true_mask.shape[0], dtype=bool)] = True
                mismatch_mask_mult = rng.random((run_params['dynamics_dim'], run_params['dynamics_dim'])) < mismatch_mask_prob / true_mask_prob
                mismatch_mask = true_mask * mismatch_mask_mult
                mismatch_mask[np.eye(mismatch_mask.shape[0], dtype=bool)] = True
                unconstrained_mask = np.ones_like(mismatch_mask)

                from matplotlib import pyplot as plt
                true_mask_plot = np.repeat(true_mask[:, :, None], 3, axis=2) * plot_color['true_model'][None, None, :]
                mismatch_mask_plot = np.repeat(mismatch_mask[:, :, None], 3, axis=2) * plot_color['synap'][None, None, :]
                alpha_channel = mismatch_mask.astype(float)
                mismatch_mask_rgba = np.dstack((mismatch_mask_plot, alpha_channel))

                save_path = '/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/nature_review_appeal/'

                plt.figure()
                plt.imshow(true_mask_plot)
                plt.imshow(mismatch_mask_rgba)
                plt.savefig(save_path + 'synth_model_mask.pdf')
                plt.figure()
                plt.imshow(mismatch_mask_plot)
                plt.savefig(save_path + 'synth_constrained_model_mask.pdf')
                plt.show()


                # define the model, setting specific parameters
                model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                                   dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'],
                                   emissions_input_lags=run_params['emissions_input_lags'], param_props=run_params['param_props'])

                model_true.param_props['mask']['dynamics_weights'] = true_mask

                model_true.randomize_weights(rng=rng)
                if model_true.param_props['update']['emissions_weights']:
                    emission_weights_values = rng.uniform(size=(model_true.emissions_dim, model_true.dynamics_lags))
                    emission_weights_values = emission_weights_values / np.sum(emission_weights_values, axis=1, keepdims=True)
                    emissions_weights_list = [np.diag(emission_weights_values[:, i]) for i in range(emission_weights_values.shape[1])]
                    model_true.emissions_weights_init = np.concatenate(emissions_weights_list, axis=1)
                else:
                    model_true.emissions_weights_init = np.eye(model_true.emissions_dim, model_true.dynamics_dim_full)
                model_true.emissions_input_weights_init = np.zeros(model_true.emissions_input_weights_init.shape)
                model_true.set_to_init()

                # sample from the randomized model
                data_train = \
                    model_true.sample_multiple(num_time=run_params['num_time'],
                                               num_data_sets=run_params['num_data_sets'],
                                               scattered_nan_freq=run_params['scattered_nan_freq'],
                                               lost_emission_freq=run_params['lost_emission_freq'],
                                               input_time_scale=run_params['input_time_scale'],
                                               rng=rng)

                data_test = \
                    model_true.sample_multiple(num_time=run_params['num_time'],
                                               num_data_sets=run_params['num_data_sets'],
                                               scattered_nan_freq=run_params['scattered_nan_freq'],
                                               lost_emission_freq=run_params['lost_emission_freq'],
                                               input_time_scale=run_params['input_time_scale'],
                                               rng=rng)

                # make a new model to fit to the random model
                model_trained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                                      verbose=run_params['verbose'], param_props=run_params['param_props'],
                                      dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'],
                                      emissions_input_lags=run_params['emissions_input_lags'], ridge_lambda=run_params['ridge_lambda'])

                model_trained_mismatch = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                                               verbose=run_params['verbose'], param_props=run_params['param_props'],
                                               dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'],
                                               emissions_input_lags=run_params['emissions_input_lags'], ridge_lambda=run_params['ridge_lambda'])

                model_trained_unconstrained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                                                    verbose=run_params['verbose'], param_props=run_params['param_props'],
                                                    dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'],
                                                    emissions_input_lags=run_params['emissions_input_lags'], ridge_lambda=run_params['ridge_lambda'])

                model_trained.param_props['mask']['dynamics_weights'] = true_mask
                model_trained_mismatch.param_props['mask']['dynamics_weights'] = mismatch_mask
                model_trained_unconstrained.param_props['mask']['dynamics_weights'] = unconstrained_mask

                # for any value that we are not fitting, set it to the true value
                for k in model_trained.param_props['update'].keys():
                    if not model_trained.param_props['update'][k]:
                        init_key = k + '_init'
                        setattr(model_trained, init_key, getattr(model_true, init_key))

                # for any value that we are not fitting, set it to the true value
                for k in model_trained_mismatch.param_props['update'].keys():
                    if not model_trained_mismatch.param_props['update'][k]:
                        init_key = k + '_init'
                        setattr(model_trained_mismatch, init_key, getattr(model_true, init_key))

                # for any value that we are not fitting, set it to the true value
                for k in model_trained_unconstrained.param_props['update'].keys():
                    if not model_trained_unconstrained.param_props['update'][k]:
                        init_key = k + '_init'
                        setattr(model_trained_unconstrained, init_key, getattr(model_true, init_key))

                model_trained.set_to_init()
                model_trained_mismatch.set_to_init()
                model_trained_unconstrained.set_to_init()

                lu.save_run(save_folder, model_true=model_true, model_trained=model_trained, ep=0, data_train=data_train,
                            data_test=data_test, params=run_params)
                lu.save_run(save_folder, model_true=model_true, model_trained=model_trained_mismatch, ep=0, data_train=data_train,
                            data_test=data_test, params=run_params)
                lu.save_run(save_folder, model_true=model_true, model_trained=model_trained_unconstrained, ep=0, data_train=data_train,
                            data_test=data_test, params=run_params)
            else:
                model_trained = None
                model_trained_mismatch = None
                model_trained_unconstrained = None
                data_train = None
                data_test = None
                model_true = None

            # get the log likelihood of the true data
            ll_true_params = iu.parallel_get_ll(model_true, data_train)

            if cpu_id == 0:
                print('log likelihood of true parameters: ', ll_true_params)

                model_true.log_likelihood = [ll_true_params]
                lu.save_run(save_folder, model_true=model_true)

            training_output = run_fitting(run_params, model_trained, data_train, data_test, save_folder, model_true=model_true)
            training_output_mismatch = run_fitting(run_params, model_trained_mismatch, data_train, data_test, save_folder, model_true=model_true)
            training_output_unconstrained = run_fitting(run_params, model_trained_unconstrained, data_train, data_test, save_folder, model_true=model_true)
            model_trained = training_output['model']
            model_trained_mismatch = training_output_mismatch['model']
            model_trained_unconstrained = training_output_unconstrained['model']

            if cpu_id == 0:
                import lgssm_utilities as ssmu
                import metrics as met
                from matplotlib import pyplot as plt

                sample_rate = 2
                window = (15, 30)

                data_irfs_train, data_irfs_sem_train, data_irfs_train_all = \
                    ssmu.get_impulse_response_functions(data_train['emissions'], data_train['inputs'],
                                                        sample_rate=sample_rate, window=(15, 30), sub_pre_stim=True)

                nan_loc = np.all(np.isnan(data_irfs_train), axis=0)
                data_irms_train = np.nansum(data_irfs_train[int(window[0]*sample_rate):], axis=0) / sample_rate
                data_irms_train[nan_loc] = np.nan

                data_irfs_test, data_irfs_sem_test, data_irfs_test_all = \
                    ssmu.get_impulse_response_functions(data_test['emissions'], data_test['inputs'],
                                                        sample_rate=sample_rate, window=(15, 30), sub_pre_stim=True)

                nan_loc = np.all(np.isnan(data_irfs_test), axis=0)
                data_irms_test = np.nansum(data_irfs_test[int(window[0]*sample_rate):], axis=0) / sample_rate
                data_irms_test[nan_loc] = np.nan

                model_irms = ssmu.calculate_irms(model_trained, window=window)
                model_irms_mismatch = ssmu.calculate_irms(model_trained_mismatch, window=window)
                model_irms_unconstrained = ssmu.calculate_irms(model_trained_unconstrained, window=window)

                data_irms_test[np.eye(data_irms_test.shape[0], dtype=bool)] = np.nan
                data_irms_train[np.eye(data_irms_train.shape[0], dtype=bool)] = np.nan
                model_irms[np.eye(model_irms.shape[0], dtype=bool)] = np.nan
                model_irms_mismatch[np.eye(model_irms_mismatch.shape[0], dtype=bool)] = np.nan
                model_irms_unconstrained[np.eye(model_irms_unconstrained.shape[0], dtype=bool)] = np.nan

                train_test_corr = met.nan_corr(data_irms_train, data_irms_test)[0]
                model_irms_score = []
                model_irms_score_ci = []

                for mi, m in enumerate([model_irms, model_irms_unconstrained, model_irms_mismatch]):
                    model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = (
                        met.nan_corr(m, data_irms_test))
                    model_irms_score.append(model_irms_to_measured_irms_test)
                    model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

                    if mi == 0:
                        true_corr.append(model_irms_to_measured_irms_test)
                        true_corr_ci.append(model_irms_to_measured_irms_test_ci)
                    elif mi == 1:
                        uncon_corr.append(model_irms_to_measured_irms_test)
                        uncon_corr_ci.append(model_irms_to_measured_irms_test_ci)
                    elif mi == 2:
                        mismatch_corr.append(model_irms_to_measured_irms_test)
                        mismatch_corr_ci.append(model_irms_to_measured_irms_test_ci)

                # plot average reconstruction over all data
                # y_limits = [-0.25, 1.25]
                # plt.figure()
                # y_val = np.array(model_irms_score)
                # y_val_ci = np.stack(model_irms_score_ci).T
                # plot_x = np.arange(y_val.shape[0])
                # bar_colors = [plot_color['data'], plot_color['unconstrained'], plot_color['synap']]
                # plt.bar(plot_x, y_val, color=bar_colors)
                # plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
                # plt.xticks(plot_x, labels=['true_model', 'unconstrained', 'connectome_constrained'], rotation=45)
                # plt.ylabel('correlation')
                # # plt.ylim(y_limits)
                # plt.tight_layout()
                # save_path = '/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/nature_review_appeal/'
                # # save_path = '/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/'
                # plt.savefig(save_path + 'mismatch.pdf')
                #
                # plt.figure()
                # plt.imshow(true_mask, cmap='gray')
                # plt.savefig(save_path + 'true_mask.pdf')
                #
                # plt.figure()
                # plt.imshow(mismatch_mask, cmap='gray')
                # plt.savefig(save_path + 'mismatch_mask.pdf')
                #
                # plt.show()

    # if cpu_id == 0:
        # plt.figure()
        # plt.plot(mask_array, true_corr)
        # plt.plot(mask_array, uncon_corr)
        # plt.plot(mask_array, mismatch_corr)
        # plt.ylim([0, 1])
        # plt.xlabel('true dynamics sparsity')
        # plt.ylabel('correlation')
        # plt.title('model prediction of STAMs')
        #
        # plt.savefig(save_path + 'mismatch_sweep.pdf')
        #
        # plt.show()
    if cpu_id == 0:
        save_path = '/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/nature_review_appeal/'
        save_dict = {'true_corr': true_corr, 'uncon_corr': uncon_corr, 'mismatch_corr': mismatch_corr}

        save_file = open(save_path + 'mismatch_data.pkl', 'wb')
        pickle.dump(save_dict, save_file)
        save_file.close()
    a=1


def fit_smoothed_mismatch(param_name, save_folder):
    import copy
    from scipy.signal import convolve
    import lgssm_utilities as ssmu
    import metrics as met
    from matplotlib import pyplot as plt

    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    cpu_id = comm.Get_rank()
    is_parallel = size > 1

    plot_color = {'data': np.array([217, 95, 2]) / 255,
                  'synap': np.array([27, 158, 119]) / 255,
                  'unconstrained': np.array([117, 112, 179]) / 255,
                  'synap_randA': np.array([231, 41, 138]) / 255,
                  # 'synap_randC': np.array([102, 166, 30]) / 255,
                  'synap_randC': np.array([128, 128, 128]) / 255,
                  'anatomy': np.array([64, 64, 64]) / 255,
                  'true_model': np.array([60, 60, 245]) / 255,
                  }

    run_params = lu.get_run_params(param_name=param_name)

    mismatch_to_true_weights_corr = []

    true_corr = []
    mismatch_corr = []
    uncon_corr = []

    true_corr_ci = []
    mismatch_corr_ci = []
    uncon_corr_ci = []

    filter_tau = [0.1, 10, 20, 30, 40]
    rng = np.random.default_rng(run_params['random_seed'])
    num_repeats = 10
    filt_length = int(np.max(filter_tau) * 3)

    for i in range(num_repeats):
        true_mask = rng.random((run_params['dynamics_dim'], run_params['dynamics_dim'])) < 0.1
        true_mask[np.eye(true_mask.shape[0], dtype=bool)] = True
        unconstrained_mask = np.ones_like(true_mask)

        # define the model, setting specific parameters
        model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                           dynamics_lags=run_params['dynamics_lags'],
                           dynamics_input_lags=run_params['dynamics_input_lags'],
                           emissions_input_lags=run_params['emissions_input_lags'],
                           param_props=run_params['param_props'])

        model_true.param_props['mask']['dynamics_weights'] = true_mask
        model_true.randomize_weights(rng=rng)
        model_true.emissions_weights_init = np.eye(model_true.emissions_dim, model_true.dynamics_dim_full)
        model_true.emissions_input_weights_init = np.zeros(model_true.emissions_input_weights_init.shape)
        model_true.set_to_init()

        # sample from the randomized model
        data_train = \
            model_true.sample_multiple(num_time=run_params['num_time'],
                                       num_data_sets=run_params['num_data_sets'],
                                       scattered_nan_freq=run_params['scattered_nan_freq'],
                                       lost_emission_freq=run_params['lost_emission_freq'],
                                       input_time_scale=run_params['input_time_scale'],
                                       rng=rng)

        data_test = \
            model_true.sample_multiple(num_time=run_params['num_time'],
                                       num_data_sets=run_params['num_data_sets'],
                                       scattered_nan_freq=run_params['scattered_nan_freq'],
                                       lost_emission_freq=run_params['lost_emission_freq'],
                                       input_time_scale=run_params['input_time_scale'],
                                       rng=rng)

        # make a new model to fit to the data generated from the true model
        model_trained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                              verbose=run_params['verbose'], param_props=run_params['param_props'],
                              dynamics_lags=run_params['dynamics_lags'],
                              dynamics_input_lags=run_params['dynamics_input_lags'],
                              emissions_input_lags=run_params['emissions_input_lags'],
                              ridge_lambda=run_params['ridge_lambda'])

        model_trained_mismatch = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                                       verbose=run_params['verbose'], param_props=run_params['param_props'],
                                       dynamics_lags=run_params['dynamics_lags'],
                                       dynamics_input_lags=run_params['dynamics_input_lags'],
                                       emissions_input_lags=run_params['emissions_input_lags'],
                                       ridge_lambda=run_params['ridge_lambda'])

        model_trained_unconstrained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'],
                                            run_params['input_dim'],
                                            verbose=run_params['verbose'], param_props=run_params['param_props'],
                                            dynamics_lags=run_params['dynamics_lags'],
                                            dynamics_input_lags=run_params['dynamics_input_lags'],
                                            emissions_input_lags=run_params['emissions_input_lags'],
                                            ridge_lambda=run_params['ridge_lambda'])

        model_trained.param_props['mask']['dynamics_weights'] = true_mask
        model_trained_mismatch.param_props['mask']['dynamics_weights'] = true_mask
        model_trained_unconstrained.param_props['mask']['dynamics_weights'] = unconstrained_mask

        # for any value that we are not fitting, set it to the true value
        for k in model_trained.param_props['update'].keys():
            if not model_trained.param_props['update'][k]:
                init_key = k + '_init'
                setattr(model_trained, init_key, getattr(model_true, init_key))

        # for any value that we are not fitting, set it to the true value
        for k in model_trained_mismatch.param_props['update'].keys():
            if not model_trained_mismatch.param_props['update'][k]:
                init_key = k + '_init'
                setattr(model_trained_mismatch, init_key, getattr(model_true, init_key))

        # for any value that we are not fitting, set it to the true value
        for k in model_trained_unconstrained.param_props['update'].keys():
            if not model_trained_unconstrained.param_props['update'][k]:
                init_key = k + '_init'
                setattr(model_trained_unconstrained, init_key, getattr(model_true, init_key))

        model_trained.set_to_init()
        model_trained_mismatch.set_to_init()
        model_trained_unconstrained.set_to_init()

        for tau in filter_tau:
            if cpu_id == 0:
                # create a filtered version of the train data
                data_train_filtered = copy.deepcopy(data_train)
                data_test_filtered = copy.deepcopy(data_test)

                filt = np.exp(-1 / tau * np.arange(filt_length))
                filt = filt[:, None] / np.sum(filt)
                data_train_filtered['emissions'] = [convolve(i, filt, mode='full')[:-filt_length+1, :] for i in data_train_filtered['emissions']]
                data_test_filtered['emissions'] = [convolve(i, filt, mode='full')[:-filt_length+1, :] for i in data_test_filtered['emissions']]

                lu.save_run(save_folder, model_true=model_true, model_trained=model_trained, ep=0, data_train=data_train,
                            data_test=data_test, params=run_params)
                lu.save_run(save_folder, model_true=model_true, model_trained=model_trained_mismatch, ep=0, data_train=data_train_filtered,
                            data_test=data_test, params=run_params)
                lu.save_run(save_folder, model_true=model_true, model_trained=model_trained_unconstrained, ep=0, data_train=data_train_filtered,
                            data_test=data_test, params=run_params)
            else:
                model_trained = None
                model_trained_mismatch = None
                model_trained_unconstrained = None
                data_train = None
                data_train_filtered = None
                data_test = None
                data_test_filtered = None
                model_true = None

            # get the log likelihood of the true data
            ll_true_params = iu.parallel_get_ll(model_true, data_train)

            if cpu_id == 0:
                print('log likelihood of true parameters: ', ll_true_params)

                model_true.log_likelihood = [ll_true_params]
                lu.save_run(save_folder, model_true=model_true)

            training_output = run_fitting(run_params, model_trained, data_train, data_test, save_folder, model_true=model_true)
            training_output_mismatch = run_fitting(run_params, model_trained_mismatch, data_train_filtered, data_test_filtered, save_folder, model_true=model_true)
            training_output_unconstrained = run_fitting(run_params, model_trained_unconstrained, data_train_filtered, data_test_filtered, save_folder, model_true=model_true)
            model_trained = training_output['model']
            model_trained_mismatch = training_output_mismatch['model']
            model_trained_unconstrained = training_output_unconstrained['model']

            if cpu_id == 0:
                sample_rate = 2
                window = (15, 30)

                # measure the impulse responses in the data
                data_irfs_train, data_irfs_sem_train, data_irfs_train_all = \
                    ssmu.get_impulse_response_functions(data_train_filtered['emissions'], data_train_filtered['inputs'],
                                                        sample_rate=sample_rate, window=(15, 30), sub_pre_stim=True)

                nan_loc = np.all(np.isnan(data_irfs_train), axis=0)
                data_irms_train = np.nansum(data_irfs_train[int(window[0]*sample_rate):], axis=0) / sample_rate
                data_irms_train[nan_loc] = np.nan

                data_irfs_test, data_irfs_sem_test, data_irfs_test_all = \
                    ssmu.get_impulse_response_functions(data_test_filtered['emissions'], data_test_filtered['inputs'],
                                                        sample_rate=sample_rate, window=(15, 30), sub_pre_stim=True)

                nan_loc = np.all(np.isnan(data_irfs_test), axis=0)
                data_irms_test = np.nansum(data_irfs_test[int(window[0]*sample_rate):], axis=0) / sample_rate
                data_irms_test[nan_loc] = np.nan

                # calculate the IRMs for each of the models
                model_irms = ssmu.calculate_irms(model_trained, window=window)
                model_irms_mismatch = ssmu.calculate_irms(model_trained_mismatch, window=window)
                model_irms_unconstrained = ssmu.calculate_irms(model_trained_unconstrained, window=window)

                data_irms_test[np.eye(data_irms_test.shape[0], dtype=bool)] = np.nan
                data_irms_train[np.eye(data_irms_train.shape[0], dtype=bool)] = np.nan
                model_irms[np.eye(model_irms.shape[0], dtype=bool)] = np.nan
                model_irms_mismatch[np.eye(model_irms_mismatch.shape[0], dtype=bool)] = np.nan
                model_irms_unconstrained[np.eye(model_irms_unconstrained.shape[0], dtype=bool)] = np.nan

                model_irms_score = []
                model_irms_score_ci = []
                true_weights = model_trained.dynamics_weights.copy()
                true_weights = true_weights[model_trained.param_props['mask']['dynamics_weights']]

                mismatch_weights = model_trained_mismatch.dynamics_weights.copy()
                mismatch_weights = mismatch_weights[model_trained_mismatch.param_props['mask']['dynamics_weights']]

                c, ci = met.nan_corr(true_weights, mismatch_weights)
                mismatch_to_true_weights_corr.append(c)
                # mismatch_to_true_weights_corr.append(ci)

                # for each model, correlate it to the measured irms in the test set
                for mi, m in enumerate([model_irms, model_irms_unconstrained, model_irms_mismatch]):
                    model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = (
                        met.nan_corr(m, data_irms_test))
                    model_irms_score.append(model_irms_to_measured_irms_test)
                    model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

                    if mi == 0:
                        true_corr.append(model_irms_to_measured_irms_test)
                        true_corr_ci.append(model_irms_to_measured_irms_test_ci)
                    elif mi == 1:
                        uncon_corr.append(model_irms_to_measured_irms_test)
                        uncon_corr_ci.append(model_irms_to_measured_irms_test_ci)
                    elif mi == 2:
                        mismatch_corr.append(model_irms_to_measured_irms_test)
                        mismatch_corr_ci.append(model_irms_to_measured_irms_test_ci)

    if cpu_id == 0:
        save_path = '/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/nature_review_appeal/'
        save_dict = {'true_corr': true_corr, 'uncon_corr': uncon_corr, 'mismatch_corr': mismatch_corr, 'mis_to_true_weights_corr': mismatch_to_true_weights_corr}

        save_file = open(save_path + 'smoothed_mismatch_data.pkl', 'wb')
        pickle.dump(save_dict, save_file)
        save_file.close()
    a=1


def fit_hessian(param_name, save_folder):
    # set up the option to parallelize the model fitting over CPUs
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    cpu_id = comm.Get_rank()

    run_params = lu.get_run_params(param_name=param_name)

    # cpu_id 0 is the parent node which will send out the data to the children nodes
    if cpu_id == 0:
        dtype = torch.float32
        device = 'cpu'

        data_folder = Path(run_params['data_folder'])

        model_file = open(data_folder / 'models' / 'model_trained.pkl', 'rb')
        model = pickle.load(model_file)
        model_file.close()

        # load posterior
        post_file = open(data_folder / 'posterior_train.pkl', 'rb')
        posterior_dict = pickle.load(post_file)
        post_file.close()

        # load data
        data_train_file = open(data_folder / 'data_train.pkl', 'rb')
        data_train = pickle.load(data_train_file)
        data_train_file.close()

        hess_dict = {
            'emissions': [torch.tensor(i, dtype=dtype, device=device) for i in data_train['emissions']],
            'inputs': [torch.tensor(i, dtype=dtype, device=device) for i in data_train['inputs']],
            'emissions_offset': [torch.tensor(i, dtype=dtype, device=device) for i in
                                 posterior_dict['emissions_offset']],
            'init_mean': [torch.tensor(i, dtype=dtype, device=device) for i in posterior_dict['init_mean']],
            'init_cov': [torch.tensor(i, dtype=dtype, device=device) for i in posterior_dict['init_cov']],
        }

        model.dynamics_weights = torch.tensor(model.dynamics_weights, dtype=dtype, device=device, requires_grad=True)
        model.dynamics_input_weights = torch.tensor(model.dynamics_input_weights, dtype=dtype, device=device)
        model.dynamics_cov = torch.tensor(model.dynamics_cov, dtype=dtype, device=device)
        model.emissions_weights = torch.tensor(model.emissions_weights, dtype=dtype, device=device)
        model.emissions_input_weights = torch.tensor(model.emissions_input_weights, dtype=dtype, device=device)
        model.emissions_cov = torch.tensor(model.emissions_cov, dtype=dtype, device=device)

        # lgssmu.approximate_hessian_diagonal(model, hess_dict)
        lgssmu.calc_hessian_torch(model, hess_dict)
