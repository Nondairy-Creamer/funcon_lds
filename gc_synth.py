import torch
from ssm_classes import Lgssm
import loading_utilities as lu
import numpy as np
import time
from mpi4py import MPI
import inference_utilities as iu
from matplotlib import pyplot as plt

run_params = lu.get_run_params(param_name='params_synth')
num_data_sets = run_params['num_data_sets']
device = run_params["device"]
dtype = getattr(torch, run_params["dtype"])
rng = np.random.default_rng(run_params['random_seed'])

model_synth_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                         dtype=dtype, device=device, param_props=run_params['param_props'],
                         dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'])
model_synth_true.randomize_weights(rng=rng)
model_synth_true.emissions_weights_init = np.eye(model_synth_true.emissions_dim, model_synth_true.dynamics_dim_full)
model_synth_true.emissions_input_weights_init = np.zeros((model_synth_true.emissions_dim, model_synth_true.input_dim_full))
model_synth_true.set_to_init()

# sample from the randomized model
data_dict = \
    model_synth_true.sample(num_time=run_params['num_time'],
                            num_data_sets=run_params['num_data_sets'],
                            scattered_nan_freq=run_params['scattered_nan_freq'],
                            lost_emission_freq=run_params['lost_emission_freq'],
                            input_time_scale=run_params['input_time_scale'],
                            rng=rng)

emissions = data_dict['emissions']
inputs = data_dict['inputs']
latents_true = data_dict['latents']
init_mean_true = data_dict['init_mean']
init_cov_true = data_dict['init_cov']

num_time = np.zeros(len(emissions))
for i in range(len(emissions)):
    num_time[i] = len(emissions[i])

model_params = model_synth_true.get_params()
A_true = model_params['trained']['dynamics_weights']
B_true = model_params['trained']['dynamics_input_weights']

# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)
num_lags = 1
# nan_list = []

for d in range(num_data_sets):
    # num_time, neurons varies depending on the dataset
    num_time, num_neurons = emissions[d].shape

    # create a mask for the dynamics_input_weights. This allows us to fit dynamics weights that are diagonal
    input_mask = torch.eye(num_neurons, dtype=dtype, device=device)
    # get rid of any inputs that never receive stimulation
    has_stims = np.any(np.concatenate(inputs, axis=0), axis=0)
    inputs = [i[:, has_stims] for i in inputs]
    input_mask = input_mask[:, has_stims]

    # need to concatenate masks
    mask = input_mask

    # need to delete columns with NaN neurons, but make a list of these indices to add them in as 0s in the end
    nans = np.isnan(emissions[d][:num_neurons, :])
    nans_mask = nans | nans.T
    non_nan_emissions = emissions[d][:, ~np.isnan(emissions[d][0, :])]
    curr_inputs = inputs[d]

    # y_target is the time series we are trying to predict from A_hat @ y_history
    # y_target should start at t=0+num_lags
    # y_target = np.zeros((num_time - num_lags, num_neurons))
    # y_target is the lagged time series, should start at t=0+num_lags-1
    # we will concatenate each of the columns of the y_history matrix where each column corresponds to a lagged time series
    y_history = np.zeros((num_time - num_lags, 0))

    # note this goes from time num_lags to T
    y_target = non_nan_emissions[num_lags:, :]

    # build lagged y_history from emissions (x_(t-1))
    for p in reversed(range(num_lags)):
        # if p - num_lags:
        y_history = np.concatenate((y_history, non_nan_emissions[p:p - num_lags, :]), axis=1)
            # y_history = np.concatenate((y_history, curr_inputs[p:p - num_lags, :]), axis=1)
        # else:
        #     y_history = np.concatenate((y_history, non_nan_emissions[p:, :]), axis=1)
            # y_history = np.concatenate((y_history, curr_inputs[p:p - num_lags, :]), axis=1)

    # add to y_history the inputs to get input weights (u_t)
    for p in reversed(range(num_lags)):
        if (p - num_lags + 1) != 0:
            y_history = np.concatenate((y_history, curr_inputs[(p + 1):(p - num_lags + 1), :]), axis=1)
            mask = np.concatenate((mask, input_mask), axis=1)
        else:
            y_history = np.concatenate((y_history, curr_inputs[(p + 1):, :]), axis=1)
        # mask = np.concatenate((mask, input_mask), axis=1)

    # A_hat = np.linalg.solve(y_history, y_target).T
    # -> linalg.solve doesn't work because y_history is not square --> use least squares instead
    # q, r = np.linalg.qr(y_history)
    # p = np.dot(q.T, y_target)
    # a_hat = np.dot(np.linalg.inv(r), p)

    # a_hat is a col vector of each A_hat_p matrix for each lag p -> need to transpose each A_hat_p
    num_emission_neurons = len(non_nan_emissions[0, :])
    num_input_neurons = len(curr_inputs[0, :])

    ab_hat = np.linalg.lstsq(y_history, y_target, rcond=None)[0]
    # torch_y_history = torch.from_numpy(y_history)
    # torch_y_target = torch.from_numpy(y_target)
    # ab_hat = iu.solve_masked(torch_y_history, torch_y_target, input_mask)

    # instead do masking from utils to get rid of the 0 entries and get proper fitting
    # mask =
    # ab_hat = iu.solve_masked(y_history, y_target, mask)

    a_hat = ab_hat[:num_lags * num_emission_neurons, :]
    b_hat = ab_hat[num_lags * num_input_neurons:, :]
    for p in range(num_lags):
        a_hat[p * num_emission_neurons:p * num_emission_neurons + num_emission_neurons, :] = \
            a_hat[p * num_emission_neurons:p * num_emission_neurons + num_emission_neurons, :].T
        b_hat[p * num_input_neurons:p * num_input_neurons + num_input_neurons, :] = \
            b_hat[p * num_input_neurons:p * num_input_neurons + num_input_neurons, :].T

    y_hat = y_history @ ab_hat
    # print(a_hat)
    # print(y_hat)
    mse = np.mean((y_target - y_hat) ** 2)
    print(mse)

    # add NaNs back in for plotting and to compare across datasets
    temp = np.zeros((num_neurons, num_neurons))
    temp[:, :] = np.nan
    i_count = 0
    j_count = 0
    for i in range(num_neurons):
        for j in range(num_neurons):
            if ~nans_mask[i, j]:
                temp[i, j] = a_hat[i_count, j_count]
                j_count = j_count + 1
                if j_count == num_emission_neurons:
                    i_count = i_count + 1
        j_count = 0
    a_hat = temp

    fig, axs = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: a_hat' % {"dataset": d, "lags": num_lags})
    a_hat_pos = plt.imshow(a_hat, aspect='auto', interpolation='nearest')
    plt.colorbar(a_hat_pos)
    plt.show()
    # str = params['fig_path'] + 'ahat%i.png' % d
    # plt.savefig(str)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: b_hat' % {"dataset": d, "lags": num_lags})
    b_hat_pos = plt.imshow(b_hat, aspect='auto', interpolation='nearest')
    plt.colorbar(b_hat_pos)
    plt.show()
    # str = params['fig_path'] + 'bhat%i.png' % d
    # plt.savefig(str)

A_pos = plt.imshow(A_true, interpolation='nearest')
plt.title('true A')
plt.colorbar(A_pos)
plt.show()

B_pos = plt.imshow(B_true, interpolation='nearest')
plt.title('true dynamics_input_weights (B)')
plt.colorbar(B_pos)
plt.show()