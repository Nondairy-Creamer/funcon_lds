import numpy as np
import pickle
import torch
import matplotlib
import matplotlib.pyplot as plt
from ssm_classes import Lgssm
import preprocessing as pp
import utilities as util

params = pp.get_params(param_name='params_synth')
num_data_sets = params['num_data_sets']
device = params["device"]
dtype = getattr(torch, params["dtype"])
rng = np.random.default_rng(params['random_seed'])

model_synth_true = Lgssm(params['dynamics_dim'], params['emissions_dim'], params['input_dim'],
                       dtype=dtype, device=device, param_props=params['param_props'],
                       dynamics_lags=params['dynamics_lags'], dynamics_input_lags=params['dynamics_input_lags'])
model_synth_true.randomize_weights(rng=rng)
model_synth_true.emissions_weights_init = np.eye(model_synth_true.emissions_dim, model_synth_true.dynamics_dim_full)
model_synth_true.emissions_input_weights_init = np.zeros((model_synth_true.emissions_dim, model_synth_true.input_dim_full))
model_synth_true.set_to_init()

# sample from the randomized model
data_dict = \
    model_synth_true.sample(num_time=params['num_time'],
                      num_data_sets=params['num_data_sets'],
                      scattered_nan_freq=params['scattered_nan_freq'],
                      lost_emission_freq=params['lost_emission_freq'],
                      input_time_scale=params['input_time_scale'],
                      rng=rng)

emissions = data_dict['emissions']
inputs = data_dict['inputs']
latents_true = data_dict['latents']
init_mean_true = data_dict['init_mean']
init_cov_true = data_dict['init_cov']

num_time = np.zeros(len(emissions))
for i in range(len(emissions)):
    num_time[i] = len(emissions[i])

A = model_synth_true.dynamics_weights.detach().numpy()

# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)
num_lags = 1
nan_list = []

for d in range(num_data_sets):
    # num_time, neurons varies depending on the dataset
    num_time, num_neurons = emissions[d].shape

    # need to delete columns with NaN neurons, but make a list of these indices to add them in as 0s in the end
    nans = np.where(np.isnan(emissions[d][0, :]))
    nan_list.append(nans)
    good_emissions = emissions[d][:, ~np.isnan(emissions[d][0, :])]
    good_inputs = inputs[d][:, ~np.isnan(emissions[d][0, :])]

    # y_target is the time series we are trying to predict from A_hat @ y_history
    # y_target should start at t=0+num_lags
    # y_target = np.zeros((num_time - num_lags, num_neurons))
    # y_target is the lagged time series, should start at t=0+num_lags-1
    # we will concatenate each of the columns of the y_history matrix where each column corresponds to a lagged time series
    y_history = np.zeros((num_time - num_lags, 0))

    # note this goes from time num_lags to T
    y_target = good_emissions[num_lags:, :]

    # build lagged y_history from emissions (x_t)
    for p in reversed(range(num_lags)):
        if p - num_lags:
            y_history = np.concatenate((y_history, good_emissions[p:p-num_lags, :]), axis=1)
        else:
            y_history = np.concatenate((y_history, good_emissions[p:, :]), axis=1)

    # add to y_history the inputs to get input weights (u_t)
    for p in reversed(range(num_lags)):
        if p - num_lags:
            y_history = np.concatenate((y_history, good_inputs[p:p - num_lags, :]), axis=1)
        else:
            y_history = np.concatenate((y_history, good_inputs[p:p - num_lags, :]), axis=1)

    # A_hat = np.linalg.solve(y_history, y_target).T
    # -> linalg.solve doesn't work because y_history is not square --> use least squares instead
    # q, r = np.linalg.qr(y_history)
    # p = np.dot(q.T, y_target)
    # a_hat = np.dot(np.linalg.inv(r), p)

    # a_hat is a col vector of each A_hat_p matrix for each lag p -> need to transpose each A_hat_p
    num_good_neurons = len(good_emissions[0, :])

    ab_hat = np.linalg.lstsq(y_history, y_target, rcond=None)[0]

    # instead do masking from utils to get rid of the 0 entries and get proper fitting
    # mask =
    # ab_hat = util.solve_masked(y_history, y_target, mask)

    a_hat = ab_hat[:num_lags*num_good_neurons, :]
    b_hat = ab_hat[num_lags*num_good_neurons:, :]
    for p in range(num_lags):
        a_hat[p*num_good_neurons:p*num_good_neurons+num_good_neurons, :] = \
            a_hat[p*num_good_neurons:p*num_good_neurons+num_good_neurons, :].T
        b_hat[p * num_good_neurons:p * num_good_neurons + num_good_neurons, :] = \
            b_hat[p * num_good_neurons:p * num_good_neurons + num_good_neurons, :].T

    y_hat = y_history @ ab_hat
    # print(a_hat)
    # print(y_hat)
    mse = np.mean((y_target - y_hat) ** 2)
    print(mse)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    plt.title('SYNTH dataset %(dataset)i GC for %(lags)i lags: a_hat' % {"dataset": d, "lags": num_lags})
    a_hat_pos = plt.imshow(a_hat, aspect='auto', interpolation='nearest')
    plt.colorbar(a_hat_pos)
    # plt.show()
    str = params['fig_path'] + 'SYNTHahat%i.png' % d
    plt.savefig(str)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    plt.title('SYNTH dataset %(dataset)i GC for %(lags)i lags: b_hat' % {"dataset": d, "lags": num_lags})
    b_hat_pos = plt.imshow(b_hat, aspect='auto', interpolation='nearest')
    plt.colorbar(b_hat_pos)
    # plt.show()
    str = params['fig_path'] + 'SYNTHbhat%i.png' % d
    plt.savefig(str)

A_pos = plt.imshow(A, interpolation='nearest')
plt.colorbar(A_pos)
plt.show()