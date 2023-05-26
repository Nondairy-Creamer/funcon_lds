import numpy as np
import torch
import loading_utilities as lu
from matplotlib import pyplot as plt
from ssm_classes import Lgssm
from mpi4py import MPI
import inference_utilities as iu

params = lu.get_run_params(param_name='params')
device = params["device"]
dtype = getattr(torch, params["dtype"])

emissions, inputs, cell_ids = \
        lu.get_model_data(params['data_path'], num_data_sets=params['num_data_sets'],
                          bad_data_sets=params['bad_data_sets'],
                          frac_neuron_coverage=params['frac_neuron_coverage'],
                          minimum_frac_measured=params['minimum_frac_measured'],
                          start_index=params['start_index'])

# If you are considering multiple lags in the past, lag the inputs
num_neurons = emissions[0].shape[1]

# create a mask for the dynamics_input_weights. This allows us to fit dynamics weights that are diagonal
input_mask = torch.eye(num_neurons, dtype=dtype, device=device)
# get rid of any inputs that never receive stimulation
has_stims = np.any(np.concatenate(inputs, axis=0), axis=0)
inputs = [i[:, has_stims] for i in inputs]
input_mask = input_mask[:, has_stims]
# set the model properties so the model fits with this mask
params['param_props']['mask']['dynamics_input_weights'] = input_mask
# get the input dimension after removing the neurons that were never stimulated
input_dim = inputs[0].shape[1]

# initialize the model and set model weights
model_true = Lgssm(num_neurons, num_neurons, input_dim,
                      dynamics_lags=params['dynamics_lags'],
                      dynamics_input_lags=params['dynamics_input_lags'],
                      dtype=dtype, device=device, verbose=params['verbose'],
                      param_props=params['param_props'])

model_true.emissions_weights = torch.eye(model_true.emissions_dim, model_true.dynamics_dim_full, device=device, dtype=dtype)
model_true.emissions_input_weights = torch.zeros((model_true.emissions_dim, model_true.input_dim_full), device=device, dtype=dtype)

num_data_sets = len(emissions)

# randomize the parameters (defaults are nonrandom)
model_true.randomize_weights()

A = model_true.dynamics_weights.detach().numpy()

# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)
num_lags = 2
# nan_list = []

# testing:
num_data_sets = 1

for d in range(num_data_sets):
    # num_time, neurons varies depending on the dataset
    num_time, num_neurons = emissions[d].shape

    # need to delete columns with NaN neurons, but make a list of these indices to add them in as 0s in the end
    nans = np.isnan(emissions[d][:num_neurons, :])
    nans_mask = nans | nans.T
    # not_nans = np.where(~np.isnan(emissions[d][0, :]))[0]
    # nan_list.append(nans)
    good_emissions = emissions[d][:, ~np.isnan(emissions[d][0, :])]
    curr_inputs = inputs[d]

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
            y_history = np.concatenate((y_history, curr_inputs[p:p - num_lags, :]), axis=1)
        else:
            y_history = np.concatenate((y_history, curr_inputs[p:p - num_lags, :]), axis=1)

    # A_hat = np.linalg.solve(y_history, y_target).T
    # -> linalg.solve doesn't work because y_history is not square --> use least squares instead
    # q, r = np.linalg.qr(y_history)
    # p = np.dot(q.T, y_target)
    # a_hat = np.dot(np.linalg.inv(r), p)

    # a_hat is a col vector of each A_hat_p matrix for each lag p -> need to transpose each A_hat_p
    num_emission_neurons = len(good_emissions[0, :])
    num_input_neurons = len(curr_inputs[0, :])

    ab_hat = np.linalg.lstsq(y_history, y_target, rcond=None)[0]
    # instead do masking from utils to get rid of the 0 entries and get proper fitting
    # torch_y_history = torch.from_numpy(y_history)
    # torch_y_target = torch.from_numpy(y_target)
    # ab_hat = iu.solve_masked(torch_y_history, torch_y_target, input_mask)

    a_hat = ab_hat[:num_lags*num_emission_neurons, :]
    b_hat = ab_hat[num_lags*num_input_neurons:, :]
    for p in range(num_lags):
        a_hat[p*num_emission_neurons:p*num_emission_neurons+num_emission_neurons, :] = \
            a_hat[p*num_emission_neurons:p*num_emission_neurons+num_emission_neurons, :].T
        # b_hat[p * num_input_neurons:p * num_input_neurons + num_input_neurons, :] = \
        #     b_hat[p * num_input_neurons:p * num_input_neurons + num_input_neurons, :].T

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

    # temp[not_nans, not_nans] = a_hat[:, :]
    # temp = np.where(nans_mask, np.nan, a_hat)

    a_hat = temp

    fig, axs = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: a_hat' % {"dataset": d, "lags": num_lags})
    a_hat_pos = plt.imshow(a_hat, aspect='auto', interpolation='nearest')
    plt.colorbar(a_hat_pos)
    # plt.show()
    str = params['fig_path'] + 'ahat%i.png' % d
    plt.savefig(str)

    fig2, axs2 = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: b_hat' % {"dataset": d, "lags": num_lags})
    b_hat_pos = plt.imshow(b_hat, aspect='auto', interpolation='nearest')
    plt.colorbar(b_hat_pos)
    # plt.show()
    str = params['fig_path'] + 'bhat%i.png' % d
    plt.savefig(str)

A_pos = plt.imshow(A, interpolation='nearest')
plt.colorbar(A_pos)
plt.show()

