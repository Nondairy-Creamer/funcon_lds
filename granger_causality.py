import numpy as np
import torch
import loading_utilities as lu
from matplotlib import pyplot as plt
from ssm_classes import Lgssm
from mpi4py import MPI
import inference_utilities as iu

run_params = lu.get_run_params(param_name='params')
device = run_params["device"]
dtype = getattr(torch, run_params["dtype"])

emissions, inputs, cell_ids = \
        lu.get_model_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                          bad_data_sets=run_params['bad_data_sets'],
                          frac_neuron_coverage=run_params['frac_neuron_coverage'],
                          minimum_frac_measured=run_params['minimum_frac_measured'],
                          start_index=run_params['start_index'])

# If you are considering multiple lags in the past, lag the inputs
num_neurons = emissions[0].shape[1]

# get the input dimension after removing the neurons that were never stimulated
input_dim = inputs[0].shape[1]

# initialize the model and set model weights
model_true = Lgssm(num_neurons, num_neurons, input_dim,
                   dynamics_lags=run_params['dynamics_lags'],
                   dynamics_input_lags=run_params['dynamics_input_lags'],
                   dtype=dtype, device=device, verbose=run_params['verbose'],
                   param_props=run_params['param_props'])

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
num_lags = 1
# nan_list = []

# testing:
num_data_sets = 1

for d in range(num_data_sets):
    # num_time, neurons varies depending on the dataset
    num_time, num_neurons = emissions[d].shape
    curr_inputs = inputs[d]

    ############ this needs to be done per data set. for the emissions data you'll have
    # the vector of nan'd intries
    # for inputs you'll have the vector of neurons that saw a stimulation
    # the inputs dimension will always be smaller than the emission dimension

    # get rid of any inputs that never receive stimulation
    has_stims = np.any(curr_inputs, axis=0)

    curr_inputs = curr_inputs[:, has_stims]

    # need to delete columns with NaN neurons, but make a list of these indices to add them in as 0s in the end
    nans = np.any(np.isnan(emissions[d]), axis=0)
    nans_mask = nans[:, None] | nans[:, None].T
    # not_nans = np.where(~np.isnan(emissions[d][0, :]))[0]
    # nan_list.append(nans)
    non_nan_emissions = emissions[d][:, ~nans]



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
        y_history = np.concatenate((y_history, non_nan_emissions[p:p - num_lags, :]), axis=1)

    # add to y_history the inputs to get input weights (u_t)
    for p in reversed(range(num_lags)):
        if (p - num_lags + 1) != 0:
            y_history = np.concatenate((y_history, curr_inputs[(p + 1):(p - num_lags + 1), :]), axis=1)
        else:
            y_history = np.concatenate((y_history, curr_inputs[(p + 1):, :]), axis=1)

    # A_hat = np.linalg.solve(y_history, y_target).T
    # -> linalg.solve doesn't work because y_history is not square --> use least squares instead
    # q, r = np.linalg.qr(y_history)
    # p = np.dot(q.T, y_target)
    # a_hat = np.dot(np.linalg.inv(r), p)
    # create a mask for the dynamics_input_weights. This allows us to fit dynamics weights that are diagonal
    input_mask = torch.eye(non_nan_emissions.shape[1], has_stims.shape[0], dtype=dtype, device=device)
    input_mask = input_mask[:, has_stims]

    input_mask = torch.cat((torch.ones(y_history.shape[0], non_nan_emissions.shape[1]), input_mask.T), dim=0)



    # a_hat is a col vector of each A_hat_p matrix for each lag p -> need to transpose each A_hat_p
    num_emission_neurons = len(non_nan_emissions[0, :])
    num_input_neurons = len(curr_inputs[0, :])

    ab_hat = np.linalg.lstsq(y_history, y_target, rcond=None)[0]
    # instead do masking from utils to get rid of the 0 entries and get proper fitting
    # torch_y_history = torch.from_numpy(y_history)
    # torch_y_target = torch.from_numpy(y_target)
    # ab_hat = iu.solve_masked(torch_y_history, torch_y_target, input_mask)

    a_hat = ab_hat[:num_lags * num_emission_neurons, :].T
    b_hat = ab_hat[num_lags * num_input_neurons:, :].T

    y_hat = y_history @ ab_hat
    # print(a_hat)
    # print(y_hat)
    mse = np.mean((y_target - y_hat) ** 2)
    print(mse)

    # add NaNs back in for plotting and to compare across datasets
    # temp = np.zeros((num_neurons * num_lags, num_neurons))
    # temp[:, :] = np.nan
    # i_count = 0
    # j_count = 0
    # for p in range(num_lags):
    #     for i in range(num_neurons):
    #         for j in range(num_neurons):
    #             if ~nans_mask[i, j]:
    #                 temp[p * num_neurons + i, j] = a_hat[i_count, j_count]
    #                 j_count = j_count + 1
    #                 if j_count == num_emission_neurons:
    #                     i_count = i_count + 1
    #         j_count = 0
    # a_hat = temp
    #
    temp = np.zeros((num_neurons, num_neurons*num_lags))
    temp[:, :] = np.nan
    i_count = 0
    j_count = 0
    for p in range(num_lags):
        for i in range(num_neurons):
            for j in range(num_neurons):
                if has_stims[i] and has_stims[j]:
                    temp[i, p * num_input_neurons + j] = b_hat[i_count, j_count]
                    j_count = j_count + 1
                    if j_count == num_input_neurons:
                        i_count = i_count + 1
            j_count = 0
    b_hat_full = temp

    # fill in nans across first dimension
    temp_nan = np.zeros((num_emission_neurons, num_neurons))
    temp_nan[:] = np.nan
    temp_nan[:, ~nans] = a_hat[:, :num_emission_neurons]
    # fill in nans across second dimension
    a_hat_full = np.zeros((num_neurons, num_neurons))
    a_hat_full[:] = np.nan
    a_hat_full[~nans, :] = temp_nan

    # temp_nan = np.zeros((num_input_neurons, num_neurons))
    # temp_nan[:] = np.nan
    # temp_nan[:, has_stims] = b_hat[:, :num_input_neurons]
    # # fill in nans across second dimension
    # b_hat_full = np.zeros((num_neurons, num_neurons))
    # b_hat_full[:] = np.nan
    # b_hat_full[has_stims, :] = temp_nan

    fig, axs = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: a_hat' % {"dataset": d, "lags": num_lags})
    a_hat_pos = plt.imshow(a_hat_full, aspect='auto', interpolation='nearest')
    plt.colorbar(a_hat_pos)
    plt.show()
    str = run_params['fig_path'] + 'ahat%i.png' % d
    plt.savefig(str)

    fig2, axs2 = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: b_hat' % {"dataset": d, "lags": num_lags})
    b_hat_pos = plt.imshow(b_hat_full, aspect='auto', interpolation='nearest')
    plt.colorbar(b_hat_pos)
    plt.show()
    str = run_params['fig_path'] + 'bhat%i.png' % d
    plt.savefig(str)

A_pos = plt.imshow(A, interpolation='nearest')
plt.colorbar(A_pos)
plt.show()

