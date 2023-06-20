import numpy as np
import torch
import loading_utilities as lu
from matplotlib import pyplot as plt
import matplotlib as mpl
from ssm_classes import Lgssm
from mpi4py import MPI
import inference_utilities as iu

colormap = mpl.colormaps['coolwarm']

run_params = lu.get_run_params(param_name='params')
device = run_params["device"]
dtype = getattr(torch, run_params["dtype"])

# load in the data for the model and do any preprocessing here
emissions, inputs, cell_ids = \
    lu.load_and_align_data(run_params['data_path'],
                           bad_data_sets=run_params['bad_data_sets'],
                           start_index=run_params['start_index'],
                           force_preprocess=run_params['force_preprocess'],
                           correct_photobleach=run_params['correct_photobleach'],
                           interpolate_nans=run_params['interpolate_nans'])

num_neurons = emissions[0].shape[1]
num_data_sets = len(emissions)

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

# randomize the parameters (defaults are nonrandom)
model_true.randomize_weights()

A = model_true.dynamics_weights.detach().numpy()

# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)
num_lags = 5

all_a_hat = np.empty((num_neurons, num_neurons*num_lags, num_data_sets))
all_b_hat = np.empty((num_neurons, num_neurons*num_lags, num_data_sets))

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
    input_mask = torch.tile(input_mask.T, (num_lags, 1))

    input_mask = torch.cat((torch.ones(non_nan_emissions.shape[1]*num_lags, non_nan_emissions.shape[1]), input_mask), dim=0)



    # a_hat is a col vector of each A_hat_p matrix for each lag p -> need to transpose each A_hat_p
    num_emission_neurons = len(non_nan_emissions[0, :])
    num_input_neurons = len(curr_inputs[0, :])

    # ab_hat = np.linalg.lstsq(y_history, y_target, rcond=None)[0]
    # instead do masking from utils to get rid of the 0 entries and get proper fitting
    torch_y_history = torch.from_numpy(y_history)
    torch_y_target = torch.from_numpy(y_target)
    ab_hat = iu.solve_masked(torch_y_history, torch_y_target, input_mask)

    ab_hat = ab_hat.detach().numpy()

    a_hat = ab_hat[:num_lags * num_emission_neurons, :].T
    b_hat = ab_hat[num_lags * num_emission_neurons:, :].T

    y_hat = y_history @ ab_hat
    # print(a_hat)
    # print(y_hat)
    mse = np.mean((y_target - y_hat) ** 2)
    print(mse)

    # add NaNs back in for plotting and to compare across datasets
    temp = np.zeros((num_neurons, num_neurons * num_lags))
    temp[:, :] = np.nan
    i_count = 0
    j_count = 0
    for p in range(num_lags):
        for i in range(num_neurons):
            for j in range(num_neurons):
                if ~nans[i] and ~nans[j] and i_count < num_emission_neurons:
                    temp[i, p * num_neurons + j] = a_hat[i_count, p * num_emission_neurons + j_count]
                    j_count = j_count + 1
                    if j_count == num_emission_neurons:
                        i_count = i_count + 1
                    # set diagonal elements = 0 for plotting
                    if i == j:
                        temp[i, p * num_neurons + j] = 0.0
            j_count = 0
        i_count = 0
    a_hat_full = temp

    temp = np.zeros((num_neurons, num_neurons*num_lags))
    temp[:, :] = np.nan
    i_count = 0
    j_count = 0
    for p in range(num_lags):
        for i in range(num_neurons):
            for j in range(num_neurons):
                if has_stims[i] and has_stims[j] and i_count < num_input_neurons:
                    temp[i, p * num_neurons + j] = b_hat[i_count, p * num_input_neurons + j_count]
                    j_count = j_count + 1
                    if j_count == num_input_neurons:
                        i_count = i_count + 1
                    # set diagonal elements = 0 for plotting
                    if i == j:
                        temp[i, p * num_neurons + j] = 0.0
            j_count = 0
        i_count = 0
    b_hat_full = temp

    # # set diagonal elements = 0 for plotting
    # np.fill_diagonal(a_hat_full, 0.0)
    # np.fill_diagonal(b_hat_full, 0.0)

    all_a_hat[:, :, d] = a_hat_full
    all_b_hat[:, :, d] = b_hat_full

    # # fill in nans across first dimension
    # temp_nan = np.zeros((num_emission_neurons, num_neurons))
    # temp_nan[:] = np.nan
    # temp_nan[:, ~nans] = a_hat[:, :num_emission_neurons]
    # # fill in nans across second dimension
    # a_hat_full = np.zeros((num_neurons, num_neurons))
    # a_hat_full[:] = np.nan
    # a_hat_full[~nans, :] = temp_nan

    # temp_nan = np.zeros((num_input_neurons, num_neurons))
    # temp_nan[:] = np.nan
    # temp_nan[:, has_stims] = b_hat[:, :num_input_neurons]
    # # fill in nans across second dimension
    # b_hat_full = np.zeros((num_neurons, num_neurons))
    # b_hat_full[:] = np.nan
    # b_hat_full[has_stims, :] = temp_nan

    fig, axs = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: a_hat' % {"dataset": d, "lags": num_lags})
    a_hat_pos = plt.imshow(a_hat_full, aspect='auto', interpolation='nearest', cmap=colormap)
    color_limits = np.nanmax(np.abs(a_hat_full))
    plt.clim((-color_limits, color_limits))
    plt.colorbar(a_hat_pos)
    # plt.show()
    string = run_params['fig_path'] + 'ahat%i.png' % d
    plt.savefig(string)

    fig2, axs2 = plt.subplots(nrows=1, ncols=1)
    plt.title('dataset %(dataset)i GC for %(lags)i lags: b_hat' % {"dataset": d, "lags": num_lags})
    b_hat_pos = plt.imshow(b_hat_full, aspect='auto', interpolation='nearest', cmap=colormap)
    color_limits = np.nanmax(np.abs(b_hat_full))
    print("max bhat " + color_limits)
    plt.clim((-color_limits, color_limits))
    plt.colorbar(b_ha t_pos)
    # plt.show()
    string = run_params['fig_path'] + 'bhat%i.png' % d
    plt.savefig(string)

color_limits = np.nanmax(np.abs(A))
A_pos = plt.imshow(A, interpolation='nearest', cmap=colormap)
plt.clim((-color_limits, color_limits))
plt.colorbar(A_pos)
plt.show()

# create averaged a_hat and b_hat matrices over all non-NaN values over all datasets
# save all a_hat and b_hat full mtxes first as 3d array, then nanmean over each element along 3rd axis
a_hat_avg = np.nanmean(all_a_hat, axis=2)
b_hat_avg = np.nanmean(all_b_hat, axis=2)

fig3, axs3 = plt.subplots(nrows=1, ncols=1)
plt.title('averaged a_hat over all datasets')
avg_a_hat_pos = plt.imshow(a_hat_avg, aspect='auto', interpolation='nearest', cmap=colormap)
color_limits = np.nanmax(np.abs(a_hat_avg))
plt.clim((-color_limits, color_limits))
plt.colorbar(avg_a_hat_pos)
# plt.show()
string = run_params['fig_path'] + 'avg_a_hat.png'
plt.savefig(string)

fig4, axs4 = plt.subplots(nrows=1, ncols=1)
plt.title('averaged b_hat over all datasets')
avg_b_hat_pos = plt.imshow(b_hat_avg, aspect='auto', interpolation='nearest', cmap=colormap)
color_limits = np.nanmax(np.abs(b_hat_avg))
plt.clim((-color_limits, color_limits))
plt.colorbar(avg_b_hat_pos)
# plt.show()
string = run_params['fig_path'] + 'avg_b_hat.png'
plt.savefig(string)