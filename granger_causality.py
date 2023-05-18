import numpy as np
import pickle
import torch
import matplotlib
import matplotlib.pyplot as plt
from ssm_classes import LgssmSimple
import preprocessing as pp


params = pp.get_params(param_name='params')
emissions_unaligned, cell_ids_unaligned, q, q_labels, stim_cell_ids, inputs_unaligned = \
    pp.load_data(params['data_path'])

# remove recordings that are noisy
data_sets_to_remove = np.sort(params['bad_data_sets'])[::-1]
for bd in data_sets_to_remove:
    emissions_unaligned.pop(bd)
    cell_ids_unaligned.pop(bd)
    inputs_unaligned.pop(bd)
    stim_cell_ids.pop(bd)

cell_ids, emissions, best_runs, inputs = \
    pp.get_combined_dataset(emissions_unaligned, cell_ids_unaligned, stim_cell_ids, inputs_unaligned,
                            frac_neuron_coverage=params['frac_neuron_coverage'],
                            minimum_freq=params['minimum_frac_measured'])

num_data_sets = len(emissions)

# remove the beginning of the recording which contains artifacts and mean subtract
for ri in range(num_data_sets):
    emissions[ri] = emissions[ri][params['index_start']:, :]
    emissions[ri] = emissions[ri] - np.mean(emissions[ri], axis=0, keepdims=True)
    inputs[ri] = inputs[ri][params['index_start']:, :]
latent_dim = len(cell_ids)

device = params["device"]
dtype = getattr(torch, params["dtype"])
model_true = LgssmSimple(latent_dim, dtype=dtype, device=device,
                            verbose=params['verbose'])

# randomize the parameters (defaults are nonrandom)
model_true.randomize_weights(random_seed=params["random_seed"])

# this below is only for the synthetic data, don't need for real datasets
# sample from the randomized model
# data_dict = model_true.sample(
#     num_time=params["num_time"],
#     # num_data_sets=params["num_data_sets"],
#     nan_freq=params["nan_freq"],
#     random_seed=params["random_seed"],
#

# num_time = np.zeros(len(emissions))
# for i in range(len(emissions)):
#     num_time[i] = len(emissions[i])

############### copy below this for multiple datasets
# init_mean_true = data_dict["init_mean"]
# init_cov_true = data_dict["init_cov"]
# latents = data_dict["latents"][0]

A = model_true.dynamics_weights.detach().numpy()

# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)
num_lags = 5

for d in range(num_data_sets):
    # num_time, neurons varies depending on the dataset
    num_time, num_neurons = emissions[d].shape
    # y_target is the time series we are trying to predict from A_hat @ y_history
    # y_target should start at t=0+num_lags
    # y_target = np.zeros((num_time - num_lags, num_neurons))
    # y_target is the lagged time series, should start at t=0+num_lags-1
    # we will concatenate each of the columns of the y_history matrix where each column corresponds to a lagged time series
    y_history = np.zeros((num_time - num_lags, 0))

    # note this goes from time num_lags to T
    y_target = emissions[d][num_lags:, :]

    for p in reversed(range(num_lags)):
        if p - num_lags:
            y_history = np.concatenate((y_history, emissions[p:p-num_lags, :]), axis=1)
        else:
            y_history = np.concatenate((y_history, emissions[p:, :]), axis=1)

    # A_hat = np.linalg.solve(y_history, y_target).T
    # -> linalg.solve doesn't work because y_history is not square --> use least squares instead
    # q, r = np.linalg.qr(y_history)
    # p = np.dot(q.T, y_target)
    # a_hat = np.dot(np.linalg.inv(r), p)

    # a_hat is a col vector of each A_hat_p matrix for each lag p -> need to transpose each A_hat_p
    a_hat = np.linalg.lstsq(y_history, y_target, rcond=None)[0]
    for p in range(num_lags):
        a_hat[p*num_neurons:p*num_neurons+num_neurons, :] = a_hat[p*num_neurons:p*num_neurons+num_neurons, :].T

    y_hat = y_history @ a_hat
    print(a_hat)
    print(y_hat)
    mse = np.mean((y_target - y_hat) ** 2)
    print(mse)

    fig, axs = plt.subplots(1, 2)
    A_pos = axs[0].imshow(A)
    a_hat_pos = axs[1].imshow(a_hat)
    fig.colorbar(A_pos, ax=axs[0])
    fig.colorbar(a_hat_pos, ax=axs[1])

    plt.show()

######## multiple datasets --> think about making function for parts that stay the same
# emissions is a d-long list of arrays, latents is a NxTxd array where d is num datasets
# emissions_list = data_dict["emissions"][:]
# num_time, num_neurons = emissions_list[0].shape
# emissions = np.zeros((num_time, num_neurons, num_data_sets))
# latents = data_dict["latents"][:]#
#
# # how to deal with multiple datasets? try averaging each neuron across datasets when fitting a_hat?
