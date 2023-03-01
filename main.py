import numpy as np
import torch
from pathlib import Path
import preprocessing as pp
import inference as infer
from matplotlib import pyplot as plt
import time
from ssm_classes import LgssmSimple


from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
# from torch.utils.tensorboard import SummaryWriter

# the goal of this function is to take the pairwise stimulation and response data from
# https://arxiv.org/abs/2208.04790
# this data is a collection of calcium recordings of ~100 neurons over ~5-15 minutes where individual neurons are
# randomly targets and stimulated optogenetically
# We want to fit a linear dynamical system to the data in order to infer the connection weights between neurons
# The model is of the form
# x_t = A @ x_(t-1) + V @ u_t + b + w_t
# y_t = x_t + d + v_t

# the mapping of x_t to y_t is the identity
# V is diagonal
# w_t is a gaussian with 0 mean, and potentially diagonal covariance
# optionally would like to add a dependence on another time point in the past, i.e. H @ x_(t-2)
# fitting needs to handle missing data in y

# load all the recordings
fun_atlas_path = Path('/home/mcreamer/Documents/data_sets/fun_con')
recordings, labels, q, q_labels, stim_cell_ids, stim_volume_inds = pp.load_data(fun_atlas_path)

# We want to take our data set of recordings and pick a subset of those recordings and neurons in order
# to maximize the number of recordings * the number of neurons that appear in those recordings
# this variable allows a neuron to only appear in this fraction of recordings to still be included
# (rather than enforcing it is in 100% of the recording subset)
frac_neuron_coverage = 0.0

# for a neuron to be included in the dataset, it must be in at least this fraction of the recordings
# this is not a principled threshold, (you could have 90 bad recordings and 10 good ones), but it's a quick heuristic
# to reduce dataset size
minimum_frac_measured = 0.0

# number of iteration through the full data set to run during gradient descent
num_grad_steps = 100
batch_size = 10
num_splits = 2
learning_rate = 1e-2

# the calcium dynamics always look strange at the start of a recording, possibly due to the laser being turned on
# cut out the first ~15 seconds to let the system equilibrate
index_start = 25

# remove datasets with bad entries, or just limit how much data to process
bad_datasets = np.sort([0, 4, 6, 10, 11, 15, 17, 24, 35, 37, 45])[::-1]
# num_datasets = 5
# bad_datasets = np.sort(np.arange(num_datasets + 2, len(recordings)))[::-1]
# bad_datasets = np.append(bad_datasets, 4)
# bad_datasets = np.append(bad_datasets, 0)
# bad_datasets = []
stim_mat = None

verbose = True
random_seed = 0
device = 'cpu'
# dtype = torch.float32
dtype = torch.float64

for bd in bad_datasets:
    recordings.pop(bd)
    labels.pop(bd)
    stim_volume_inds.pop(bd)
    stim_cell_ids.pop(bd)

# choose a subset of the data sets to maximize the number of recordings * the number of neurons included
cell_ids, calcium_data, best_runs, stim_mat_full = \
    pp.get_combined_dataset(recordings, labels, stim_cell_ids, stim_volume_inds,
                            frac_neuron_coverage=frac_neuron_coverage,
                            minimum_freq=minimum_frac_measured)

# split the data in half for lower memory usage
data_final = []
stim_mat = []
for cd, sm in zip(calcium_data, stim_mat_full):
    half_data_ind = int(np.ceil(cd.shape[0]/2))
    data_final.append(cd[:half_data_ind, :])
    data_final.append(cd[half_data_ind:, :])
    stim_mat.append(sm[:half_data_ind, :])
    stim_mat.append(sm[half_data_ind:, :])

num_data = len(data_final)
num_epochs = int(np.ceil(num_grad_steps * batch_size / num_data))

# limit the data size
num_neurons = 200
for ri in range(num_data):
    cell_ids = cell_ids[:num_neurons]
    data_final[ri] = data_final[ri][index_start:, :num_neurons]
    data_final[ri] = data_final[ri] - np.mean(data_final[ri], axis=0, keepdims=True)
    stim_mat[ri] = stim_mat[ri][index_start:, :num_neurons]

# initialize and train model
latent_dim = len(cell_ids)
lgssm_model = LgssmSimple(latent_dim, dtype=dtype, device=device, random_seed=random_seed, verbose=verbose)
loss_out = lgssm_model.fit_batch_sgd(data_final, stim_mat, learning_rate=learning_rate,
                                     num_steps=num_grad_steps, batch_size=batch_size,
                                     num_splits=num_splits)

# Plots
# Plot the loss
plt.figure()
plt.plot(loss_out)
plt.xlabel('iterations')
plt.ylabel('negative log likelihood')
plt.tight_layout()

# Plot the dynamics weights
plt.figure()
colorbar_shrink = 0.4
plt.subplot(1, 2, 1)
plt.imshow(lgssm_model.dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
plt.title('dynamics weights')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar(shrink=colorbar_shrink)

plt.subplot(1, 2, 2)
plt.imshow(lgssm_model.dynamics_weights.detach().cpu().numpy() - np.identity(latent_dim), interpolation='Nearest')
plt.title('dynamics weights - I')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar(shrink=colorbar_shrink)
plt.tight_layout()

# Plot the input weights
plt.figure()
plt.plot(np.exp(lgssm_model.inputs_weights_log_init))
plt.plot(np.exp(lgssm_model.inputs_weights_log.detach().cpu().numpy()))
plt.legend(['init', 'final'])
plt.xlabel('neurons')
plt.ylabel('input weights')

# plot the covariances
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.exp(lgssm_model.dynamics_cov_log_init))
plt.plot(np.exp(lgssm_model.dynamics_cov_log.detach().cpu().numpy()))
plt.legend(['init', 'final'])
plt.xlabel('neurons')
plt.ylabel('dynamics noise cov')

plt.subplot(1, 2, 2)
plt.plot(np.exp(lgssm_model.emissions_cov_log_init))
plt.plot(np.exp(lgssm_model.emissions_cov_log.detach().cpu().numpy()))
plt.legend(['init', 'final'])
plt.xlabel('neurons')
plt.ylabel('emissions noise cov')
plt.tight_layout()

plt.show()
