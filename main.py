import numpy as np
import torch
from pathlib import Path
import preprocessing as pp
import inference as infer
from matplotlib import pyplot as plt
import time
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
num_grad_steps = 50
batch_size = 1

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

# initialize parameters, see lds_construction.pdf for model structure
model_dim = len(cell_ids)
init_std = 0.1
var_log_offset = 0
dynamics_max_eig = 0.8

# initialize the parameters, see PDF for name definitions
rng_torch = torch.Generator(device=device)
rng_torch.manual_seed(random_seed)

init_mean = torch.zeros(model_dim, device=device, dtype=dtype)

dynamics_weights = torch.randn((model_dim, model_dim), device=device, dtype=dtype, generator=rng_torch) * init_std
dynamics_weights[torch.eye(model_dim, device=device, dtype=torch.bool)] = 1
dynamics_weights = dynamics_max_eig * dynamics_weights / torch.max(torch.abs(torch.linalg.eigvals(dynamics_weights)))

inputs_weights_log = torch.randn(model_dim, device=device, dtype=dtype, generator=rng_torch) * init_std - var_log_offset
dynamics_cov_log = torch.randn(model_dim, device=device, dtype=dtype, generator=rng_torch) * init_std - var_log_offset
emissions_cov_log = torch.randn(model_dim, device=device, dtype=dtype, generator=rng_torch) * init_std - var_log_offset

# after initializing, require gradient on all the tensors
dynamics_weights.requires_grad = True
inputs_weights_log.requires_grad = True
dynamics_cov_log.requires_grad = True
emissions_cov_log.requires_grad = True

# save a copy of the initialization for plotting
dynamics_weights_init = dynamics_weights.detach().cpu().numpy()
inputs_weights_log_init = inputs_weights_log.detach().cpu().numpy()
dynamics_cov_log_init = dynamics_cov_log.detach().cpu().numpy()
emissions_cov_log_init = emissions_cov_log.detach().cpu().numpy()


def loss_fun(model_params, emissions, inputs):
    dynamics_weights = model_params[0]
    input_weights = torch.exp(model_params[1])
    dynamics_cov = torch.exp(model_params[2])
    emissions_cov = torch.exp(model_params[3])

    ll, filtered_means, filtered_covs = \
        infer.lgssm_filter_simple(init_mean, init_cov, dynamics_weights, input_weights,
                                  dynamics_cov, emissions_cov, emissions, inputs=inputs)

    return -ll


# perform gradient descent on ll with respect to the parameters
model_params = [dynamics_weights, inputs_weights_log, dynamics_cov_log, emissions_cov_log]
optimizer = torch.optim.Adam(model_params, lr=1e-3)
loss_out = []

start = time.time()
emissions_torch = [torch.tensor(i, device=device, dtype=dtype) for i in data_final]
inputs_torch = [torch.tensor(i, device=device, dtype=dtype) for i in stim_mat]
rng_np = np.random.default_rng(random_seed)

for ep in range(num_epochs):
    randomized_data_inds = torch.randperm(num_data, device=device, generator=rng_torch)
    batch_data_inds = torch.split(randomized_data_inds, batch_size)

    for batchi, batch in enumerate(batch_data_inds):
        optimizer.zero_grad()
        loss = torch.tensor(0, device=device, dtype=dtype)

        for bi, b in enumerate(batch):
            this_data = emissions_torch[b]
            this_input = inputs_torch[b]
            init_cov = pp.estimate_cov(this_data)

            loss += loss_fun(model_params, this_data, this_input)
            print('Finished data ' + str(bi + 1) + '/' + str(len(batch)))

        loss.backward()
        loss_out.append(loss.detach().cpu().numpy())
        optimizer.step()

        print('Finished batch ' + str(batchi + 1) + '/' + str(len(batch_data_inds)))
        print('Loss = ' + str(loss_out[-1]))

    end = time.time()
    print('Finished epoch ' + str(ep + 1) + '/' + str(num_epochs))
    print('Time elapsed = ' + str(end - start))

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
plt.imshow(dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
plt.title('dynamics weights')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar(shrink=colorbar_shrink)

plt.subplot(1, 2, 2)
plt.imshow(dynamics_weights.detach().cpu().numpy() - np.identity(dynamics_weights.shape[0]), interpolation='Nearest')
plt.title('dynamics weights - I')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar(shrink=colorbar_shrink)
plt.tight_layout()

# Plot the input weights
plt.figure()
plt.plot(np.exp(inputs_weights_log_init))
plt.plot(np.exp(inputs_weights_log.detach().cpu().numpy()))
plt.legend(['init', 'final'])
plt.xlabel('neurons')
plt.ylabel('input weights')

# plot the covariances
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.exp(dynamics_cov_log_init))
plt.plot(np.exp(dynamics_cov_log.detach().cpu().numpy()))
plt.legend(['init', 'final'])
plt.xlabel('neurons')
plt.ylabel('dynamics noise cov')

plt.subplot(1, 2, 2)
plt.plot(np.exp(emissions_cov_log_init))
plt.plot(np.exp(emissions_cov_log.detach().cpu().numpy()))
plt.legend(['init', 'final'])
plt.xlabel('neurons')
plt.ylabel('emissions noise cov')
plt.tight_layout()

plt.show()

# plot some model predictions
b = torch.zeros(model_dim, device=device, dtype=dtype)
C = torch.eye(model_dim, device=device, dtype=dtype)
F = torch.zeros((model_dim, model_dim), device=device, dtype=dtype)
d = torch.zeros(model_dim, device=device, dtype=dtype)

data_example_ind = 0
data_example = emissions_torch[data_example_ind]
inputs_example = inputs_torch[data_example_ind]
init_cov_example = pp.estimate_cov(data_example)
V = torch.diag(torch.exp(inputs_weights_log))
sigma_w = torch.diag(torch.exp(dynamics_cov_log))
sigma_v = torch.diag(torch.exp(emissions_cov_log))

initial_weights_dict = {'mean': init_mean,
                        'cov': init_cov_example,
                        }

dynamic_weights_dict = {'weights': dynamics_weights,
                        'input_weights': V,
                        'bias': b,
                        'cov': sigma_w,
                        }

emission_weights_dict = {'weights': C,
                         'input_weights': F,
                         'bias': d,
                         'cov': sigma_v,
                         }

params_example = {'initial': initial_weights_dict,
                  'dynamics': dynamic_weights_dict,
                  'emissions': emission_weights_dict,
                  }

ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_cross = \
    infer.lgssm_smoother(params_example, data_example, inputs=inputs_example)

example_neurons = [1, 10, 30]
num_examples = len(example_neurons)
plt.figure()

for eni, en in enumerate(example_neurons):
    plt.subplot(num_examples, 1, eni+1)
    plt.plot(data_example[:, en].detach().cpu().numpy())
    plt.plot(torch.sum(inputs_example, dim=1).detach().cpu().numpy())
    plt.plot(filtered_means[:, en].detach().cpu().numpy())
    plt.xlabel('time steps (0.5s)')
    plt.ylabel('activity')

plt.subplot(num_examples, 1, 1)
plt.legend(['observed', 'predicted'])
plt.tight_layout()

time_range = [0, 500]
plt.figure()
for eni, en in enumerate(example_neurons):
    plt.subplot(num_examples, 1, eni+1)
    plt.plot(data_example[time_range[0]:time_range[1], en].detach().cpu().numpy())
    plt.plot(torch.sum(inputs_example[time_range[0]:time_range[1], :], dim=1).detach().cpu().numpy())
    plt.plot(filtered_means[time_range[0]:time_range[1], en].detach().cpu().numpy())
    plt.xlabel('time steps (0.5s)')
    plt.ylabel('activity')

plt.subplot(num_examples, 1, 1)
plt.legend(['observed', 'predicted'])
plt.tight_layout()


plt.show()
a=1


