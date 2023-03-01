import numpy as np
import torch
import preprocessing as pp
from matplotlib import pyplot as plt
from ssm_classes import LgssmSimple
import yaml

from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
# from torch.utils.tensorboard import SummaryWriter

# the goal of this function is to take the pairwise stimulation and response data from
# https://arxiv.org/abs/2208.04790
# this data is a collection of calcium recordings of ~100 neurons over ~5-15 minutes where individual neurons are
# randomly targets and stimulated optogenetically
# We want to fit a linear dynamical system to the data in order to infer the connection weights between neurons
# The model is of the form
# x_t = A @ x_(t-1) + V @ u_t + w_t
# y_t = x_t + v_t

# the mapping of x_t to y_t is the identity
# V is diagonal
# w_t, v_t are gaussian with 0 mean, and diagonal covariance

# get run parameters, yaml file contains descriptions of the parameters
with open('run_params.yaml', 'r') as file:
    run_params = yaml.safe_load(file)

device = run_params['device']
dtype = getattr(torch, run_params['dtype'])

# load all the recordings
emissions_unaligned, cell_ids_unaligned, q, q_labels, stim_cell_ids, inputs_unaligned = \
    pp.load_data(run_params['data_path'])

# remove recordings that are noisy
data_sets_to_remove = np.sort(run_params['bad_data_sets'])[::-1]
for bd in data_sets_to_remove:
    emissions_unaligned.pop(bd)
    cell_ids_unaligned.pop(bd)
    inputs_unaligned.pop(bd)
    stim_cell_ids.pop(bd)

# choose a subset of the data sets to maximize the number of recordings * the number of neurons included
cell_ids, emissions, best_runs, inputs = \
    pp.get_combined_dataset(emissions_unaligned, cell_ids_unaligned, stim_cell_ids, inputs_unaligned,
                            frac_neuron_coverage=run_params['frac_neuron_coverage'],
                            minimum_freq=run_params['minimum_frac_measured'])

num_data = len(emissions)
num_epochs = int(np.ceil(run_params['num_grad_steps'] * run_params['batch_size'] / num_data))

# remove the beginning of the recording which contains artifacts and mean subtract
for ri in range(num_data):
    emissions[ri] = emissions[ri][run_params['index_start']:, :]
    emissions[ri] = emissions[ri] - np.mean(emissions[ri], axis=0, keepdims=True)
    inputs[ri] = inputs[ri][run_params['index_start']:, :]

# initialize and train model
latent_dim = len(cell_ids)
lgssm_model = LgssmSimple(latent_dim, dtype=dtype, device=device, random_seed=run_params['random_seed'], verbose=run_params['verbose'])
loss_out = lgssm_model.fit_batch_sgd(emissions, inputs, learning_rate=run_params['learning_rate'],
                                     num_steps=run_params['num_grad_steps'], batch_size=run_params['batch_size'],
                                     num_splits=run_params['num_splits'])

if run_params['save_model']:
    lgssm_model.save(path=run_params['save_path'])

if run_params['plot_figures']:
    import matplotlib as mpl
    mpl.use('TkAgg')

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
