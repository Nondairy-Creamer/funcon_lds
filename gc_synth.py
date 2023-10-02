import numpy as np
import loading_utilities as lu
from matplotlib import pyplot as plt
import matplotlib as mpl
from ssm_classes import Lgssm
import gc_utilities as gcu

# rng=0
colormap = mpl.colormaps['coolwarm']
run_params = lu.get_run_params(param_name='submission_scripts/gc_test.yml')
fig_path = run_params['fig_path']

rng = np.random.default_rng(run_params['random_seed'])

# define the model, setting specific parameters
model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                   param_props=run_params['param_props'],
                   dynamics_lags=run_params['dynamics_lags'], dynamics_input_lags=run_params['dynamics_input_lags'])
model_true.randomize_weights(rng=rng)
model_true.emissions_weights_init = np.eye(model_true.emissions_dim, model_true.dynamics_dim_full)
model_true.emissions_input_weights_init = np.zeros((model_true.emissions_dim, model_true.input_dim_full))
model_true.set_to_init()

# sample from the randomized model
data_dict = model_true.sample(num_time=run_params['num_time'], num_data_sets=run_params['num_data_sets'],
                              scattered_nan_freq=run_params['scattered_nan_freq'],
                              lost_emission_freq=run_params['lost_emission_freq'],
                              input_time_scale=run_params['input_time_scale'], rng=rng)

emissions = data_dict['emissions']
inputs = data_dict['inputs']
latents_true = data_dict['latents']
init_mean_true = data_dict['init_mean']
init_cov_true = data_dict['init_cov']
num_data_sets = run_params['num_data_sets']
num_neurons = emissions[0].shape[1]

num_time = np.zeros(len(emissions))
for i in range(len(emissions)):
    num_time[i] = len(emissions[i])

model_params = model_true.get_params()
A_true = model_params['trained']['dynamics_weights']
B_true = model_params['trained']['dynamics_input_weights']

# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)
emissions_num_lags = run_params['dynamics_lags']
inputs_num_lags = run_params['dynamics_input_lags']

all_a_hat, all_a_hat_0, all_b_hat, mse = gcu.run_gc(num_data_sets, emissions_num_lags, inputs_num_lags, num_neurons,
                                                    inputs, emissions, f_name='synth_data',
                                                    load_dir='/Users/lsmith/Documents/python/', rerun=True)

# create averaged a_hat and b_hat matrices over all non-NaN values over all datasets
# save all a_hat and b_hat full mtxes first as 3d array, then nanmean over each element along 3rd axis
a_hat_avg = np.nanmean(all_a_hat, axis=2)
b_hat_avg = np.nanmean(all_b_hat, axis=2)

# pick subset of neurons to look at
cell_ids = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL']
cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL']
neuron_to_stim = 'AFDL'
# array of neuron indices
neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])
neuron_stim_index = cell_ids.index(neuron_to_stim)

num_sim = 120

avg_pred_x_all_data = gcu.impulse_response_func(num_sim, cell_ids, cell_ids_chosen, num_neurons, num_data_sets,
                                                emissions, inputs, all_a_hat, all_b_hat, emissions_num_lags,
                                                inputs_num_lags, f_name='synth_impulse_response_data',
                                                rerun=run_params['imp_resp_func_rerun'])

gcu.plot_l2_norms(emissions, inputs, cell_ids, cell_ids_chosen, avg_pred_x_all_data, colormap,
                  save=False, fig_path=fig_path)

# plot on y axis the gc results for a chosen neuron after stimulus of another neuron
# so plot the a_hat matrix value corresponding to these two neurons vs time, where we start later in time lags and go up
# until no lags

gcu.plot_imp_resp(emissions, inputs, neuron_inds_chosen, num_neurons, num_data_sets, cell_ids, cell_ids_chosen,
                  neuron_to_stim, avg_pred_x_all_data, save=False, fig_path=fig_path)


# for d in range(num_data_sets):
#     plt.figure()
#     plt.title('dataset %(dataset)i GC for %(lags)i lags: a_hat' % {"dataset": d, "lags": emissions_num_lags})
#     a_hat_pos = plt.imshow(all_a_hat[:, :, d], aspect='auto', interpolation='nearest', cmap=colormap)
#     plt.colorbar(a_hat_pos)
#     plt.show()
#
#     plt.figure()
#     plt.title('dataset %(dataset)i GC for %(lags)i lags: b_hat' % {"dataset": d, "lags": inputs_num_lags})
#     b_hat_pos = plt.imshow(all_b_hat[:, :, d], aspect='auto', interpolation='nearest', cmap=colormap)
#     plt.colorbar(b_hat_pos)
#     plt.show()

# fig3, axs3 = plt.subplots(nrows=1, ncols=1)
# plt.title('averaged a_hat over all datasets')
# avg_a_hat_pos = plt.imshow(a_hat_avg, interpolation='nearest', cmap=colormap)
# color_limits = np.nanmax(np.abs(a_hat_avg))
# plt.clim((-color_limits, color_limits))
# plt.colorbar(avg_a_hat_pos)
# plt.show()
#
# A_pos = plt.imshow(A_true[:num_neurons, :], interpolation='nearest', cmap=colormap)
# plt.title('true A')
# plt.clim((-color_limits, color_limits))
# plt.colorbar(A_pos)
# plt.show()
#
# fig4, axs4 = plt.subplots(nrows=1, ncols=1)
# plt.title('averaged b_hat over all datasets')
# avg_b_hat_pos = plt.imshow(b_hat_avg, interpolation='nearest', cmap=colormap)
# color_limits = np.nanmax(np.abs(b_hat_avg))
# plt.clim((-color_limits, color_limits))
# plt.colorbar(avg_b_hat_pos)
# plt.show()
#
# B_pos = plt.imshow(B_true[:num_neurons, :], interpolation='nearest', cmap=colormap)
# plt.title('true dynamics_input_weights (B)')
# plt.clim((-color_limits, color_limits))
# plt.colorbar(B_pos)
# plt.show()

a = 0