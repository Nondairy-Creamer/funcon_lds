import numpy as np
import loading_utilities as lu
from matplotlib import pyplot as plt
import matplotlib as mpl
import gc_utilities as gcu

make_avg_med_figs = True
make_best_figs = True

colormap = mpl.colormaps['coolwarm']
run_params = lu.get_run_params(param_name='submission_scripts/gc_exp_test.yml')
fig_path = run_params['fig_path']

A, num_neurons, num_data_sets, num_neurons, emissions, inputs, cell_ids = gcu.gc_preprocessing(run_params, rerun=False)

emissions_num_lags = run_params['dynamics_lags']
inputs_num_lags = run_params['dynamics_input_lags']

# num_data_sets = 1 #for testing

all_a_hat, all_a_hat_0, all_b_hat, mse = gcu.run_gc(num_data_sets, emissions_num_lags, inputs_num_lags, num_neurons,
                                                    inputs, emissions, rerun=False)

# pick subset of neurons to look at
cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
# neuron_to_remove = 'AVDL'
neuron_to_stim = 'AFDR'
# array of neuron indices
neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])
neuron_stim_index = cell_ids.index(neuron_to_stim)

# give best dataset with high coverage of neurons measured and lowest mse
best_data_ind = np.where(mse < .0375)
nan_count = np.zeros(num_data_sets)
test_neuron_stimmed = np.full(num_data_sets, False)
for i in range(num_data_sets):
    nan_count[i] = np.count_nonzero(np.isnan(emissions[i][0, :]))
    test_neuron_stimmed[i] = np.any(inputs[i][:, neuron_stim_index])
plt.figure()
plt.scatter(nan_count[best_data_ind], mse[best_data_ind])
plt.show()
mask1 = np.logical_and(mse < .037, nan_count < 175)
mask2 = np.logical_and(mask1, test_neuron_stimmed)
best_data_set = np.where(mask2)[0][0]

# PLOTTING:
# we want the colorbars to be the same scale for all datasets to easily compare values between them
# so, calc the max and min values to set the colorbar scale, there are some large outliers so omit them from the
# colorbar

color_limits_a_hat = np.nanquantile(np.abs(all_a_hat).flatten(), 0.99)
color_limits_a_hat_0 = np.nanquantile(np.abs(all_a_hat_0).flatten(), 0.99)
color_limits_b_hat = np.nanquantile(np.abs(all_b_hat).flatten(), 0.99)

# color_limits = np.nanmax(np.abs(A))
# A_pos = plt.imshow(A, interpolation='nearest', cmap=colormap)
# plt.clim((-color_limits, color_limits))
# plt.colorbar(A_pos)
# # plt.show()

if make_avg_med_figs:
    gcu.plot_weights_all_data_sets(all_a_hat_0, num_data_sets, colormap, color_limits_a_hat_0, emissions_num_lags,
                                   save=True, fig_path=fig_path, data_name='100a_hat')
    gcu.plot_weights_all_data_sets(all_b_hat, num_data_sets, colormap, color_limits_b_hat, inputs_num_lags, save=True,
                                   fig_path=fig_path, data_name='100b_hat')

    # create averaged a_hat and b_hat matrices over all non-NaN values over all datasets
    # save all a_hat and b_hat full mtxes first as 3d array, then nanmean over each element along 3rd axis
    a_hat_avg = np.nanmean(all_a_hat, axis=2)
    a_hat_0_avg = np.nanmean(all_a_hat_0, axis=2)
    b_hat_avg = np.nanmean(all_b_hat, axis=2)

    # repeat but with median instead of mean
    a_hat_median = np.nanmedian(all_a_hat, axis=2)
    a_hat_0_median = np.nanmedian(all_a_hat_0, axis=2)
    b_hat_median = np.nanmedian(all_b_hat, axis=2)

    # mean
    gcu.plot_weights_mean_median(a_hat_avg, colormap, save=True, fig_path=fig_path, data_title='averaged a_hat over all datasets',
                                 data_name='100avg_a_hat')
    gcu.plot_weights_mean_median(b_hat_avg, colormap, save=True, fig_path=fig_path, data_title='averaged b_hat over all datasets',
                                 data_name='100avg_b_hat')
    gcu.plot_weights_mean_median(a_hat_0_avg, colormap, save=True, fig_path=fig_path,
                                 data_title='100averaged a_hat, diag=0 over all datasets', data_name='avg_a_hat_0')
    # median
    gcu.plot_weights_mean_median(a_hat_median, colormap, save=True, fig_path=fig_path, data_title='median a_hat over all datasets',
                                 data_name='100median_a_hat')
    gcu.plot_weights_mean_median(b_hat_median, colormap, save=True, fig_path=fig_path, data_title='median b_hat over all datasets',
                                 data_name='100median_b_hat')
    gcu.plot_weights_mean_median(a_hat_0_median, colormap, save=True, fig_path=fig_path,
                                 data_title='100median a_hat, diag=0 over all datasets', data_name='median_a_hat_0')

    # if emissions_num_lags > 1:
    #     gcu.plot_weights_mean_median_split(a_hat_avg, colormap, emissions_num_lags, save=True, fig_path=fig_path,
    #                                        data_title='averaged a_hat over all datasets', data_name='avg_a_hat_split')
    #     gcu.plot_weights_mean_median_split(a_hat_0_avg, colormap, emissions_num_lags, save=True, fig_path=fig_path,
    #                                        data_title='averaged a_hat (diag=0) over all datasets',
    #                                        data_name='avg_a_hat_0_split')
    #     gcu.plot_weights_mean_median_split(a_hat_median, colormap, emissions_num_lags, save=True, fig_path=fig_path,
    #                                        data_title='median a_hat over all datasets', data_name='median_a_hat_split')
    #     gcu.plot_weights_mean_median_split(a_hat_0_median, colormap, emissions_num_lags, save=True, fig_path=fig_path,
    #                                        data_title='median a_hat (diag=0) over all datasets',
    #                                        data_name='median_a_hat_0_split')

    # if inputs_num_lags > 1:
    #     gcu.plot_weights_mean_median_split(b_hat_avg, colormap, inputs_num_lags, save=True, fig_path=fig_path,
    #                                        data_title='averaged b_hat over all datasets', data_name='avg_b_hat_split')
    #     gcu.plot_weights_mean_median_split(b_hat_median, colormap, inputs_num_lags, save=True, fig_path=fig_path,
    #                                        data_title='median b_hat over all datasets', data_name='median_b_hat_split')

    # fitted input neurons (bhat) vs lags in time
    gcu.plot_input_weights_neurons(b_hat_avg, inputs_num_lags, cell_ids, colormap, '100avg_b_hat', False, [], save=True,
                                   fig_path=fig_path)
    gcu.plot_input_weights_neurons(b_hat_median, inputs_num_lags, cell_ids, colormap, '100median_b_hat', False, [],
                                   save=True, fig_path=fig_path)
    gcu.plot_input_weights_neurons(b_hat_avg, inputs_num_lags, cell_ids, colormap,
                                   '100avg_b_hat_subset', True, neuron_inds_chosen, save=True, fig_path=fig_path)
    gcu.plot_input_weights_neurons(b_hat_median, inputs_num_lags, cell_ids, colormap,
                                   '100median_b_hat_subset', True, neuron_inds_chosen, save=True, fig_path=fig_path)

    # eval stuff
    zero_nan_a_hat = a_hat_avg
    zero_nan_a_hat[np.isnan(zero_nan_a_hat)] = 0
    eig_vals = np.linalg.eigvals(zero_nan_a_hat[:, :num_neurons])

    plt.figure()
    plt.scatter(np.real(eig_vals), np.imag(eig_vals))
    # plt.show()
    string = fig_path + 'eigs.png'
    plt.savefig(string)

    zero_nan_a_hat_med = a_hat_median
    zero_nan_a_hat_med[np.isnan(zero_nan_a_hat_med)] = 0
    eig_vals = np.linalg.eigvals(zero_nan_a_hat_med[:, :num_neurons])

    plt.figure()
    plt.scatter(np.real(eig_vals), np.imag(eig_vals))
    # plt.show()
    string = fig_path + 'eigs_med.png'
    plt.savefig(string)

if make_best_figs:
    gcu.plot_weights_mean_median(all_b_hat[:, :, best_data_set], colormap, save=True, fig_path=fig_path,
                                 data_title='best b_hat', data_name='best_b_hat')
    gcu.plot_weights_mean_median(all_a_hat_0[:, :, best_data_set], colormap, save=True, fig_path=fig_path,
                                 data_title='best a_hat, diag=0', data_name='best_a_hat_0')

    if emissions_num_lags > 1:
        gcu.plot_weights_mean_median_split(all_a_hat_0[:, :, best_data_set], colormap, emissions_num_lags, save=True,
                                           fig_path=fig_path, data_title='best a_hat (diag=0)',
                                           data_name='best_a_hat_0_split')

    gcu.plot_input_weights_neurons(all_b_hat[:, :, best_data_set], inputs_num_lags, cell_ids, colormap, 'best_b_hat',
                                   False, [], save=True, fig_path=fig_path)
    gcu.plot_input_weights_neurons(all_b_hat[:, :, best_data_set], inputs_num_lags, cell_ids, colormap,
                                   'best_b_hat_subset', True, neuron_inds_chosen, save=True, fig_path=fig_path)

    nans = np.any(np.isnan(emissions[best_data_set]), axis=0)
    best_a_no_nans = all_a_hat[~nans, :, best_data_set]
    nans_mask = np.tile(nans, emissions_num_lags)
    best_a_no_nans = best_a_no_nans[:, ~nans_mask]

    color_lims = np.nanquantile(np.abs(best_a_no_nans).flatten(), 0.99)
    plt.figure()
    plt.imshow(best_a_no_nans, aspect='auto', interpolation='none', cmap=colormap)
    plt.clim((-color_lims, color_lims))
    plt.colorbar()
    string = fig_path + 'best_a_hat_no_nans.png'
    plt.savefig(string)

    # eig_vals = np.linalg.eigvals(best_a_no_nans)
    # plt.figure()
    # plt.scatter(np.real(eig_vals), np.imag(eig_vals))
    # string = fig_path + 'eigs_best.png'
    # plt.savefig(string)



# set NaNs to 0, feed in an input vector for a specific neuron, run GC model, should see some sort of response
# impulse response functions:

# sample from model for a specific stimulated neuron for init_time
# init_time = num_lags
# how long to run the model forward and simulate
num_sim = 120

avg_pred_x_all_data = gcu.impulse_response_func(num_sim, cell_ids, cell_ids_chosen, num_neurons, num_data_sets,
                                                emissions, inputs, all_a_hat, all_b_hat, emissions_num_lags,
                                                inputs_num_lags, rerun=True)

gcu.plot_l2_norms(neuron_inds_chosen, emissions, inputs, cell_ids, cell_ids_chosen, avg_pred_x_all_data, colormap,
                  save=True, fig_path=fig_path)

# plot on y axis the gc results for a chosen neuron after stimulus of another neuron
# so plot the a_hat matrix value corresponding to these two neurons vs time, where we start later in time lags and go up
# until no lags

gcu.plot_imp_resp(emissions, inputs, neuron_inds_chosen, num_neurons, num_data_sets, cell_ids, cell_ids_chosen,
                  neuron_to_stim, avg_pred_x_all_data, save=True, fig_path=fig_path)

# do these plots for best data set
avg_pred_x_all_data_best = gcu.impulse_response_func(num_sim, cell_ids, cell_ids_chosen, num_neurons, 1,
                                                     [emissions[best_data_set]], [inputs[best_data_set]],
                                                     np.reshape(all_a_hat[:, :, best_data_set],
                                                               (num_neurons, num_neurons*emissions_num_lags, 1)),
                                                     np.reshape(all_b_hat[:, :, best_data_set],
                                                               (num_neurons, num_neurons*inputs_num_lags, 1)),
                                                     emissions_num_lags, inputs_num_lags,
                                                     f_name='impulse_response_data_best', rerun=True)
gcu.plot_l2_norms(neuron_inds_chosen, emissions, inputs, cell_ids, cell_ids_chosen, avg_pred_x_all_data_best,
                  colormap, save=True, fig_path=fig_path + 'best_')
gcu.plot_imp_resp(emissions, inputs, neuron_inds_chosen, num_neurons, num_data_sets, cell_ids, cell_ids_chosen,
                  neuron_to_stim, avg_pred_x_all_data_best, save=True, fig_path=fig_path + 'best_')
a=0

# todo: hold out 30 datasets for test ds, train on rest
# todo: find 'best' dataset that has good coverage of neurons