import numpy as np
import pickle
import analysis_utilities as au
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path
import scipy

colormap = mpl.colormaps['coolwarm']

# load in the model and training data
# model_folder = Path('/home/mcreamer/Documents/data_sets/fun_con/fun_con_fullprocess/')
data_folder = Path('/home/mcreamer/Documents/data_sets/fun_con/filtered_interp_cbleach/')
# data_folder = Path('/home/mcreamer/Documents/data_sets/fun_con/filtered_nointerp_cbleach/')
# data_folder2 = Path('/home/mcreamer/Documents/data_sets/fun_con/unfiltered_nointerp_cbleach/')
data_folder2 = Path('/home/mcreamer/Documents/data_sets/fun_con/filtered_interp_fbleach/')
# data_folder2 = Path('/home/mcreamer/Documents/data_sets/fun_con/filtered_nointerp_fbleach/')
cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
# cell_ids_chosen = None

data_path = data_folder / 'data_train.pkl'
data_file = open(data_path, 'rb')
data = pickle.load(data_file)
data_file.close()

data_path2 = data_folder2 / 'data_train.pkl'
data_file2 = open(data_path2, 'rb')
data2 = pickle.load(data_file2)
data_file2.close()

emissions = data['emissions']
inputs = data['inputs']
cell_ids = data['cell_ids']

emissions2 = data2['emissions']
inputs2 = data2['inputs']
cell_ids2 = data2['cell_ids']

data_ind = 10
neuron_ind = 101
time_range = (0, 10000)
# neurons_to_plot = ~np.all(np.isnan(emissions[data_ind]), axis=0)

plt.figure()
plt.plot(emissions[data_ind][time_range[0]:time_range[1], neuron_ind])
plt.plot(emissions2[data_ind][time_range[0]:time_range[1], neuron_ind])
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(emissions[data_ind][time_range[0]:time_range[1], :].T, interpolation='nearest', aspect='auto')
plt.subplot(2, 1, 2)
plt.imshow(emissions2[data_ind][time_range[0]:time_range[1], :].T, interpolation='nearest', aspect='auto')
plt.show()


#
#
# window = (-20, 60)
#
# filter_std = 50
# filter_mult = 3
# window_size = np.arange(-filter_mult * filter_std, filter_mult * filter_std + 1)
# norm_filter = scipy.stats.norm.pdf(window_size, 0, filter_std)
# moving_variance = []
# slopes = np.zeros((len(emissions), emissions[0].shape[1]))
#
# for ei, e in enumerate(emissions):
#     size_after_filter = e.shape[0] - window_size.shape[0] + 1
#
#     if size_after_filter > 1:
#         this_var = np.zeros((size_after_filter, e.shape[1]))
#
#         x = np.arange(size_after_filter)[:, None]
#         b = np.ones((size_after_filter, 1))
#         xb = np.concatenate((x, b), axis=1)
#
#         for n in range(e.shape[1]):
#             mean_of_square = au.nan_convolve(e[:, n]**2, norm_filter)
#             square_of_mean = au.nan_convolve(e[:, n], norm_filter)**2
#             this_var[:, n] = mean_of_square - square_of_mean
#
#             non_nan_loc = ~np.isnan(this_var[:, n])
#
#             if np.sum(non_nan_loc) > 1:
#                 xb_this = xb.copy()
#                 xb_this = xb_this[non_nan_loc, :]
#                 y = this_var[:, n]
#                 y = y[non_nan_loc]
#
#                 slopes[ei, n] = np.linalg.lstsq(xb_this, y)[0][0]
#             else:
#                 slopes[ei, n] = np.nan
#
#         moving_variance.append(this_var)
#     else:
#         moving_variance.append(np.nan)
#
# ave_slope = np.nanmean(slopes, axis=0)
# plt.figure()
# plt.hist(ave_slope)
# plt.figure()
# plt.hist(slopes.reshape(-1))
# plt.show()
#
# if cell_ids_chosen is None:
#     cell_ids_chosen = cell_ids.copy()
#
# chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
# measured_irf, measured_irf_std, measured_irf_all = au.get_stim_response(emissions, inputs, window=window)
# measured_irf_l2 = au.rms(measured_irf, axis=0)
# measured_irf_mean = np.nanmean(measured_irf, axis=0)
#
# cmax = np.nanpercentile(np.abs(measured_irf_mean), 95)
# num_plot = 50
# plt.figure()
# ax = plt.gca()
# plt.imshow(measured_irf_l2[:num_plot, :num_plot], interpolation='nearest', cmap=mpl.colormaps['coolwarm'])
# plt.xticks(ticks=np.arange(num_plot), labels=cell_ids[:num_plot])
# plt.yticks(ticks=np.arange(num_plot), labels=cell_ids[:num_plot])
# for label in ax.get_xticklabels():
#     label.set_rotation(90)
# plt.clim((-cmax, cmax))
#
# plt.show()
#
# stim_neuron = 'AVJR'
# resp_neuron = 'AVDR'
#
# stim_neuron_ind = cell_ids.index('AVJR')
# target_neuron_ind = cell_ids.index('AVDR')
#
# stim_responses = measured_irf_all[stim_neuron_ind]
# stim_responses_auto = stim_responses[:, :, stim_neuron_ind]
# stim_responses_target = stim_responses[:, :, target_neuron_ind]
#
# measured_target = np.any(~np.isnan(stim_responses_target), axis=1)
# stim_responses_auto = stim_responses_auto[measured_target, :]
# stim_responses_target = stim_responses_target[measured_target, :]
#
# cmax = np.nanpercentile(np.abs([stim_responses_auto, stim_responses_target]), 95)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(stim_responses_auto, interpolation='nearest', aspect='auto', cmap=colormap)
# plt.axvline(20)
# plt.clim(-cmax, cmax)
# plt.subplot(1, 2, 2)
# plt.imshow(stim_responses_target, interpolation='nearest', aspect='auto', cmap=colormap)
# plt.axvline(20)
# plt.clim(-cmax, cmax)
# plt.show()
# debug=1