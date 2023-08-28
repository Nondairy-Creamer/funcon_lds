import numpy as np
import pickle
import analysis_utilities as au
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path

colormap = mpl.colormaps['coolwarm']

# load in the model and training data
model_folder = Path('/home/mcreamer/Documents/data_sets/fun_con/fun_con_fullprocess/')
# model_folder = Path('/home/mcreamer/Documents/data_sets/fun_con/fun_con_nointerp/')
cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
# cell_ids_chosen = None
data_path = model_folder / 'data_train.pkl'

data_file = open(data_path, 'rb')
data = pickle.load(data_file)
data_file.close()

emissions = data['emissions']
inputs = data['inputs']
cell_ids = data['cell_ids']
window = (-20, 60)

if cell_ids_chosen is None:
    cell_ids_chosen = cell_ids.copy()

chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
measured_irf, measured_irf_all = au.get_stim_response(emissions, inputs, window=window)
measured_irf_l2 = au.rms(measured_irf, axis=0)
measured_irf_mean = np.nanmean(measured_irf, axis=0)

cmax = np.nanpercentile(np.abs(measured_irf_mean), 95)
num_plot = 50
plt.figure()
ax = plt.gca()
plt.imshow(measured_irf_mean[:num_plot, :num_plot], interpolation='nearest', cmap=mpl.colormaps['coolwarm'])
plt.xticks(ticks=np.arange(num_plot), labels=cell_ids[:num_plot])
plt.yticks(ticks=np.arange(num_plot), labels=cell_ids[:num_plot])
for label in ax.get_xticklabels():
    label.set_rotation(90)
plt.clim((-cmax, cmax))

plt.show()

stim_neuron = 'AVJR'
resp_neuron = 'AVDR'

stim_neuron_ind = cell_ids.index('AVJR')
target_neuron_ind = cell_ids.index('AVDR')

stim_responses = measured_irf_all[stim_neuron_ind]
stim_responses_auto = stim_responses[:, :, stim_neuron_ind]
stim_responses_target = stim_responses[:, :, target_neuron_ind]

measured_target = np.any(~np.isnan(stim_responses_target), axis=1)
stim_responses_auto = stim_responses_auto[measured_target, :]
stim_responses_target = stim_responses_target[measured_target, :]

cmax = np.nanpercentile(np.abs([stim_responses_auto, stim_responses_target]), 95)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(stim_responses_auto, interpolation='nearest', aspect='auto', cmap=colormap)
plt.axvline(20)
plt.clim(-cmax, cmax)
plt.subplot(1, 2, 2)
plt.imshow(stim_responses_target, interpolation='nearest', aspect='auto', cmap=colormap)
plt.axvline(20)
plt.clim(-cmax, cmax)
plt.show()
debug=1