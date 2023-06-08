import numpy as np
from matplotlib import pyplot as plt
import pumpprobe as pp, wormdatamodel as wormdm


data_folder = '/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210930/pumpprobe_20210930_101310/'
# this is the raw data
raw_data_name = 'green.txt'
# this is the data coming from the funatlas
processed_data_path = 'green.pickle'


raw_data = np.loadtxt(data_folder + raw_data_name)

signal_kwargs = {"remove_spikes": True,
                 "smooth": False, "smooth_mode": "sg_causal",
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,
                 "matchless_nan_th_from_file": False, "matchless_nan_th": 0.5,
                 "matchless_nan_th_added_only": True,
                 "verbose": False,
                 "preprocess": False}
funa = pp.Funatlas.from_datasets([data_folder], ds_tags=None, ds_exclude_tags="mutant",
                                 merge_bilateral=False, merge_dorsoventral=False,
                                 merge_numbered=False, merge_AWC=False,
                                 signal="green", signal_kwargs=signal_kwargs,
                                 verbose=False)

processed_data = funa.sig[0].data

neuron_ind = 3
time_window = (500, 1000)

plt.figure()

plt.subplot(2, 1, 1)
plt.imshow(processed_data[time_window[0]:time_window[1], :].T, interpolation='nearest', aspect='auto')
plt.title('fun atlas data')
plt.ylabel('neurons')
plt.xlabel('time')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(raw_data[time_window[0]:time_window[1], :].T, interpolation='nearest', aspect='auto')
plt.title('raw data')
plt.ylabel('neurons')
plt.xlabel('time')
plt.colorbar()

plt.tight_layout()
plt.figure()

plt.plot(processed_data[time_window[0]:time_window[1], neuron_ind])
plt.plot(raw_data[time_window[0]:time_window[1], neuron_ind])

plt.legend(['fun atlas processed', 'raw data'])
plt.show()
