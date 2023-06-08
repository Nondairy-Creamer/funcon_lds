import numpy as np
from matplotlib import pyplot as plt


data_folder = '/home/mcreamer/Documents/data_sets/fun_con_unfiltered/pumpprobe_20210917_104948'
# this is directly loaded from green.txt using np.readtext
raw_data_name = 'francesco_green_raw.npy'
# this is the data coming from the funatlas
processed_data_path = 'francesco_green.npy'


raw_data = np.load(data_folder + '/' + raw_data_name)
processed_data = np.load(data_folder + '/' + processed_data_path)

neuron_ind = 102
time_window = (500, 100000)

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

a=1

