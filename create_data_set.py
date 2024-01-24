import numpy as np
import time
import scipy.io as sio
from pathlib import Path
import loading_utilities as lu
import pickle
import tmac.preprocessing as tp
import scipy.signal as ssig

run_params = lu.get_run_params(param_name='submission_params/create_data_set.yml')
data_path = Path(run_params['data_path'])
start_index = run_params['start_index']
filter_size = run_params['filter_size']
correct_photobleach = run_params['correct_photobleach']
interpolate_nans = run_params['interpolate_nans']
upsample_factor = run_params['upsample_factor']

preprocess_filename = 'funcon_preprocessed_data.pkl'

# look for data from francesco's pipeline
for i in sorted(data_path.rglob('francesco_green.npy'))[::-1]:
    preprocess_path = i.parent / preprocess_filename
    this_emissions = np.load(str(i))

    if not interpolate_nans:
        this_nan_mask = np.load(str(i.parent / 'nan_mask.npy'))
        this_emissions[this_nan_mask] = np.nan

    this_cell_ids = list(np.load(str(i.parent / 'labels.npy')))

    # load stimulation data
    this_stim_cell_ids = np.load(str(i.parent / 'stim_recording_cell_inds.npy'), allow_pickle=True)
    this_stim_volume_inds = np.load(str(i.parent / 'stim_volumes_inds.npy'), allow_pickle=True)

    this_inputs = np.zeros_like(this_emissions)
    nonnegative_inds = (this_stim_cell_ids != -1) & (this_stim_cell_ids != -2) & (this_stim_cell_ids != -3)
    this_stim_volume_inds = this_stim_volume_inds[nonnegative_inds]
    this_stim_cell_ids = this_stim_cell_ids[nonnegative_inds]
    this_inputs[this_stim_volume_inds, this_stim_cell_ids] = 1

    start = time.time()
    this_emissions, this_inputs = lu.preprocess_data(this_emissions, this_inputs, start_index=start_index,
                                                     correct_photobleach=correct_photobleach,
                                                     filter_size=filter_size, upsample_factor=upsample_factor)

    if interpolate_nans:
        full_nan_loc = np.all(np.isnan(this_emissions), axis=0)
        interp_emissions = tp.interpolate_over_nans(this_emissions[:, ~full_nan_loc])[0]
        this_emissions[:, ~full_nan_loc] = interp_emissions

    preprocessed_file = open(preprocess_path, 'wb')
    pickle.dump({'emissions': this_emissions, 'inputs': this_inputs, 'cell_ids': this_cell_ids}, preprocessed_file)
    preprocessed_file.close()

    print('Data set', i.parent, 'preprocessed')
    print('Took', time.time() - start, 's')

# look for data from my pipeline
for i in sorted(data_path.rglob('calcium_to_multicolor_alignment.mat'))[::-1]:
    preprocess_path = i.parent / preprocess_filename
    c2m_align = sio.loadmat(str(i.parent / 'calcium_to_multicolor_alignment.mat'))
    this_emissions = c2m_align['calcium_data'][0, 0]['gRaw'].T
    this_inputs = np.zeros_like(this_emissions)
    this_cell_ids = []

    # downsample from 6hz to 2hz
    filter_shape = np.ones((3, 1)) / 3
    this_emissions = ssig.convolve2d(this_emissions, filter_shape, mode='same')
    this_emissions = this_emissions[::3, :]
    this_inputs = this_inputs[::3, :]

    for j in c2m_align['labels'][0, 0]['tracked_human_labels']:
        if len(j[0]) > 0:
            this_cell_ids.append(str(j[0][0]))
        else:
            this_cell_ids.append('')

    start = time.time()
    this_emissions, this_inputs = lu.preprocess_data(this_emissions, this_inputs, start_index=start_index,
                                                     correct_photobleach=correct_photobleach,
                                                     filter_size=filter_size)

    if interpolate_nans:
        full_nan_loc = np.all(np.isnan(this_emissions), axis=0)
        interp_emissions = tp.interpolate_over_nans(this_emissions[:, ~full_nan_loc])[0]
        this_emissions[:, ~full_nan_loc] = interp_emissions

    preprocessed_file = open(preprocess_path, 'wb')
    pickle.dump({'emissions': this_emissions, 'inputs': this_inputs, 'cell_ids': this_cell_ids},
                preprocessed_file)
    preprocessed_file.close()

    print('Data set', i.parent, 'preprocessed')
    print('Took', time.time() - start, 's')

