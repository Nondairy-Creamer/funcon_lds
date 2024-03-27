import numpy as np
from pathlib import Path
import yaml
import pickle
import tmac.preprocessing as tp
import analysis_utilities as au
import warnings
import os


# utilities for loading and saving the data
def get_run_params(param_name):
    # load in the parameters for the run which dictate how many data sets to use,
    # or how many time lags the model will fit etc

    with open(param_name, 'r') as file:
        params = yaml.safe_load(file)

    return params


def preprocess_data(emissions, inputs, start_index=0, correct_photobleach=False, filter_size=2, upsample_factor=1):
    # remove the beginning of the recording which contains artifacts and mean subtract
    emissions = emissions[start_index:, :]
    inputs = inputs[start_index:, :]

    # remove stimulation events with interpolation
    window = np.array((-2, 3))
    for c in range(emissions.shape[1]):
        stim_locations = np.where(inputs[:, c])[0]

        for s in stim_locations:
            data_x = window + s
            interp_x = np.arange(data_x[0], data_x[1])
            emissions[interp_x, c] = np.interp(interp_x, data_x, emissions[data_x, c])

    if filter_size > 0:
        # filter out noise at the nyquist frequency
        filter_shape = np.ones(filter_size) / filter_size
        emissions_filtered = np.zeros((emissions.shape[0] - filter_size + 1, emissions.shape[1]))

        for c in range(emissions.shape[1]):
            emissions_filtered[:, c] = au.nan_convolve(emissions[:, c], filter_shape)
    else:
        emissions_filtered = emissions.copy()

    if correct_photobleach:
        # photobleach correction
        emissions_filtered_corrected = np.zeros_like(emissions_filtered)
        for c in range(emissions_filtered.shape[1]):
            emissions_filtered_corrected[:, c] = tp.photobleach_correction(emissions_filtered[:, c], num_exp=2, fit_offset=False)[:, 0]

        # occasionally the fit fails check for outputs who don't have a mean close to 1
        # fit those with a single exponential
        # all the nan mean calls throw warnings when averaging over nans so supress those
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bad_fits_2exp = np.where(np.abs(np.nanmean(emissions_filtered_corrected, axis=0) - 1) > 0.1)[0]

            for bf in bad_fits_2exp:
                emissions_filtered_corrected[:, bf] = tp.photobleach_correction(emissions_filtered[:, bf], num_exp=1, fit_offset=True)[:, 0]

            bad_fits_1xp = np.where(np.abs(np.nanmean(emissions_filtered_corrected, axis=0) - 1) > 0.2)[0]
            if len(bad_fits_1xp) > 0:
                warnings.warn('Photobleach correction problems found in neurons ' + str(bad_fits_1xp) + ' setting to nan')
                emissions_filtered_corrected[:, bad_fits_1xp] = np.nan

            # divide by the mean and subtract 1. Will throw warnings on the all nan data, ignore htem
            emissions_time_mean = np.nanmean(emissions_filtered_corrected, axis=0)
            emissions_filtered_corrected = emissions_filtered_corrected / emissions_time_mean - 1

        emissions_filtered_corrected[emissions_filtered_corrected > 5] = np.nan

    else:
        emissions_filtered_corrected = emissions_filtered
        emissions_mean = np.nanmean(emissions_filtered_corrected, axis=0)
        emissions_std = np.nanstd(emissions_filtered_corrected, axis=0)
        emissions_filtered_corrected = (emissions_filtered_corrected - emissions_mean) / emissions_std

    # truncate inputs to match emissions after filtering
    inputs = inputs[:emissions_filtered_corrected.shape[0], :]

    if upsample_factor != 1:
        emissions_out = np.empty((emissions_filtered_corrected.shape[0] * upsample_factor, emissions_filtered_corrected.shape[1]))
        emissions_out[:] = np.nan
        emissions_out[::upsample_factor, :] = emissions_filtered_corrected

        inputs_out = np.zeros((inputs.shape[0] * upsample_factor, inputs.shape[1]))
        inputs_out[::upsample_factor, :] = inputs
    else:
        emissions_out = emissions_filtered_corrected
        inputs_out = inputs

    return emissions_out, inputs_out


def load_data(data_path, num_data_sets=None, neuron_freq=0.0, held_out_data=[],
              hold_out='worm', upsample_factor=1):
    data_path = Path(data_path)

    preprocess_filename = 'funcon_preprocessed_data.pkl'
    emissions_train = []
    inputs_train = []
    cell_ids_train = []
    path_name = []
    sample_rate = 0.5 / upsample_factor  # seconds per sample DEFAULT

    # find all files in the folder that have francesco_green.npy
    for i in sorted(data_path.rglob('francesco_green.npy'))[::-1]:
        path_name.append(i.parts[-2])

        # check if a processed version exists
        preprocess_path = i.parent / preprocess_filename

        data_file = open(preprocess_path, 'rb')
        preprocessed_data = pickle.load(data_file)
        data_file.close()

        this_emissions = preprocessed_data['emissions']
        this_inputs = preprocessed_data['inputs']
        this_cell_ids = preprocessed_data['cell_ids']

        emissions_train.append(this_emissions)
        inputs_train.append(this_inputs)
        cell_ids_train.append(this_cell_ids)

    emissions_test = []
    inputs_test = []
    cell_ids_test = []

    if hold_out == 'worm':
        for i in reversed(range(len(emissions_train))):
            # skip any data that is being held out
            if path_name[i] in held_out_data:
                emissions_test.append(emissions_train.pop(i))
                inputs_test.append(inputs_train.pop(i))
                cell_ids_test.append(cell_ids_train.pop(i))

        emissions_test += emissions_train[num_data_sets:]
        inputs_test += inputs_train[num_data_sets:]
        cell_ids_test += cell_ids_train[num_data_sets:]

        emissions_test = emissions_test[:num_data_sets]
        inputs_test = inputs_test[:num_data_sets]
        cell_ids_test = cell_ids_test[:num_data_sets]

        emissions_train = emissions_train[:num_data_sets]
        inputs_train = inputs_train[:num_data_sets]
        cell_ids_train = cell_ids_train[:num_data_sets]

    elif hold_out == 'middle':
        frac = 0.3

        for i in range(num_data_sets):
            num_time = emissions_train[i].shape[0]
            test_inds = np.arange((1 - frac) * num_time / 2, (1 + frac) * num_time / 2, dtype=int)
            emissions_test.append(emissions_train[i][test_inds, :])
            inputs_test.append(inputs_train[i][test_inds, :])
            cell_ids_test.append(cell_ids_train[i])

            emissions_train[i][test_inds, :] = np.nan
            inputs_train[i][test_inds, :] = 0

        emissions_train = emissions_train[:num_data_sets]
        inputs_train = inputs_train[:num_data_sets]

    elif hold_out == 'end':
        frac = 0.3

        for i in range(num_data_sets):
            num_time = emissions_train[i].shape[0]
            start_ind = int(frac * num_time)
            emissions_test.append(emissions_train[i][start_ind:, :])
            inputs_test.append(inputs_train[i][start_ind:, :])
            cell_ids_test.append(cell_ids_train[i])

            emissions_train[i] = emissions_train[i][:start_ind, :]
            inputs_train[i] = inputs_train[i][:start_ind, :]

        emissions_train = emissions_train[:num_data_sets]
        inputs_train = inputs_train[:num_data_sets]

    else:
        raise Exception('Hold out style is not recognized')

    print('Size of data set:', len(emissions_train))

    # align the data sets so that each column corresponds to the same cell ID
    data_train = {}
    data_test = {}

    data_train['emissions'], data_train['inputs'], data_train['cell_ids'] = \
        align_data_cell_ids(emissions_train, inputs_train, cell_ids_train)

    data_test['emissions'], data_test['inputs'], data_test['cell_ids'] = \
        align_data_cell_ids(emissions_test, inputs_test, cell_ids_test, cell_ids_unique=data_train['cell_ids'])

    # eliminate neurons that don't show up often enough
    measured_neurons = np.stack([np.mean(np.isnan(i), axis=0) <= 0.5 for i in data_train['emissions']])
    measured_freq = np.mean(measured_neurons, axis=0)
    neurons_to_keep = measured_freq >= neuron_freq

    data_train['emissions'] = [i[:, neurons_to_keep] for i in data_train['emissions']]
    data_train['inputs'] = [i[:, neurons_to_keep] for i in data_train['inputs']]
    data_train['cell_ids'] = [data_train['cell_ids'][i] for i in range(len(data_train['cell_ids'])) if neurons_to_keep[i]]
    data_train['sample_rate'] = sample_rate

    data_test['emissions'] = [i[:, neurons_to_keep] for i in data_test['emissions']]
    data_test['inputs'] = [i[:, neurons_to_keep] for i in data_test['inputs']]
    data_test['cell_ids'] = [data_test['cell_ids'][i] for i in range(len(data_test['cell_ids'])) if neurons_to_keep[i]]
    data_test['sample_rate'] = sample_rate

    return data_train, data_test


def save_run(save_folder, model_trained=None, model_true=None, ep=None, **vars_to_save):
    save_folder = Path(save_folder)
    model_save_folder = save_folder / 'models'

    # save the trained model
    if model_trained is not None:
        if not model_save_folder.exists():
            os.mkdir(model_save_folder)

        if ep is not None:
            trained_model_save_path = model_save_folder / ('model_trained_' + str(ep) + '.pkl')
            model_trained.save(path=trained_model_save_path)

        trained_model_save_path = model_save_folder / 'model_trained.pkl'
        model_trained.save(path=trained_model_save_path)

    # save the true model, if it exists
    if model_true is not None:
        if not model_save_folder.exists():
            os.mkdir(model_save_folder)

        true_model_save_path = model_save_folder / 'model_true.pkl'
        model_true.save(path=true_model_save_path)

    for k, v in vars_to_save.items():
        save_path = save_folder / (k + '.pkl')

        save_file = open(save_path, 'wb')
        pickle.dump(v, save_file)
        save_file.close()


def align_data_cell_ids(emissions, inputs, cell_ids, cell_ids_unique=None):
    if cell_ids_unique is None:
        cell_ids_unique = list(np.unique(np.concatenate(cell_ids)))
        if '' in cell_ids_unique:
            cell_ids_unique.pop(cell_ids_unique.index(''))

    num_neurons = len(cell_ids_unique)

    emissions_aligned = []
    inputs_aligned = []

    # now update the neural data and fill in nans where we don't have a recording from a neuron
    for e, i, c in zip(emissions, inputs, cell_ids):
        this_emissions = np.empty((e.shape[0], num_neurons))
        this_emissions[:] = np.nan
        this_inputs = np.zeros((e.shape[0], num_neurons))

        # loop through all the labels from this data set
        for unique_cell_index, cell_name in enumerate(cell_ids_unique):
            # find the index of the full list of cell ids
            if cell_name in c and cell_name != '':
                unaligned_cell_index = c.index(cell_name)
                this_emissions[:, unique_cell_index] = e[:, unaligned_cell_index]
                this_inputs[:, unique_cell_index] = i[:, unaligned_cell_index]

        emissions_aligned.append(this_emissions)
        inputs_aligned.append(this_inputs)

    return emissions_aligned, inputs_aligned, cell_ids_unique




