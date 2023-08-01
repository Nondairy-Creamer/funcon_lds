import numpy as np
import itertools as it
import scipy.special as ss
from pathlib import Path
import yaml
import os
import pickle
import tmac.preprocessing as tp
import time
import analysis_utilities as au
import warnings

# utilities for loading and saving the data


def get_run_params(param_name):
    # load in the parameters for the run which dictate how many data sets to use,
    # or how many time lags the model will fit etc

    with open(param_name, 'r') as file:
        params = yaml.safe_load(file)

    return params


def preprocess_data(emissions, inputs, start_index=0, correct_photobleach=False):
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

    # filter out noise at the nyquist frequency
    filter_size = 2
    filter_shape = np.ones(filter_size) / filter_size
    emissions_filtered = np.zeros((emissions.shape[0] - filter_size + 1, emissions.shape[1]))

    for c in range(emissions.shape[1]):
        emissions_filtered[:, c] = au.nan_convolve(emissions[:, c].copy(), filter_shape)

    if correct_photobleach:
        # photobleach correction
        emissions_filtered_corrected = np.zeros_like(emissions_filtered)
        for c in range(emissions_filtered.shape[1]):
            emissions_filtered_corrected[:, c] = tp.photobleach_correction(emissions_filtered[:, c], num_exp=2)[:, 0]

        # occasionally the fit fails check for outputs who don't have a mean close to 1
        # fit those with a single exponential
        # all the nan mean calls throw warnings when averaging over nans so supress those
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bad_fits_2exp = np.where(np.abs(np.nanmean(emissions_filtered_corrected, axis=0) - 1) > 0.1)[0]

            for bf in bad_fits_2exp:
                emissions_filtered_corrected[:, bf] = tp.photobleach_correction(emissions_filtered[:, bf], num_exp=1)[:, 0]

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

    return emissions_filtered_corrected, inputs


def load_and_align_data(data_path, force_preprocess=False, num_data_sets=None, start_index=0,
                        correct_photobleach=False, interpolate_nans=False, held_out_data=[]):
    # load all the recordings of neural activity
    data_train_unaligned, data_test_unaligned = \
        load_and_preprocess_data(data_path, num_data_sets=num_data_sets,
                                 force_preprocess=force_preprocess, start_index=start_index,
                                 correct_photobleach=correct_photobleach, interpolate_nans=interpolate_nans,
                                 held_out_data=held_out_data)

    # choose a subset of the data sets to maximize the number of recordings * the number of neurons included
    data_train = {}
    data_test = {}
    data_train['emissions'], data_train['inputs'], data_train['cell_ids'] = get_combined_dataset(data_train_unaligned['emissions'], data_train_unaligned['inputs'], data_train_unaligned['cell_ids'])

    # generate the test dataset so that it matches the train dataset
    num_neurons = len(data_train['cell_ids'])

    data_test['emissions'] = []
    data_test['inputs'] = []
    data_test['cell_ids'] = []

    for d in range(len(data_test_unaligned['emissions'])):
        this_emissions = data_test_unaligned['emissions'][d]
        this_inputs = data_test_unaligned['inputs'][d]
        this_cell_ids = data_test_unaligned['cell_ids'][d]

        num_time = this_emissions.shape[0]
        mapped_emissions = np.empty((num_time, num_neurons))
        mapped_emissions[:] = np.nan
        mapped_inputs = np.zeros((num_time, num_neurons))

        for ci, c in enumerate(data_train['cell_ids']):
            if c in this_cell_ids:
                cell_id_loc = this_cell_ids.index(c)
                mapped_emissions[:, ci] = this_emissions[:, cell_id_loc]
                mapped_inputs[:, ci] = this_inputs[:, cell_id_loc]

        data_test['emissions'].append(mapped_emissions)
        data_test['inputs'].append(mapped_inputs)
        data_test['cell_ids'] = data_train['cell_ids']

    return data_train, data_test


def load_and_preprocess_data(fun_atlas_path, num_data_sets=None, force_preprocess=False, start_index=0,
                             correct_photobleach=False, interpolate_nans=True, held_out_data=[]):
    fun_atlas_path = Path(fun_atlas_path)

    preprocess_filename = 'funcon_preprocessed_data.pkl'
    emissions_train = []
    inputs_train = []
    cell_ids_train = []
    path_name = []

    # find all files in the folder that have francesco_green.npy
    for i in sorted(fun_atlas_path.rglob('francesco_green.npy'))[::-1]:
        path_name.append(i.parts[-2])

        # check if a processed version exists
        preprocess_path = i.parent / preprocess_filename

        if not force_preprocess and preprocess_path.exists():
            preprocessed_data = np.load(str(preprocess_path), allow_pickle=True)
            this_emissions = preprocessed_data['emissions']
            this_inputs = preprocessed_data['inputs']
            this_cell_ids = preprocessed_data['cell_ids']

        else:
            this_emissions = np.load(str(i))

            if not interpolate_nans:
                this_nan_mask = np.load(str(i.parent / 'nan_mask.npy'))
                this_emissions[this_nan_mask] = np.nan

            this_cell_ids = list(np.load(str(i.parent / 'labels.npy')))

            # load stimulation data
            this_stim_cell_ids = np.load(str(i.parent / 'stim_recording_cell_inds.npy'), allow_pickle=True)
            this_stim_volume_inds = np.load(str(i.parent / 'stim_volumes_inds.npy'), allow_pickle=True)

            this_inputs = np.zeros_like(this_emissions)
            this_stim_volume_inds = this_stim_volume_inds[this_stim_cell_ids != -2]
            this_stim_cell_ids = this_stim_cell_ids[this_stim_cell_ids != -2]
            this_inputs[this_stim_volume_inds, this_stim_cell_ids] = 1

            start = time.time()
            this_emissions, this_inputs = preprocess_data(this_emissions, this_inputs, start_index=start_index,
                                                          correct_photobleach=correct_photobleach)

            if interpolate_nans:
                full_nan_loc = np.all(np.isnan(this_emissions), axis=0)
                interp_emissions = tp.interpolate_over_nans(this_emissions[:, ~full_nan_loc])[0]
                this_emissions[:, ~full_nan_loc] = interp_emissions

            preprocessed_file = open(preprocess_path, 'wb')
            pickle.dump({'emissions': this_emissions, 'inputs': this_inputs, 'cell_ids': this_cell_ids}, preprocessed_file)
            preprocessed_file.close()

            print('Data set', i.parent, 'preprocessed')
            print('Took', time.time() - start, 's')

        emissions_train.append(this_emissions)
        inputs_train.append(this_inputs)
        cell_ids_train.append(this_cell_ids)

    emissions_test = []
    inputs_test = []
    cell_ids_test = []

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

    print('Size of data set:', len(emissions_train))

    data_train = {'emissions': emissions_train,
                  'inputs': inputs_train,
                  'cell_ids': cell_ids_train,
                  }

    data_test = {'emissions': emissions_test,
                 'inputs': inputs_test,
                 'cell_ids': cell_ids_test,
                 }

    return data_train, data_test


def save_run(model_save_folder, model_trained, model_true=None, **vars_to_save):
    model_save_folder = Path(model_save_folder)

    # save the trained model
    trained_model_save_path = model_save_folder / 'model_trained.pkl'
    model_trained.save(path=trained_model_save_path)

    # save the true model, if it exists
    if model_true is not None:
        true_model_save_path = model_save_folder / 'model_true.pkl'
        model_true.save(path=true_model_save_path)

    for k, v in vars_to_save.items():
        save_path = model_save_folder / (k + '.pkl')

        save_file = open(save_path, 'wb')
        pickle.dump(v, save_file)
        save_file.close()


def get_combined_dataset(emissions, inputs, cell_ids):
    cell_ids_unique = list(np.unique(np.concatenate(cell_ids)))
    if cell_ids_unique[0] == '':
        cell_ids_unique = cell_ids_unique[1:]

    num_neurons = len(cell_ids_unique)

    emissions_aligned = []
    inputs_aligned = []

    # now update the neural data and fill in nans where we don't have a recording from a neuron
    for ei, e in enumerate(emissions):
        this_emissions = np.empty((e.shape[0], num_neurons))
        this_emissions[:] = np.nan
        this_inputs = np.zeros((e.shape[0], num_neurons))

        # loop through all the labels from this data set
        for ci, c in enumerate(cell_ids[ei]):
            # find the index of the full list of cell ids
            if c != '':
                this_cell_ind = cell_ids_unique.index(c)
                this_emissions[:, this_cell_ind] = e[:, ci]
                this_inputs[:, this_cell_ind] = inputs[ei][:, ci]

        emissions_aligned.append(this_emissions)
        inputs_aligned.append(this_inputs)

    return emissions_aligned, inputs_aligned, cell_ids_unique


def max_set(a, cell_id_frac_cutoff=1.0, num_rep=1e6) -> (np.ndarray, int, float):
    """ Finds the largest sub matrix of a that is all ones
    :param a: matrix with entires of 1 or 0
    :param num_rep: number of sub matricies to test
    :param cell_id_frac_cutoff: The fraction of datasets which must contain the neuron for it to be included. If 1,
    then every one of the subset of datasets returned will contain every neuron.
    :return: the row and column indicies of the sub matrix
    """

    num_rep = int(num_rep)
    max_val = 0
    best_rows = []
    best_cols = []
    rng = np.random.default_rng()
    rows_list = np.arange(a.shape[0])

    # loop through the number of rows
    for r in range(1, a.shape[0] + 1):
        print('trying number of rows = ' + str(r+1) + '/' + str(a.shape[0]))
        total_comb = ss.comb(a.shape[0], r)

        # if num_rep is high enough, test all possible sets
        if total_comb < num_rep*10:
            all_possible = list(it.combinations(np.arange(a.shape[0]), r))

            for rows_to_test in all_possible:
                score, cols_to_test = get_matrix_best_score(a[rows_to_test, :], cell_id_frac_cutoff)

                if score > max_val:
                    max_val = score
                    best_rows = rows_to_test
                    best_cols = cols_to_test

        # if num_rep isn't high enough, randomly sample the space and return the best you can find.
        else:
            for n in range(num_rep):
                rows_to_test = rng.choice(rows_list, size=r, replace=False)

                score, cols_to_test = get_matrix_best_score(a[rows_to_test, :], cell_id_frac_cutoff)

                if score > max_val:
                    max_val = score
                    best_rows = rows_to_test
                    best_cols = cols_to_test

    return max_val, best_rows, best_cols


def get_matrix_best_score(a, coverage) -> (np.ndarray, float):
    cols_to_test = np.arange(a.shape[1])
    cols_to_test = cols_to_test[np.mean(a, axis=0) >= coverage]

    score = a.shape[0] * len(cols_to_test)
    return score, cols_to_test



