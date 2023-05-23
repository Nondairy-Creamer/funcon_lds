import numpy as np
import itertools as it
import scipy.special as ss
from pathlib import Path
import yaml
import os
import pickle


def get_params(param_name='params'):
    with open('.' + param_name + '.yml', 'r') as file:
        params = yaml.safe_load(file)

    if Path(param_name + '_update.yml').exists():
        with open(param_name + '_update.yml', 'r') as file:
            params_update = yaml.safe_load(file)

        params.update(params_update)

    return params


def get_model_data(data_path, num_data_sets=None, bad_data_sets=[], frac_neuron_coverage=0.0, minimum_frac_measured=0.0,
                   start_index=0):
    # load all the recordings
    emissions_unaligned, cell_ids_unaligned, q, q_labels, stim_cell_ids, inputs_unaligned = \
        load_data(data_path)

    # remove recordings that are noisy
    data_sets_to_remove = np.sort(bad_data_sets)[::-1]
    for bd in data_sets_to_remove:
        emissions_unaligned.pop(bd)
        cell_ids_unaligned.pop(bd)
        inputs_unaligned.pop(bd)
        stim_cell_ids.pop(bd)

    if num_data_sets is None:
        num_data_sets = len(emissions_unaligned)

    emissions_unaligned = emissions_unaligned[:num_data_sets]
    cell_ids_unaligned = cell_ids_unaligned[:num_data_sets]
    inputs_unaligned = inputs_unaligned[:num_data_sets]
    stim_cell_ids = stim_cell_ids[:num_data_sets]

    # choose a subset of the data sets to maximize the number of recordings * the number of neurons included
    cell_ids, emissions, best_runs, inputs = \
        get_combined_dataset(emissions_unaligned, cell_ids_unaligned, stim_cell_ids, inputs_unaligned,
                             frac_neuron_coverage=frac_neuron_coverage,
                             minimum_freq=minimum_frac_measured)

    num_data = len(emissions)

    # remove the beginning of the recording which contains artifacts and mean subtract
    for ri in range(num_data):
        emissions[ri] = emissions[ri][start_index:, :]
        emissions[ri] = emissions[ri] - np.mean(emissions[ri], axis=0, keepdims=True)
        inputs[ri] = inputs[ri][start_index:, :]

    return emissions, inputs, cell_ids


def load_data(fun_atlas_path):
    fun_atlas_path = Path(fun_atlas_path)

    recordings = []
    labels = []
    label_indicies = []
    stim_cell_inds = []
    stim_volume_inds = []
    stim_ids = []

    for i in fun_atlas_path.rglob('francesco_green.npy'):
        recordings.append(np.load(str(i), allow_pickle=False))
        labels.append(np.load(str(i.parent / 'labels.npy'), allow_pickle=False))
        # currently don't know how francesco labels his bit connectivity matrix. reconstruct it with the indicies he
        # provided that map labels to the connectivity matrix
        label_indicies.append(np.load(str(i.parent / 'label_indicies.npy'), allow_pickle=False))

        nan_neurons = np.all(np.isnan(recordings[-1]), axis=0) | (np.std(recordings[-1], axis=0) == 0)
        # print(labels[-1].shape)
        # print(labels[-1].shape[0] - recordings[-1].shape[1])
        labels[-1] = list(labels[-1][~nan_neurons])
        recordings[-1] = recordings[-1][:, ~nan_neurons]

        # tau = 2
        # t = np.arange(0, tau * 3, 0.5)
        # filt = np.exp(-t / tau)
        #
        # for c in range(recordings[-1].shape[1]):
        #     recordings[-1][:, c] = np.convolve(recordings[-1][:, c], filt, mode='full')[:recordings[-1].shape[0]]

        neuron_std = np.std(recordings[-1], axis=0)
        recordings[-1] = (recordings[-1] - np.mean(recordings[-1], axis=0)) / neuron_std
        # recordings[-1] = recordings[-1] / np.mean(recordings[-1], axis=0) - 1

        # load stimulation data
        stim_atlas_inds = np.load(str(i.parent / 'stim_atlas_inds.npy'), allow_pickle=True)
        stim_ids.append(np.load(str(i.parent / 'stim_ids.npy'), allow_pickle=True))
        stim_cell_inds.append(np.load(str(i.parent / 'stim_recording_cell_inds.npy'), allow_pickle=True))
        stim_volume_inds.append(np.load(str(i.parent / 'stim_volumes_inds.npy'), allow_pickle=True))

    # fun con matricies are [i, j] where j is stimulated and i is responding
    dff = np.load(str(fun_atlas_path / 'dFF.npy'))
    occ1 = np.load(str(fun_atlas_path / 'occ1.npy'))
    occ3 = np.load(str(fun_atlas_path / 'occ3.npy'))
    q = np.load(str(fun_atlas_path / 'q.npy'))
    tost_q = np.load(str(fun_atlas_path / 'tost_q.npy'))

    dff_all = np.load(str(fun_atlas_path / 'dFF_all.npy'), allow_pickle=True)
    occ2 = np.load(str(fun_atlas_path / 'occ2.npy'), allow_pickle=True)
    resp_traces = np.load(str(fun_atlas_path / 'resp_traces.npz'), allow_pickle=True)

    q_labels = np.array([''] * q.shape[0], dtype=object)
    for ri in range(len(recordings)):
        if len(labels[ri]) == len(label_indicies[ri]):
            for li, l in enumerate(label_indicies[ri]):
                q_labels[l] = labels[ri][li]

    return recordings, labels, q, q_labels, stim_ids, stim_volume_inds


def save_run(model_save_folder, model_trained, model_true=None, data=None, run_params=None):
    if 'SLURM_JOB_ID' in os.environ:
        slurm_tag = '_' + os.environ['SLURM_JOB_ID']
    else:
        slurm_tag = 'local'

    full_save_folder = Path(model_save_folder) / slurm_tag
    true_model_save_path = full_save_folder / 'model_true.pkl'
    trained_model_save_path = full_save_folder / 'model_trained.pkl'
    data_save_path = full_save_folder / 'data.pkl'
    params_save_path = full_save_folder / 'params.pkl'

    if not full_save_folder.exists():
        os.mkdir(full_save_folder)

    # save the trained model
    model_trained.save(path=trained_model_save_path)

    # save the true model, if it exists
    if model_true is not None:
        model_true.save(path=true_model_save_path)
    else:
        # if there is an old "true" model delete it because it doesn't correspond to this trained model
        if os.path.exists(true_model_save_path):
            os.remove(true_model_save_path)

    # save the data
    if data is not None:
        data_file = open(data_save_path, 'wb')
        pickle.dump(data, data_file)
        data_file.close()

    # save the input parameters
    if run_params is not None:
        params_file = open(params_save_path, 'wb')
        pickle.dump(run_params, params_file)
        params_file.close()


def get_combined_dataset(neural_data, neuron_labels, stim_ids, stim_volume_inds,
                         frac_neuron_coverage=0.75, num_rep=1e6, minimum_freq=0.5):

    neural_labels_unique, cell_label_count = np.unique(np.concatenate(neuron_labels), return_counts=True)

    if neural_labels_unique[0] == '':
        cell_label_count = cell_label_count[1:]
        neural_labels_unique = neural_labels_unique[1:]

    cell_label_freq = cell_label_count / len(neuron_labels)
    neural_labels_unique = neural_labels_unique[cell_label_freq >= minimum_freq]

    num_neurons = len(neural_labels_unique)
    num_datasets = len(neural_data)
    neural_data_labeled = []
    stim_mat = []
    runs_by_neurons = np.zeros((num_datasets, num_neurons))

    # now update the neural data and fill in nans where we don't have a recording from a neuron
    for wi, w in enumerate(neural_data):
        nan_mat = np.empty([w.shape[0], num_neurons])
        this_stim_mat = np.zeros((w.shape[0], num_neurons))
        nan_mat[:] = np.nan
        neural_data_labeled.append(nan_mat)

        for ci, c in enumerate(neural_labels_unique):
            # fill in where this neuron was stimulated
            for si, s in enumerate(stim_ids[wi]):
                if s == c:
                    this_stim_index = stim_volume_inds[wi][si]
                    this_stim_mat[this_stim_index, ci] = 1

            # find the trace associated with this cell_id
            this_trace = None
            for li, l in enumerate(neuron_labels[wi]):
                if c == l:
                    this_trace = li

            if this_trace is not None:
                neural_data_labeled[wi][:, ci] = w[:, this_trace]
                runs_by_neurons[wi, ci] = 1

        stim_mat.append(this_stim_mat)

    if frac_neuron_coverage > 0:
        # choose which dataset to pass on
        max_val, best_runs, best_neurons = max_set(runs_by_neurons, cell_id_frac_cutoff=frac_neuron_coverage, num_rep=num_rep)

        neural_data_selected = [neural_data_labeled[i][:, best_neurons] for i in best_runs]
        stim_mat_selected = [stim_mat[i][:, best_neurons] for i in best_runs]
        neural_labels_selected = neural_labels_unique[best_neurons]
    else:
        best_runs = list(np.arange(len(neural_data)))
        neural_data_selected = neural_data_labeled
        neural_labels_selected = neural_labels_unique
        stim_mat_selected = stim_mat

    return neural_labels_selected, neural_data_selected, best_runs, stim_mat_selected


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



