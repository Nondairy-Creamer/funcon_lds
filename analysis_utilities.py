import numpy as np
import wormneuroatlas as wa
import pickle
from pathlib import Path
import metrics as met


def auto_select_ids(inputs, cell_ids, num_neurons=10):
    num_stim = np.sum(np.stack([np.sum(i, axis=0) for i in inputs]), axis=0)
    top_stims = np.argsort(num_stim)[-num_neurons:]
    cell_ids_chosen = [cell_ids[i] for i in top_stims]

    return cell_ids_chosen


def nan_argsort(data):
    sorted_inds = np.argsort(data)
    sorted_inds = sorted_inds[~np.isnan(data[sorted_inds])]
    return sorted_inds


def p_norm(data, power=1, axis=None):
    return np.nanmean(np.abs(data)**power, axis=axis)**(1/power)


def ave_fun(data, axis=None):
    return np.nanmean(data, axis=axis)


def nan_convolve(data, filter, mode='valid'):
    # attempt to ignore nans during a convolution
    # this isn't particularly principled, will just replace nans with 0s and divide the convolution
    # by the fraction of data that was in the window
    # only makes sense for nonnegative filters

    if np.any(filter < 0):
        raise Exception('nan_filter can only handle nonnegative filters')

    nan_loc = np.isnan(data)
    data_no_nan = data
    data_no_nan[nan_loc] = 0
    data_filtered = np.convolve(data_no_nan, filter, mode=mode)
    nan_count = np.convolve(~nan_loc, filter / np.sum(filter), mode=mode)
    nan_count[nan_count == 0] = 1
    data_nan_conv = data_filtered / nan_count

    nan_loc_pad = np.zeros(filter.size - 1) == 0
    nan_loc = np.concatenate((nan_loc, nan_loc_pad))
    nan_loc = nan_loc[:data_filtered.shape[0]]
    data_nan_conv[nan_loc] = np.nan

    return data_nan_conv


def stack_weights(weights, num_split, axis=-1):
    return np.stack(np.split(weights, num_split, axis=axis))


def load_anatomical_data(cell_ids=None):
    # load in anatomical data
    chem_path = Path('anatomical_data/chemical.pkl')
    if not chem_path.exists():
        chem_path = Path('../') / chem_path
    chem_file = open(chem_path, 'rb')
    chemical_synapse_connectome = pickle.load(chem_file)
    chem_file.close()

    gap_path = Path('anatomical_data/gap.pkl')
    if not gap_path.exists():
        gap_path = Path('../') / gap_path
    gap_file = open(gap_path, 'rb')
    gap_junction_connectome = pickle.load(gap_file)
    gap_file.close()

    peptide_path = Path('anatomical_data/peptide.pkl')
    if not peptide_path.exists():
        peptide_path = Path('../') / peptide_path
    peptide_file = open(peptide_path, 'rb')
    peptide_connectome = pickle.load(peptide_file)
    peptide_file.close()

    ids_path = Path('anatomical_data/cell_ids.pkl')
    if not ids_path.exists():
        ids_path = Path('../') / ids_path
    ids_file = open(ids_path, 'rb')
    atlas_ids = pickle.load(ids_file)
    ids_file.close()

    if cell_ids is not None:
        if '0' in cell_ids:
            # if the data is synthetic just choose the first n neurons for testing
            atlas_inds = np.arange(len(cell_ids))
        else:
            atlas_inds = [atlas_ids.index(i) for i in cell_ids]

        chemical_synapse_connectome = chemical_synapse_connectome[np.ix_(atlas_inds, atlas_inds)]
        gap_junction_connectome = gap_junction_connectome[np.ix_(atlas_inds, atlas_inds)]
        peptide_connectome = peptide_connectome[np.ix_(atlas_inds, atlas_inds)]

    anatomy_dict = {'chem_conn': chemical_synapse_connectome,
                    'gap_conn': gap_junction_connectome,
                    'pep_conn': peptide_connectome}

    return anatomy_dict


def get_anatomical_data(cell_ids):
    # load in anatomical data
    watlas = wa.NeuroAtlas()
    atlas_ids = list(watlas.neuron_ids)
    chemical_connectome_full = watlas.get_chemical_synapses()
    gap_junction_connectome_full = watlas.get_gap_junctions()
    peptide_connectome_full = watlas.get_peptidergic_connectome()
    atlas_ids[atlas_ids.index('AWCON')] = 'AWCR'
    atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCL'
    atlas_inds = [atlas_ids.index(i) for i in cell_ids]
    chem_conn = chemical_connectome_full[np.ix_(atlas_inds, atlas_inds)]
    gap_conn = gap_junction_connectome_full[np.ix_(atlas_inds, atlas_inds)]
    pep_conn = peptide_connectome_full[np.ix_(atlas_inds, atlas_inds)]

    return chem_conn, gap_conn, pep_conn


def interleave(a, b):
    c = np.empty(a.size + b.size, dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b

    return c


def find_stim_events(inputs, emissions=None, chosen_neuron_ind=None, window_size=1000):
    max_data_set = 0
    max_ind = 0
    max_val = 0
    max_window = 0

    for ii, i in enumerate(inputs):
        # some data sets might be smaller than window size
        this_window_size = np.min((window_size, i.shape[0]))

        # we're going to pass a square filter over the data to find the locations with the most stimulation events
        t_filt = np.ones(this_window_size)
        inputs_filtered = np.zeros((i.shape[0] - this_window_size + 1, i.shape[1]))

        for n in range(i.shape[1]):
            inputs_filtered[:, n] = np.convolve(i[:, n], t_filt, mode='valid')

        # sum the filtered inputs over neurons
        total_stim = inputs_filtered.sum(1)

        if chosen_neuron_ind is not None:
            recording_has_neuron = np.mean(np.isnan(emissions[ii][:, chosen_neuron_ind]), axis=0) < 0.3
            total_stim = total_stim * recording_has_neuron

        this_max_val = np.max(total_stim)
        this_max_ind = np.argmax(total_stim)

        if (ii == 0) or (this_max_val > max_val):
            max_val = this_max_val
            max_ind = this_max_ind
            max_data_set = ii
            max_window = this_window_size

    time_window = (max_ind, max_ind + max_window)

    return max_data_set, time_window


def normalize_model(model, posterior=None, init_mean=None, init_cov=None):
    c_sum = model.emissions_weights.sum(1)
    c_sum_stacked = np.tile(c_sum, model.dynamics_lags)
    h = np.diag(c_sum_stacked)
    h_inv = np.diag(1 / c_sum_stacked)

    model.dynamics_weights = h @ model.dynamics_weights @ h_inv
    model.dynamics_input_weights = h @ model.dynamics_input_weights
    model.dynamics_cov = h @ model.dynamics_cov @ h.T

    model.emissions_weights = model.emissions_weights @ h_inv

    if posterior is not None:
        posterior = [i @ h[:model.dynamics_dim, :model.dynamics_dim].T for i in posterior]

    if init_mean is not None:
        init_mean = [h @ i for i in init_mean]

    if init_cov is not None:
        init_cov = [h @ i @ h.T for i in init_cov]

    return model, posterior, init_mean, init_cov


def nan_corr_data(data, alpha=0.05):
    data_cat = np.concatenate(data, axis=0)
    data_corr = np.zeros((data_cat.shape[1], data_cat.shape[1]))
    data_corr_ci = np.zeros((2, data_cat.shape[1], data_cat.shape[1]))

    for i in range(data_cat.shape[1]):
        for j in range(data_cat.shape[1]):
            data_corr[i, j], data_corr_ci[:, i, j] = met.nan_corr(data_cat[:, i], data_cat[:, j], alpha=alpha)

        print(i+1, '/', data_cat.shape[1], 'neurons correlated')

    return data_corr, data_corr_ci

