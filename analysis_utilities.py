import numpy as np
import warnings
import wormneuroatlas as wa
import scipy
import pickle


def auto_select_ids(inputs, cell_ids, num_neurons=10):
    num_stim = np.sum(np.stack([np.sum(i, axis=0) for i in inputs]), axis=0)
    top_stims = np.argsort(num_stim)[-num_neurons:]
    cell_ids_chosen = [cell_ids[i] for i in top_stims]
    neuron_to_remove = cell_ids_chosen[-1]
    neuron_to_stim = neuron_to_remove

    return cell_ids_chosen, neuron_to_remove, neuron_to_stim


def p_norm(data, power=1, axis=None):
    return np.nanmean(np.abs(data)**power, axis=axis)**(1/power)


def ave_fun(data, axis=None):
    return np.sum(data, axis=axis)


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
    data_nan_conv[nan_loc[:data_filtered.shape[0]]] = np.nan

    return data_nan_conv


def stack_weights(weights, num_split, axis=-1):
    return np.stack(np.split(weights, num_split, axis=axis))


def get_impulse_response_function(data, inputs, window=(-60, 120), sub_pre_stim=False, return_pre=True):
    num_neurons = data[0].shape[1]

    responses = []
    for n in range(num_neurons):
        responses.append([])

    for e, i in zip(data, inputs):
        num_time = e.shape[0]
        stim_events = np.where(i == 1)

        for time, target in zip(stim_events[0], stim_events[1]):
            if window[0] + time >= 0 and window[1] + time < num_time:
                this_clip = e[window[0]+time:window[1]+time, :]

                if sub_pre_stim:
                    if window[0] < 0:
                        baseline = np.nanmean(this_clip[:-window[0], :], axis=0)
                        this_clip = this_clip - baseline

                if not return_pre:
                    if window[0] < 0:
                        this_clip = this_clip[-window[0]:, :]

                responses[target].append(this_clip)
                a=1

    for ri, r in enumerate(responses):
        if len(r) > 0:
            responses[ri] = np.stack(r)
        else:
            if return_pre:
                responses[ri] = np.zeros((0, window[1] - window[0], num_neurons))
            else:
                responses[ri] = np.zeros((0, window[1], num_neurons))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ave_responses = [np.nanmean(j, axis=0) for j in responses]
    ave_responses = np.stack(ave_responses)
    ave_responses = np.transpose(ave_responses, axes=(1, 2, 0))

    ave_responses_sem = [np.nanstd(j, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(j), axis=0)) for j in responses]
    ave_responses_sem = np.stack(ave_responses_sem)
    ave_responses_sem = np.transpose(ave_responses_sem, axes=(1, 2, 0))

    return ave_responses, ave_responses_sem, responses


def load_anatomical_data(cell_ids):
    # load in anatomical data
    chem_file = open('anatomical_data/chemical.pkl', 'rb')
    chemical_connectome_full = pickle.load(chem_file)
    chem_file.close()

    gap_file = open('anatomical_data/gap.pkl', 'rb')
    gap_junction_connectome_full = pickle.load(gap_file)
    gap_file.close()

    peptide_file = open('anatomical_data/peptide.pkl', 'rb')
    peptide_connectome_full = pickle.load(peptide_file)
    peptide_file.close()

    ids_file = open('anatomical_data/cell_ids.pkl', 'rb')
    atlas_ids = pickle.load(ids_file)
    ids_file.close()

    atlas_inds = [atlas_ids.index(i) for i in cell_ids]
    chem_syn_conn = chemical_connectome_full[np.ix_(atlas_inds, atlas_inds)]
    gap_conn = gap_junction_connectome_full[np.ix_(atlas_inds, atlas_inds)]
    pep_conn = peptide_connectome_full[np.ix_(atlas_inds, atlas_inds)]

    return chem_syn_conn, gap_conn, pep_conn


def get_anatomical_data(cell_ids):
    # load in anatomical data
    watlas = wa.NeuroAtlas()
    atlas_ids = list(watlas.neuron_ids)
    anatomical_connectome_full = watlas.get_chemical_synapses()
    gap_junction_connectome_full = watlas.get_gap_junctions()
    peptide_connectome_full = watlas.get_peptidergic_connectome()
    atlas_ids[atlas_ids.index('AWCON')] = 'AWCR'
    atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCL'
    atlas_inds = [atlas_ids.index(i) for i in cell_ids]
    chem_syn_conn = anatomical_connectome_full[np.ix_(atlas_inds, atlas_inds)]
    gap_conn = gap_junction_connectome_full[np.ix_(atlas_inds, atlas_inds)]
    pep_conn = peptide_connectome_full[np.ix_(atlas_inds, atlas_inds)]

    return chem_syn_conn, gap_conn, pep_conn


def compare_matrix_sets(left_side, right_side):
    if type(left_side) is not list:
        left_side = [left_side]

    if type(right_side) is not list:
        right_side = [right_side]

    # find the best linear combination to make the sum of the matricies on the left side equal to the ones on the right
    num_left = len(left_side)
    num_right = len(right_side)
    left_side_col = np.stack([i.reshape(-1) for i in left_side]).T
    right_side_col = np.stack([i.reshape(-1) for i in right_side]).T

    if num_left > 1 or num_right > 1:
        x0 = np.ones(num_left + num_right - 2)

        def obj_fun(x):
            left_weights = np.concatenate((np.ones(1), x[:num_left-1]), axis=0)
            right_weights = np.concatenate((np.ones(1), x[num_left-1:]), axis=0)

            left_val = left_side_col @ left_weights
            right_val = right_side_col @ right_weights

            return -nan_corr(left_val, right_val)[0]

        x_hat = scipy.optimize.minimize(obj_fun, x0).x

        left_weights_hat = np.concatenate((np.ones(1), x_hat[:num_left - 1]), axis=0)
        right_weights_hat = np.concatenate((np.ones(1), x_hat[num_left - 1:]), axis=0)
    else:
        left_weights_hat = np.ones(1)
        right_weights_hat = np.ones(1)

    left_recon = (left_side_col @ left_weights_hat).reshape(left_side[0].shape)
    right_recon = (right_side_col @ right_weights_hat).reshape(right_side[0].shape)

    score, score_ci = nan_corr(left_recon, right_recon)

    return score, score_ci, left_recon, right_recon


def find_stim_events(inputs, window_size=1000):
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
        this_max_val = np.max(total_stim)
        this_max_ind = np.argmax(total_stim)

        if this_max_val > max_val:
            max_val = this_max_val
            max_ind = this_max_ind
            max_data_set = ii
            max_window = this_window_size

    time_window = (max_ind, max_ind + max_window)

    return max_data_set, time_window


def nan_r2(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot


def nan_corr(y_true, y_hat, mean_sub=True):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    if mean_sub:
        y_true = y_true - np.mean(y_true)
        y_hat = y_hat - np.mean(y_hat)

    y_true_std = np.std(y_true, ddof=1)
    y_hat_std = np.std(y_hat, ddof=1)

    corr = np.mean(y_true * y_hat) / y_true_std / y_hat_std

    # now estimate the confidence intervals for the correlation
    alpha = 0.5
    n = y_true.shape[0]
    z_a = scipy.stats.norm.ppf(1 - alpha / 2)
    z_r = np.log((1 + corr) / (1 - corr)) / 2
    l = z_r - (z_a / np.sqrt(n - 3))
    u = z_r + (z_a / np.sqrt(n - 3))
    ci_l = (np.exp(2 * l) - 1) / (np.exp(2 * l) + 1)
    ci_u = (np.exp(2 * u) - 1) / (np.exp(2 * u) + 1)
    ci = (ci_l, ci_u)

    return corr, ci


def nan_corr_data(data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        # calculate the average cross correlation between neurons
        emissions_cov = []
        num_neurons = data[0].shape[1]
        for i in range(len(data)):
            emissions_this = data[i]
            nan_loc = np.isnan(emissions_this)
            em_z_score = (emissions_this - np.nanmean(emissions_this, axis=0)) / np.nanstd(emissions_this, ddof=1, axis=0)
            em_z_score[nan_loc] = 0

            # figure out how many times the two neurons were measured together
            num_measured = np.zeros((num_neurons, num_neurons))
            for j1 in range(num_neurons):
                for j2 in range(num_neurons):
                    num_measured[j1, j2] = np.sum(~nan_loc[:, j1] & ~nan_loc[:, j2])

            emissions_cov_this = em_z_score.T @ em_z_score / num_measured
            emissions_cov.append(emissions_cov_this)

        correlation = np.nanmean(np.stack(emissions_cov), axis=0)

    return correlation


def nancorrcoef(data):
    num_data = len(data)
    corr_coef = np.zeros((num_data, num_data))

    for ii, i in enumerate(data):
        for ji, j in enumerate(data):
            # columnize everything in data
            i = i.reshape(-1)
            j = j.reshape(-1)

            i = (i - np.nanmean(i, axis=0)) / np.nanstd(i, ddof=1)
            j = (j - np.nanmean(j, axis=0)) / np.nanstd(j, ddof=1)

            corr_coef[ii, ji] = np.nanmean(i * j)

    return corr_coef


