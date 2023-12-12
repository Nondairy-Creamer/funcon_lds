import numpy as np
import warnings
import wormneuroatlas as wa
import scipy
import pickle
from pathlib import Path


def auto_select_ids(inputs, cell_ids, num_neurons=10):
    num_stim = np.sum(np.stack([np.sum(i, axis=0) for i in inputs]), axis=0)
    top_stims = np.argsort(num_stim)[-num_neurons:]
    cell_ids_chosen = [cell_ids[i] for i in top_stims]

    return cell_ids_chosen


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
    data_nan_conv[nan_loc[:data_filtered.shape[0]]] = np.nan

    return data_nan_conv


def stack_weights(weights, num_split, axis=-1):
    return np.stack(np.split(weights, num_split, axis=axis))


def get_impulse_response_function(data, inputs, window=(30, 60), sub_pre_stim=True, return_pre=True):
    num_neurons = data[0].shape[1]

    responses = []
    for n in range(num_neurons):
        responses.append([])

    for e, i in zip(data, inputs):
        num_time = e.shape[0]
        stim_events = np.where(i == 1)

        for time, target in zip(stim_events[0], stim_events[1]):
            if time - window[0] >= 0 and window[1] + time < num_time:
                this_clip = e[time-window[0]:time+window[1], :]

                if sub_pre_stim:
                    baseline = np.nanmean(this_clip[:window[0], :], axis=0)
                    this_clip = this_clip - baseline

                if not return_pre:
                    this_clip = this_clip[window[0]:, :]

                responses[target].append(this_clip)

    for ri, r in enumerate(responses):
        if len(r) > 0:
            responses[ri] = np.stack(r)
        else:
            if return_pre:
                responses[ri] = np.zeros((0, np.sum(window), num_neurons))
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


def get_impulse_response_magnitude(data, inputs, window=(-60, 120), sub_pre_stim=True):
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

                if window[0] < 0:
                    this_clip = this_clip[-window[0]:, :]

                responses[target].append(np.nanmean(this_clip, axis=0))

    for ri, r in enumerate(responses):
        if len(r) > 0:
            responses[ri] = np.stack(r)
        else:
            responses[ri] = np.zeros((0, num_neurons))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ave_responses = [np.nanmean(j, axis=0) for j in responses]

    ave_responses = np.stack(ave_responses)
    ave_responses = ave_responses.T

    ave_responses_sem = [np.nanstd(j, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(j), axis=0)) for j in responses]
    ave_responses_sem = np.stack(ave_responses_sem)
    ave_responses_sem = ave_responses_sem.T

    return ave_responses, ave_responses_sem, responses


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


def compare_matrix_sets(left_side, right_side, positive_weights=False):
    if type(left_side) is not list:
        left_side = [left_side]

    if type(right_side) is not list:
        right_side = [right_side]

    # find the best linear combination to make the sum of the matricies on the left side equal to the ones on the right
    num_left = len(left_side)
    num_right = len(right_side)
    left_side_col = np.stack([i.reshape(-1) for i in left_side]).T
    right_side_col = np.stack([i.reshape(-1) for i in right_side]).T

    # if num_left > 1 or num_right > 1:
    x0 = np.zeros(num_left + num_right - 1) + (not positive_weights)

    def obj_fun(x):
        if positive_weights:
            x = np.exp(x)
        left_weights = np.concatenate((np.ones(1), x[:num_left-1]), axis=0)
        right_weights = x[num_left-1:]

        left_val = left_side_col @ left_weights
        right_val = right_side_col @ right_weights

        return -nan_corr(left_val, right_val)[0]

    x_hat = scipy.optimize.minimize(obj_fun, x0).x

    if positive_weights:
        x_hat = np.exp(x_hat)

    left_weights_hat = np.concatenate((np.ones(1), x_hat[:num_left - 1]), axis=0)
    right_weights_hat = x_hat[num_left - 1:]

    if positive_weights:
        left_weights_hat /= np.sum(left_weights_hat)
        right_weights_hat /= np.sum(right_weights_hat)

    left_recon = (left_side_col @ left_weights_hat).reshape(left_side[0].shape)
    right_recon = (right_side_col @ right_weights_hat).reshape(right_side[0].shape)

    score, score_ci = nan_corr(left_recon, right_recon)

    return score, score_ci, left_recon, right_recon


def simple_get_irms(data_in, inputs_in, sample_rate=0.5, required_num_stim=5, window=[30, 60], sub_pre_stim=True):
    if window[0] < 0 or window[1] < 0 or np.sum(window) <= 0:
        raise Exception('window must be positive and sum to > 0')

    window = [int(i / sample_rate) for i in window]

    irfs, irfs_sem, irfs_all = get_impulse_response_function(data_in, inputs_in, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

    irms = np.nanmean(irfs[window[0]:], axis=0)
    irms[np.eye(irms.shape[0], dtype=bool)] = np.nan

    num_neurons = irfs.shape[1]
    num_stim = np.zeros((num_neurons, num_neurons))
    for ni in range(num_neurons):
        for nj in range(num_neurons):
            resp_to_stim = irfs_all[ni][:, window[0]:, nj]
            num_obs_when_stim = np.sum(np.mean(~np.isnan(resp_to_stim), axis=1) >= 0.5)
            num_stim[nj, ni] += num_obs_when_stim

    irms[num_stim < required_num_stim] = np.nan

    return irms, irfs, irfs_sem


def calculate_irfs(model, duration=60):
    # get the direct impulse response function between each pair of neurons
    A = model.stack_dynamics_weights()
    B = model.dynamics_input_weights_diagonal()
    num_t = int(duration / model.sample_rate)
    u = np.zeros(num_t + model.dynamics_input_lags)
    u[int(model.dynamics_input_lags)] = 1
    irfs = np.zeros((num_t, model.dynamics_dim, model.dynamics_dim))

    print('Calculating IRFs')
    # loop through the stimulated neuron
    for s in range(model.dynamics_dim):
        # loop through responding neuron
        input_weights = B[:, s]
        activity_history = np.zeros((model.dynamics_lags, model.dynamics_dim))

        for t in np.arange(num_t):
            current_activity = np.sum(A @ activity_history[:, :, None], axis=0)[:, 0]
            current_activity[s] += input_weights @ u[t:t + model.dynamics_input_lags]
            irfs[t, :, s] = current_activity
            activity_history = np.roll(activity_history, 1, axis=0)
            activity_history[0, :] = current_activity

        print(s + 1, '/', model.dynamics_dim)

    return irfs


def calculate_dirfs(model, duration=60):
    # get the direct impulse response function between each pair of neurons
    A = model.stack_dynamics_weights()
    B = model.dynamics_input_weights_diagonal()
    num_t = int(duration / model.sample_rate)
    u = np.zeros(num_t + model.dynamics_input_lags)
    u[int(model.dynamics_input_lags)] = 1
    dirfs = np.zeros((num_t, model.dynamics_dim, model.dynamics_dim))

    print('Calculating dIRFs')
    # loop through the stimulated neuron
    for s in range(model.dynamics_dim):
        # loop through responding neuron
        for r in range(model.dynamics_dim):
            weights = A[:, (s, r), :][:, :, (s, r)]
            input_weights = B[:, s]
            activity_history = np.zeros((model.dynamics_lags, 2))

            if np.all(weights[:, 1, 0] == 0):
                continue

            for t in np.arange(num_t):
                current_activity = np.sum(weights @ activity_history[:, :, None], axis=0)[:, 0]
                current_activity[0] += input_weights @ u[t:t + model.dynamics_input_lags]
                dirfs[t, r, s] = current_activity[1]
                activity_history = np.roll(activity_history, 1, axis=0)
                activity_history[0, :] = current_activity

        print(s + 1, '/', model.dynamics_dim)

    return dirfs


def balanced_accuracy(y_true, y_hat):
    # number of correct hits out of total correct
    true_positives = y_true == 1
    tpr = np.nanmean(y_hat[true_positives] == 1)
    true_negatives = y_true == 0
    tnr = np.nanmean(y_hat[true_negatives] == 0)

    return (tpr + tnr) / 2


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


def nan_r2(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot

    return r2


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
    ci = (np.abs(ci_l - corr), ci_u - corr)

    return corr, ci


def frac_explainable_var(y_true, y_hat, y_true_std_trials):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)
    y_true_std_trials = y_true_std_trials.reshape(-1)

    non_nan = ~np.isnan(y_true_std_trials)

    y_true = y_true[non_nan]
    y_hat = y_hat[non_nan]
    y_true_std_trials = y_true_std_trials[non_nan]

    m = y_true.shape[0]
    y_true = y_true - np.mean(y_true)
    y_hat = y_hat - np.mean(y_hat)

    y_true_var_trials = y_true_std_trials**2
    y_true_var_trials_mean = np.mean(y_true_var_trials)

    numerator = np.dot(y_true, y_hat)**2
    denominator = np.sum(y_true**2) * np.sum(y_hat**2)
    numerator_correction = y_true_var_trials_mean * np.sum(y_hat**2)
    denominator_correction = (m - 1) * y_true_var_trials_mean * np.sum(y_hat**2)

    r_er_square = (numerator - numerator_correction) / (denominator - denominator_correction)

    return r_er_square


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


