import numpy as np
import warnings
import wormneuroatlas as wa
import scipy
import pickle
from pathlib import Path
from scipy.stats import norm


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


def simple_get_irms(data_in, inputs_in, sample_rate=0.5, required_num_stim=0, window=[15, 30], sub_pre_stim=True):
    if window[0] < 0 or window[1] < 0 or np.sum(window) <= 0:
        raise Exception('window must be positive and sum to > 0')

    window = [int(i / sample_rate) for i in window]

    irfs, irfs_sem, irfs_all = get_impulse_response_function(data_in, inputs_in, window=window, sub_pre_stim=sub_pre_stim, return_pre=True)

    irms = np.nanmean(irfs[window[0]:], axis=0)
    irms[np.eye(irms.shape[0], dtype=bool)] = np.nan

    # count the number of stimulation events
    num_neurons = irfs.shape[1]
    num_stim = np.zeros((num_neurons, num_neurons))
    for ni in range(num_neurons):
        for nj in range(num_neurons):
            resp_to_stim = irfs_all[ni][:, window[0]:, nj]
            num_obs_when_stim = np.sum(np.mean(~np.isnan(resp_to_stim), axis=1) >= 0.5)
            num_stim[nj, ni] += num_obs_when_stim

    irms[num_stim < required_num_stim] = np.nan

    return irms, irfs, irfs_sem, num_stim


def calculate_irfs(model, window=(30, 60)):
    # get the direct impulse response function between each pair of neurons
    A = model.stack_dynamics_weights()
    B = model.dynamics_input_weights_diagonal()
    num_t = int(window[1] / model.sample_rate)
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

    zero_pad = np.zeros((int(window[0] / model.sample_rate), model.dynamics_dim, model.dynamics_dim))
    irfs = np.concatenate((zero_pad, irfs), axis=0)

    return irfs


def calculate_dirfs(model, window=(30, 60)):
    # get the direct impulse response function between each pair of neurons
    A = model.stack_dynamics_weights()
    B = model.dynamics_input_weights_diagonal()
    num_t = int(window[1] / model.sample_rate)
    u = np.zeros(num_t + model.dynamics_input_lags)
    u[int(model.dynamics_input_lags)] = 1
    dirfs = np.zeros((num_t, model.dynamics_dim, model.dynamics_dim))
    dirfs[:, np.eye(model.dynamics_dim, dtype=bool)] = np.nan

    print('Calculating dIRFs')
    # loop through the stimulated neuron
    for s in range(model.dynamics_dim):
        # loop through responding neuron
        for r in range(model.dynamics_dim):
            if s == r:
                continue

            weights = A[:, (s, r), :][:, :, (s, r)]
            weights[:, 0, 1] = 0
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

    zero_pad = np.zeros((int(window[0] / model.sample_rate), model.dynamics_dim, model.dynamics_dim))
    dirfs = np.concatenate((zero_pad, dirfs), axis=0)

    return dirfs


def calculate_eirfs(model, window=(30, 60)):
    # get the direct impulse response function between each pair of neurons
    A = model.stack_dynamics_weights()
    num_t = int(window[1] / model.sample_rate)
    effective_weights = np.zeros((num_t, model.dynamics_dim, model.dynamics_dim))
    effective_weights[:, np.eye(model.dynamics_dim, dtype=bool)] = np.nan

    print('Calculating effective weights')
    # loop through the stimulated neuron
    for s in range(model.dynamics_dim):
        # loop through responding neuron
        for r in range(model.dynamics_dim):
            if s == r:
                continue

            weights = A[:, (s, r), :][:, :, (s, r)]
            weights[:, 0, 1] = 0
            activity_history = np.zeros((model.dynamics_lags, 2))
            activity_history[0, 0] = 1

            if np.all(weights[:, 1, 0] == 0):
                continue

            for t in np.arange(num_t):
                current_activity = np.sum(weights @ activity_history[:, :, None], axis=0)[:, 0]
                effective_weights[t, r, s] = current_activity[1]
                activity_history = np.roll(activity_history, 1, axis=0)
                activity_history[0, :] = current_activity
                a=1

        print(s + 1, '/', model.dynamics_dim)

    zero_pad = np.zeros((int(window[0] / model.sample_rate), model.dynamics_dim, model.dynamics_dim))
    effective_weights = np.concatenate((zero_pad, effective_weights), axis=0)

    return effective_weights


def accuracy(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    return np.mean(y_true == y_hat)


def precision(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    true_positives = np.sum((y_true == 1) & (y_hat == 1))
    false_positives = np.sum((y_true == 0) & (y_hat == 1))

    return true_positives / (true_positives + false_positives)


def recall(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    true_positives = np.sum((y_true == 1) & (y_hat == 1))
    false_negatives = np.sum((y_true == 1) & (y_hat == 0))

    return true_positives / (true_positives + false_negatives)


def f_measure(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    p = precision(y_true, y_hat)
    r = recall(y_true, y_hat)

    return (2 * p * r) / (p + r)


def mutual_info(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    p_y_true = np.array([1 - np.mean(y_true), np.mean(y_true)])
    p_y_hat = np.array([1 - np.mean(y_hat), np.mean(y_hat)])

    p_joint = np.zeros((2, 2))
    p_joint[0, 0] = np.mean((y_true == 0) & (y_hat == 0))
    p_joint[1, 0] = np.mean((y_true == 1) & (y_hat == 0))
    p_joint[0, 1] = np.mean((y_true == 0) & (y_hat == 1))
    p_joint[1, 1] = np.mean((y_true == 1) & (y_hat == 1))

    p_outer = p_y_true[:, None] * p_y_hat[None, :]

    mi = 0
    for i in range(2):
        for j in range(2):
            if p_joint[i, j] != 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / p_outer[i, j])

    return mi


def metric_ci(metric, y_true, y_hat, alpha=0.05, n_boot=1000, rng=np.random.default_rng()):
    y_true = y_true.astype(float)
    y_hat = y_hat.astype(float)

    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    mi = metric(y_true, y_hat)
    booted_mi = np.zeros(n_boot)
    mi_ci = np.zeros(2)

    for n in range(n_boot):
        sample_inds = rng.integers(0, high=y_true.shape[0], size=y_true.shape[0])
        y_true_resampled = y_true[sample_inds]
        y_hat_resampled = y_hat[sample_inds]
        booted_mi[n] = metric(y_true_resampled, y_hat_resampled)

    mi_ci[0] = np.percentile(booted_mi, alpha * 100)
    mi_ci[1] = np.percentile(booted_mi, (1 - alpha) * 100)

    mi_ci = np.abs(mi_ci - mi)

    return mi, mi_ci


def metric_null(metric, y_true, n_sample=1000, rng=np.random.default_rng()):
    y_true = y_true.reshape(-1)

    nan_loc = np.isnan(y_true)
    y_true = y_true[~nan_loc]

    py = np.mean(y_true)
    sampled_mi = np.zeros(n_sample)

    for n in range(n_sample):
        random_example = rng.uniform(0, 1, size=y_true.shape) < py
        sampled_mi[n] = metric(y_true, random_example)

    return np.mean(sampled_mi)


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


def nan_corr(y_true, y_hat, alpha=0.05, mean_sub=True):
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

    corr = (np.mean(y_true * y_hat) / y_true_std / y_hat_std)

    # now estimate the confidence intervals for the correlation
    n = y_true.shape[0]
    z_a = scipy.stats.norm.ppf(1 - alpha / 2)
    z_r = np.log((1 + corr) / (1 - corr)) / 2
    l = z_r - (z_a / np.sqrt(n - 3))
    u = z_r + (z_a / np.sqrt(n - 3))
    ci_l = (np.exp(2 * l) - 1) / (np.exp(2 * l) + 1)
    ci_u = (np.exp(2 * u) - 1) / (np.exp(2 * u) + 1)
    ci = [np.abs(ci_l - corr), ci_u - corr]

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
            data_corr[i, j], data_corr_ci[:, i, j] = nan_corr(data_cat[:, i], data_cat[:, j], alpha=alpha)

        print(i+1, '/', data_cat.shape[1], 'neurons correlated')

    return data_corr, data_corr_ci


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


