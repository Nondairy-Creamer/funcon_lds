from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
colormap = mpl.colormaps['coolwarm']
plot_percent = 95


def rms(data, axis=-1):
    return np.sqrt(np.nanmean(data**2, axis=axis))


def nan_convolve(data, filter):
    # attempt to ignore nans during a convolution
    # this isn't particularly principled, will just replace nans with 0s and divide the convolution
    # by the fraction of data that was in the window
    # only makes sense for nonnegative filters

    if np.any(filter < 0):
        raise Exception('nan_filter can only handle nonnegative filters')

    nan_loc = np.isnan(data)
    data_no_nan = data
    data_no_nan[nan_loc] = 0
    data_filtered = np.convolve(data_no_nan, filter, mode='valid')
    nan_count = np.convolve(~nan_loc, filter / np.sum(filter), mode='valid')
    nan_count[nan_count == 0] = 1
    data_nan_conv = data_filtered / nan_count
    data_nan_conv[nan_loc[:data_filtered.shape[0]]] = np.nan

    return data_nan_conv


def stack_weights(weights, num_split, axis=-1):
    return np.stack(np.split(weights, num_split, axis=axis))


def plot_log_likelihood(model):
    plt.figure()
    plt.plot(model.log_likelihood)
    plt.xlabel('EM iterations')
    plt.ylabel('log likelihood')
    plt.show()


def plot_model_params(model, cell_ids_chosen):
    model_params = model.get_params()
    cell_ids = model.cell_ids
    A_full = model_params['trained']['dynamics_weights'][:model.dynamics_dim, :]
    A = np.split(A_full, model.dynamics_lags, axis=1)

    # limit the matrix to the chosen neurons
    neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])

    for aa in range(len(A)):
        A[aa] = A[aa][neuron_inds_chosen, :]
        A[aa] = A[aa][:, neuron_inds_chosen]

    plot_x = np.arange(A[0].shape[0])

    # get rid of the diagonal
    for aa in range(len(A)):
        A[aa][np.eye(A[aa].shape[0], dtype=bool)] = 0

    cmax = np.max(np.abs(A))

    # plot the A matrix
    for ai, aa in enumerate(A):
        plt.figure()
        plt.imshow(aa, interpolation='nearest', cmap=colormap)
        plt.clim((-cmax, cmax))
        plt.yticks(plot_x, cell_ids_chosen)
        plt.xticks(plot_x, cell_ids_chosen)
        plt.colorbar()
        plt.title('A' + str(ai))

    # plot the B matrix
    if model.param_props['mask']['dynamics_input_weights'] is None:
        stimulated_neurons = np.zeros(model.input_dim) == 0
    else:
        stimulated_neurons = np.any(model.param_props['mask']['dynamics_input_weights'].astype(bool), axis=1)

    B_full = model_params['trained']['dynamics_input_weights'][:model.dynamics_dim, :]
    B = np.split(B_full, model.dynamics_input_lags, axis=1)
    B = [np.diag(i) for i in B]
    B = [i[:, None] for i in B]
    B = np.concatenate(B, axis=1)
    B[~stimulated_neurons, :] = np.nan
    B = B[neuron_inds_chosen, :]
    cmax = np.nanpercentile(np.abs(B), plot_percent)

    plt.figure()
    plt.imshow(B, interpolation='nearest', cmap=colormap)
    plt.yticks(plot_x, cell_ids_chosen)
    plt.xlabel('lags')
    plt.title('B')
    plt.clim((-cmax, cmax))
    plt.colorbar()

    # Plot the Q matrix
    Q = model_params['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim]
    Q = Q[neuron_inds_chosen, :]
    Q = Q[:, neuron_inds_chosen]
    cmax = np.max(np.abs(Q))
    plt.figure()
    plt.imshow(Q, interpolation='nearest', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.yticks(plot_x, cell_ids_chosen)
    plt.xticks(plot_x, cell_ids_chosen)
    plt.title('Q')
    plt.colorbar()

    # Plot the R matrix
    R = model_params['trained']['emissions_cov']
    R = R[neuron_inds_chosen, :]
    R = R[:, neuron_inds_chosen]
    cmax = np.max(np.abs(R))
    plt.figure()
    plt.imshow(R, interpolation='nearest', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.yticks(plot_x, cell_ids_chosen)
    plt.xticks(plot_x, cell_ids_chosen)
    plt.title('R')
    plt.colorbar()

    plt.show()


def get_stim_response(data, inputs, window=(-20, 60), sub_start=False):
    num_neurons = data[0].shape[1]

    responses = []
    for n in range(num_neurons):
        responses.append([])

    for e, i in zip(data, inputs):
        num_time = e.shape[0]
        stim_events = np.where(i)

        for time, target in zip(stim_events[0], stim_events[1]):
            if window[0] + time >= 0 and window[1] + time < num_time:
                this_clip = e[window[0]+time:window[1]+time, :]

                if sub_start:
                    if window[0] < 0:
                        baseline = np.nanmean(this_clip[0:-window[0], :], axis=0)
                        baseline[np.isnan(baseline)] = 0
                        this_clip -= baseline

                responses[target].append(this_clip)

    for ri, r in enumerate(responses):
        if len(r) > 0:
            responses[ri] = np.stack(r)
        else:
            responses[ri] = np.zeros((0, window[1] - window[0], num_neurons))

    ave_responses = [np.nanmean(j, axis=0) for j in responses]
    ave_responses = np.stack(ave_responses)
    ave_responses = np.transpose(ave_responses, axes=(1, 2, 0))

    sem_responses = [np.nanstd(j, axis=0, ddof=1) / np.sqrt(j.shape[0]) for j in responses]
    sem_responses = np.stack(sem_responses)
    sem_responses = np.transpose(sem_responses, axes=(1, 2, 0))

    return ave_responses, sem_responses, responses


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


def plot_posterior(data, posterior_dict, cell_ids_chosen, sample_rate=0.5, window_size=1000):
    emissions = data['emissions']
    inputs = data['inputs']
    posterior = posterior_dict['posterior']
    post_pred = posterior_dict['post_pred_noise']

    neuron_inds_chosen = np.array([data['cell_ids'].index(i) for i in cell_ids_chosen])

    # get all the inputs but with only the chosen neurons
    inputs_truncated = [i[:, neuron_inds_chosen] for i in inputs]
    data_ind_chosen, time_window = find_stim_events(inputs_truncated, window_size=window_size)

    emissions_chosen = emissions[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    inputs_chosen = inputs[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    posterior_chosen = posterior[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    post_pred_chosen = post_pred[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]

    filt_shape = np.ones(5)
    for i in range(inputs_chosen.shape[1]):
        inputs_chosen[:, i] = np.convolve(inputs_chosen[:, i], filt_shape, mode='same')

    plot_y = np.arange(len(cell_ids_chosen))
    plot_x = np.arange(0, emissions_chosen.shape[0], emissions_chosen.shape[0]/10)
    cmax = np.nanpercentile(np.abs((emissions_chosen, posterior_chosen)), plot_percent)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(inputs_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.clim((-1, 1))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(emissions_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('measured data')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(posterior_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('model posterior, all data')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.tight_layout()

    plt.figure()
    cmax = np.nanpercentile(np.abs((post_pred_chosen, posterior_chosen)), plot_percent)

    plt.subplot(3, 1, 1)
    plt.imshow(inputs_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.clim((-1, 1))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(post_pred_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('sampled model')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(posterior_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('model posterior, all data')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.tight_layout()

    plt.show()


def plot_missing_neuron(model, data, posterior_dict, cell_ids, neuron_to_remove, time_window, sample_rate=0.5):
    emissions = data['emissions']
    inputs = data['inputs']
    init_mean = data['init_mean']
    init_cov = data['init_cov']

    missing_neuron_ind = cell_ids.index(neuron_to_remove)

    # plot the posterior with a neuron missing
    # get the posterior with a neuron missing
    emissions_missing = emissions
    emissions_missing[:, missing_neuron_ind] = np.nan
    posterior_missing = model.lgssm_smoother(emissions_missing, inputs, init_mean, init_cov)[1]
    posterior_missing = (posterior_missing @ model.emissions_weights.T)
    posterior_missing = posterior_missing[time_window[0]:time_window[1], missing_neuron_ind]

    posterior = posterior_dict['posterior'][time_window[0]:time_window[1], missing_neuron_ind]
    emissions = emissions[time_window[0]:time_window[1], missing_neuron_ind]

    plot_x = np.arange(emissions.shape[0]) * sample_rate
    plt.figure()
    plt.plot(plot_x, emissions, label='measured')
    plt.plot(plot_x, posterior, label='full posterior')
    plt.plot(plot_x, posterior_missing, label='posterior w missing neuron')
    plt.xlabel('time (s)')
    plt.legend()
    plt.title(neuron_to_remove + ' missing')
    plt.show()


def nan_corr(data):
    # calculate the average cross correlation between neurons
    emissions_cov = []
    num_neurons = data[0].shape[1]
    for i in range(len(data)):
        emissions_this = data[i]
        nan_loc = np.isnan(emissions_this)
        em_z_score = (emissions_this - np.nanmean(emissions_this, axis=0)) / np.nanstd(emissions_this, axis=0)
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


def plot_stim_l2_norm(model, data, posterior_dict, cell_ids_chosen, window=(0, 120)):
    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']
    posterior = posterior_dict['posterior']
    post_pred = posterior_dict['post_pred']

    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]

    plot_x = np.arange(len(chosen_neuron_inds))

    # go through emissions and get the average input response for each neuron
    measured_stim_responses, measured_stim_responses_sem = get_stim_response(emissions, inputs, window=window)[:2]
    post_pred_stim_responses, post_pred_stim_responses_sem = get_stim_response(post_pred, inputs, window=window)[:2]
    posterior_stim_responses, posterior_stim_responses_sem = get_stim_response(posterior, inputs, window=window)[:2]
    model_weights = model.dynamics_weights[:model.dynamics_dim, :]

    # calculate the rms for each response. This deals with the fact that responses can vary and go negative
    measured_response_norm = rms(measured_stim_responses, axis=0)
    post_pred_response_norm = rms(post_pred_stim_responses, axis=0)
    posterior_response_norm = rms(posterior_stim_responses, axis=0)
    model_weights_norm = rms(stack_weights(model_weights, model.dynamics_lags, axis=1), axis=0)
    correlation = nan_corr(emissions)

    correlation[np.isnan(correlation)] = 0
    if np.linalg.det(correlation) > 0:
        correlation = np.abs(np.linalg.inv(correlation))[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    elif np.linalg.det(correlation[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]) > 0:
        correlation = np.abs(np.linalg.inv(correlation[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]))
    else:
        correlation = np.zeros((len(chosen_neuron_inds), len(chosen_neuron_inds)))

    measured_response_norm = measured_response_norm[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    post_pred_response_norm = post_pred_response_norm[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    posterior_response_norm = posterior_response_norm[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    model_weights_norm = model_weights_norm[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]

    # set the diagonal to 0 for visualization
    correlation[np.diag(~np.isnan(np.diag(correlation)))] = 0
    measured_response_norm[np.diag(~np.isnan(np.diag(measured_response_norm)))] = 0
    post_pred_response_norm[np.diag(~np.isnan(np.diag(post_pred_response_norm)))] = 0
    posterior_response_norm[np.diag(~np.isnan(np.diag(posterior_response_norm)))] = 0
    model_weights_norm[np.diag(~np.isnan(np.diag(model_weights_norm)))] = 0

    # normalize so that everything is on the same scale
    correlation = correlation / np.nanmax(correlation)
    measured_response_norm = measured_response_norm / np.nanmax(measured_response_norm)
    post_pred_response_norm = post_pred_response_norm / np.nanmax(post_pred_response_norm)
    posterior_response_norm = posterior_response_norm / np.nanmax(posterior_response_norm)
    model_weights_norm = model_weights_norm / np.max(model_weights_norm)

    plt.figure()

    ax = plt.subplot(2, 2, 1)
    plt.imshow(measured_response_norm, interpolation='nearest', cmap=colormap)
    plt.title('measured response L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 2)
    plt.imshow(post_pred_response_norm, interpolation='nearest', cmap=colormap)
    plt.title('post pred response L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 3)
    plt.imshow(correlation, interpolation='nearest', cmap=colormap)
    plt.title('correlation inverse')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 4)
    plt.imshow(model_weights_norm, interpolation='nearest', cmap=colormap)
    plt.title('model weights L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    plt.tight_layout()

    plt.figure()
    ax = plt.gca()
    plt.imshow(measured_response_norm, interpolation='nearest', cmap=colormap)
    plt.title('measured response L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    plt.figure()
    ax = plt.gca()
    plt.imshow(post_pred_response_norm, interpolation='nearest', cmap=colormap)
    plt.title('post pred response L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    plt.figure()
    ax = plt.gca()
    plt.imshow(correlation, interpolation='nearest', cmap=colormap)
    plt.title('correlation')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    plt.figure()
    ax = plt.gca()
    plt.imshow(model_weights_norm, interpolation='nearest', cmap=colormap)
    plt.title('model weights L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    plt.show()


def plot_stim_response(data, posterior_dict, cell_ids_chosen, neuron_to_stim,
                       window=(-60, 120), sample_rate=0.5):
    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']
    posterior = posterior_dict['posterior']
    post_pred = posterior_dict['post_pred']

    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    plot_x = np.arange(window[0], window[1]) * sample_rate
    neuron_to_stim_ind = cell_ids_chosen.index(neuron_to_stim)

    # go through emissions and get the average input response for each neuron
    measured_stim_responses, measured_stim_responses_sem = get_stim_response(emissions, inputs, window=window)[:2]
    post_pred_stim_responses, post_pred_stim_responses_sem = get_stim_response(post_pred, inputs, window=window)[:2]
    posterior_stim_responses, posterior_stim_responses_sem = get_stim_response(posterior, inputs, window=window)[:2]

    # pull out the neurons we care about
    measured_stim_responses = measured_stim_responses[:, chosen_neuron_inds, :]
    measured_stim_responses = measured_stim_responses[:, :, chosen_neuron_inds]
    measured_stim_responses_sem = measured_stim_responses_sem[:, chosen_neuron_inds, :]
    measured_stim_responses_sem = measured_stim_responses_sem[:, :, chosen_neuron_inds]

    post_pred_stim_responses = post_pred_stim_responses[:, chosen_neuron_inds, :]
    post_pred_stim_responses = post_pred_stim_responses[:, :, chosen_neuron_inds]
    post_pred_stim_responses_sem = post_pred_stim_responses_sem[:, chosen_neuron_inds, :]
    post_pred_stim_responses_sem = post_pred_stim_responses_sem[:, :, chosen_neuron_inds]

    posterior_stim_responses = posterior_stim_responses[:, chosen_neuron_inds, :]
    posterior_stim_responses = posterior_stim_responses[:, :, chosen_neuron_inds]
    posterior_stim_responses_sem = posterior_stim_responses_sem[:, chosen_neuron_inds, :]
    posterior_stim_responses_sem = posterior_stim_responses_sem[:, :, chosen_neuron_inds]

    vals = [measured_stim_responses[:, :, neuron_to_stim_ind], posterior_stim_responses[:, :, neuron_to_stim_ind],
            post_pred_stim_responses[:, :, neuron_to_stim_ind]]
    ylim = (np.nanmin(vals), np.nanmax(vals))

    for i in range(measured_stim_responses.shape[1]):
        plt.figure()
        this_measured_resp = measured_stim_responses[:, i, neuron_to_stim_ind]
        this_measured_resp_sem = measured_stim_responses_sem[:, i, neuron_to_stim_ind]
        plt.plot(plot_x, this_measured_resp, label='measured')
        plt.fill_between(plot_x, this_measured_resp - this_measured_resp_sem/2, this_measured_resp + this_measured_resp_sem/2, alpha=0.4)
        plt.plot(plot_x, posterior_stim_responses[:, i, neuron_to_stim_ind], label='smoothed')
        plt.plot(plot_x, post_pred_stim_responses[:, i, neuron_to_stim_ind], label='model')
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylim(ylim)
        plt.ylabel(cell_ids_chosen[i])
        plt.xlabel('time (s)')

        plt.legend()
        plt.title('average responses to stimulation of: ' + neuron_to_stim)

    plt.show()


def plot_dynamics_eigs(model):
    A = model.dynamics_weights

    d_eigvals = np.linalg.eigvals(A)

    plt.figure()
    plt.plot(np.sort(np.abs(d_eigvals))[::-1])
    plt.ylabel('magnitude')
    plt.xlabel('eigen values')

    plt.figure()
    plt.scatter(np.real(d_eigvals), np.imag(d_eigvals))
    plt.xlabel('real components')
    plt.ylabel('imaginary components')
    plt.title('eigenvalues of the dynamics matrix')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    plt.show()

