from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import wormneuroatlas as wa
colormap = mpl.colormaps['coolwarm']
plot_percent = 95


def rms(data, axis=-1):
    return np.sqrt(np.nanmean(data**2, axis=axis))


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


def get_stim_response(data, inputs, window=(-60, 120), sub_start=False, return_pre=True):
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

                if sub_start:
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

    ave_responses = [np.nanmean(j, axis=0) for j in responses]
    ave_responses = np.stack(ave_responses)
    ave_responses = np.transpose(ave_responses, axes=(1, 2, 0))

    ave_responses_sem = [np.nanstd(j, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(j), axis=0)) for j in responses]
    ave_responses_sem = np.stack(ave_responses_sem)
    ave_responses_sem = np.transpose(ave_responses_sem, axes=(1, 2, 0))

    return ave_responses, ave_responses_sem, responses


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


def plot_missing_neuron(model, data, posterior_dict, cell_ids_chosen, neuron_to_remove, window_size=1000, force_calc=False):
    cell_ids = data['cell_ids']
    cell_ids_chosen_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    chosen_inputs = [i[:, cell_ids_chosen_inds] for i in data['inputs']]
    data_ind, time_window = find_stim_events(chosen_inputs, window_size=window_size)

    emissions = data['emissions'][data_ind]
    inputs = data['inputs'][data_ind]
    init_mean = posterior_dict['init_mean'][data_ind]
    init_cov = posterior_dict['init_cov'][data_ind]
    sample_rate = model.sample_rate

    missing_neuron_ind = cell_ids.index(neuron_to_remove)

    # plot the posterior with a neuron missing
    # get the posterior with a neuron missing
    if 'posterior_missing' in posterior_dict.keys():
        if posterior_dict['posterior_missing']['missing_neuron'] != neuron_to_remove:
            force_calc = True
    else:
        force_calc = True

    if force_calc:
        emissions_missing = emissions.copy()
        emissions_missing[:, missing_neuron_ind] = np.nan
        posterior_missing = model.lgssm_smoother(emissions_missing, inputs, init_mean, init_cov)[1]
        posterior_missing = (posterior_missing @ model.emissions_weights.T)
        posterior_missing = posterior_missing[time_window[0]:time_window[1], missing_neuron_ind]
    else:
        posterior_missing = posterior_dict['posterior_missing']['posterior']

    emissions = emissions[time_window[0]:time_window[1], missing_neuron_ind]
    inputs = inputs[time_window[0]:time_window[1], :]
    stim_time, stim_inds = np.where(inputs==1)
    stim_names = [cell_ids[i] for i in stim_inds]

    plot_x = np.arange(emissions.shape[0]) * sample_rate
    plt.figure()
    plt.plot(plot_x, emissions, label='measured')
    plt.plot(plot_x, posterior_missing, label='posterior, ' + neuron_to_remove + ' unmeasured')
    ylim = plt.ylim()
    for si, s in enumerate(stim_time):
        plt.axvline(plot_x[s], color=[0.5, 0.5, 0.5])
        plt.text(plot_x[s], ylim[1], stim_names[si], rotation=45)
    plt.xlabel('time (s)')
    plt.ylabel(neuron_to_remove + ' activity')
    plt.legend()
    plt.show()

    return {'posterior': posterior_missing, 'missing_neuron': neuron_to_remove}


def nan_corr_data(data):
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


def nancorrcoef(data):
    num_data = len(data)
    corr_coef = np.zeros((num_data, num_data))

    for ii, i in enumerate(data):
        for ji, j in enumerate(data):
            i = (i - np.nanmean(i, axis=0)) / np.nanstd(i, ddof=1)
            j = (j - np.nanmean(j, axis=0)) / np.nanstd(j, ddof=1)

            corr_coef[ii, ji] = np.nanmean(i * j)

    return corr_coef


def plot_stim_l2_norm(model, data, posterior_dict, cell_ids_chosen, window=(-60, 120), sub_start=True):
    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']
    post_pred = posterior_dict['post_pred']

    watlas = wa.NeuroAtlas()
    atlas_ids = list(watlas.neuron_ids)
    anatomical_connectome_full = watlas.get_anatomical_connectome(signed=False)
    peptide_connectome_full = watlas.get_peptidergic_connectome()
    atlas_ids[atlas_ids.index('AWCON')] = 'AWCR'
    atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCL'
    atlas_inds = [atlas_ids.index(i) for i in cell_ids]
    anatomical_connectome = anatomical_connectome_full[np.ix_(atlas_inds, atlas_inds)]
    peptide_connectome = peptide_connectome_full[np.ix_(atlas_inds, atlas_inds)]

    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]

    # go through emissions and get the average input response for each neuron
    measured_stim_responses = get_stim_response(emissions, inputs, window=window, sub_start=sub_start, return_pre=False)[0]
    post_pred_stim_responses = get_stim_response(post_pred, inputs, window=window, sub_start=sub_start, return_pre=False)[0]
    model_weights = model.dynamics_weights[:model.dynamics_dim, :]

    # calculate the rms for each response. This deals with the fact that responses can vary and go negative
    measured_response_norm = rms(measured_stim_responses, axis=0)
    post_pred_response_norm = rms(post_pred_stim_responses, axis=0)
    model_weights_norm = rms(stack_weights(model_weights, model.dynamics_lags, axis=1), axis=0)
    correlation = nan_corr_data(emissions)
    correlation = np.abs(correlation)

    # set the diagonal to 0 for visualization
    correlation[np.eye(correlation.shape[0], dtype=bool)] = np.nan
    measured_response_norm[np.eye(correlation.shape[0], dtype=bool)] = np.nan
    post_pred_response_norm[np.eye(correlation.shape[0], dtype=bool)] = np.nan
    model_weights_norm[np.eye(correlation.shape[0], dtype=bool)] = np.nan

    mask = np.isnan(correlation) | np.isnan(measured_response_norm)
    post_pred_masked = post_pred_response_norm.copy()
    post_pred_masked[mask] = np.nan
    model_weights_masked = model_weights_norm.copy()
    model_weights_masked[mask] = np.nan
    correlation_masked = correlation.copy()
    correlation_masked[mask] = np.nan

    # compare each of the weights against measured
    m_c = nancorrcoef([measured_response_norm.reshape(-1), correlation_masked.reshape(-1)])[0, 1]
    m_pp = nancorrcoef([measured_response_norm.reshape(-1), post_pred_masked.reshape(-1)])[0, 1]
    m_w = nancorrcoef([measured_response_norm.reshape(-1), model_weights_masked.reshape(-1)])[0, 1]

    # compare each of the weights against the model weights
    w_c = nancorrcoef([model_weights_norm.reshape(-1), correlation_masked.reshape(-1)])[0, 1]
    w_pp = nancorrcoef([model_weights_norm.reshape(-1), post_pred_masked.reshape(-1)])[0, 1]

    # fit the connectome to the correlation matrix and fit it to the model weights
    A = np.concatenate((anatomical_connectome.reshape(-1)[:, None], peptide_connectome.reshape(-1)[:, None]), axis=1)
    correlation_no_nan = correlation_masked.copy().reshape(-1)
    nan_loc = np.isnan(correlation_no_nan)
    weights_no_nan = model_weights_masked.copy().reshape(-1)

    correlation_no_nan = correlation_no_nan[~nan_loc]
    weights_no_nan = weights_no_nan[~nan_loc]
    A = A[~nan_loc, :]

    connectome_to_corr = A @ np.linalg.lstsq(A, correlation_no_nan)[0]
    connectome_to_weights = A @ np.linalg.lstsq(A, weights_no_nan)[0]

    connect_corr_score = nancorrcoef([connectome_to_corr, correlation_no_nan])[0, 1]
    connect_weights_score = nancorrcoef([connectome_to_weights, weights_no_nan])[0, 1]

    plt.figure()
    plot_x = np.arange(2)
    plt.bar(plot_x, [connect_corr_score, connect_weights_score])
    plt.xticks(plot_x, ['correlation', 'model weights'])
    plt.xlabel('correlation')
    plt.title('correlation to connectome')

    # plot the comparison of the matricies
    plt.figure()
    plot_x = np.arange(3)
    ax = plt.subplot(1, 2, 1)
    plt.bar(plot_x, [m_c, m_pp, m_w])
    plt.xticks(plot_x, ['correlation', 'post_pred', 'model_weights'])
    plt.ylabel('correlation to impulse responses')
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    ax = plt.subplot(1, 2, 2)
    plt.bar(plot_x, [w_c, w_pp, m_w])
    plt.xticks(plot_x, ['correlation', 'post_pred', 'impulse responses'])
    plt.ylabel('correlation to model weights')
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    plt.tight_layout()


    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(correlation.reshape(-1), measured_response_norm.reshape(-1))
    plt.xlabel('weights from correlation')
    plt.ylabel('measured impulse response')
    plt.subplot(2, 2, 2)
    plt.scatter(post_pred_masked.reshape(-1), measured_response_norm.reshape(-1))
    plt.xlabel('model impulse responses')
    plt.ylabel('measured impulse response')
    plt.subplot(2, 2, 3)
    plt.scatter(post_pred_masked.reshape(-1), correlation.reshape(-1))
    plt.xlabel('model impulse responses')
    plt.ylabel('weights from correlation')
    plt.tight_layout()

    # plot each of the estimated weight matricies
    correlation_plt = correlation.copy()
    measured_response_norm_plt = measured_response_norm.copy()
    post_pred_response_norm_plt = post_pred_response_norm.copy()
    model_weights_norm_plt = model_weights_norm.copy()

    correlation_plt = correlation_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    measured_response_norm_plt = measured_response_norm_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    post_pred_response_norm_plt = post_pred_response_norm_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    model_weights_norm_plt = model_weights_norm_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]

    correlation_plt = correlation_plt / np.nanmax(correlation_plt)
    measured_response_norm_plt = measured_response_norm_plt / np.nanmax(measured_response_norm_plt)
    post_pred_response_norm_plt = post_pred_response_norm_plt / np.nanmax(post_pred_response_norm_plt)
    model_weights_norm_plt = model_weights_norm_plt / np.nanmax(model_weights_norm_plt)

    plt.figure()
    plot_x = np.arange(len(chosen_neuron_inds))

    ax = plt.subplot(2, 2, 1)
    plt.imshow(measured_response_norm_plt, interpolation='nearest', cmap=colormap)
    plt.title('measured response L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 2)
    plt.imshow(post_pred_response_norm_plt, interpolation='nearest', cmap=colormap)
    plt.title('post pred response L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 3)
    plt.imshow(correlation_plt, interpolation='nearest', cmap=colormap)
    plt.title('correlation')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 4)
    plt.imshow(model_weights_norm_plt, interpolation='nearest', cmap=colormap)
    plt.title('model weights L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    plt.tight_layout()

    plt.show()


def plot_stim_response(data, posterior_dict, cell_ids_chosen, neuron_to_stim,
                       window=(-60, 120), sample_rate=0.5, sub_start=True, num_plot=5):
    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']
    post_pred = posterior_dict['post_pred']
    posterior = posterior_dict['posterior']

    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    plot_x = np.arange(window[0], window[1]) * sample_rate
    neuron_to_stim_ind = cell_ids_chosen.index(neuron_to_stim)

    # go through emissions and get the average input response for each neuron
    measured_stim_responses, measured_stim_responses_sem = get_stim_response(emissions, inputs, window=window, sub_start=sub_start)[:2]
    posterior_stim_responses = get_stim_response(posterior, inputs, window=window, sub_start=sub_start)[0]
    post_pred_stim_responses = get_stim_response(post_pred, inputs, window=window, sub_start=sub_start)[0]

    # pull out the neurons we care about
    measured_stim_responses = measured_stim_responses[:, chosen_neuron_inds, :]
    measured_stim_responses = measured_stim_responses[:, :, chosen_neuron_inds]
    measured_stim_responses_sem = measured_stim_responses_sem[:, chosen_neuron_inds, :]
    measured_stim_responses_sem = measured_stim_responses_sem[:, :, chosen_neuron_inds]
    measured_stim_responses_l2 = rms(measured_stim_responses, axis=0)
    measured_stim_responses_l2[np.eye(measured_stim_responses_l2.shape[0], dtype=bool)] = 0
    post_pred_stim_responses = post_pred_stim_responses[:, chosen_neuron_inds, :]
    post_pred_stim_responses = post_pred_stim_responses[:, :, chosen_neuron_inds]
    posterior_stim_responses = posterior_stim_responses[:, chosen_neuron_inds, :]
    posterior_stim_responses = posterior_stim_responses[:, :, chosen_neuron_inds]

    # find the 5 highest responses to plot
    sorted_vals = np.sort(measured_stim_responses_l2.reshape(-1))
    plot_inds = []
    for m in range(num_plot):
        best = np.where(measured_stim_responses_l2 == sorted_vals[-(m+1)])
        plot_inds.append((best[0][0], best[1][0]))

    measured_response_chosen = np.zeros((measured_stim_responses.shape[0], num_plot))
    measured_response_sem_chosen = np.zeros((measured_stim_responses.shape[0], num_plot))
    post_pred_response_chosen = np.zeros((measured_stim_responses.shape[0], num_plot))
    posterior_response_chosen = np.zeros((measured_stim_responses.shape[0], num_plot))

    for pi, p in enumerate(plot_inds):
        measured_response_sem_chosen[:, pi] = measured_stim_responses_sem[:, p[0], p[1]] #/ np.max(measured_stim_responses[:, p[0], p[1]])
        measured_response_chosen[:, pi] = measured_stim_responses[:, p[0], p[1]] #/ np.max(measured_stim_responses[:, p[0], p[1]])
        post_pred_response_chosen[:, pi] = post_pred_stim_responses[:, p[0], p[1]] #/ np.max(post_pred_stim_responses[:, p[0], p[1]])
        posterior_response_chosen[:, pi] = posterior_stim_responses[:, p[0], p[1]] #/ np.max(posterior_stim_responses[:, p[0], p[1]])

    ylim = (np.nanmin([measured_response_chosen - measured_response_sem_chosen, post_pred_response_chosen, posterior_response_chosen]),
            np.nanmax([measured_response_chosen + measured_response_sem_chosen, post_pred_response_chosen, posterior_response_chosen]))

    for i in range(measured_response_chosen.shape[1]):
        plt.figure()
        this_measured_resp = measured_response_chosen[:, i]
        this_measured_resp_sem = measured_response_sem_chosen[:, i]
        plt.plot(plot_x, this_measured_resp, label='measured')
        plt.fill_between(plot_x, this_measured_resp - this_measured_resp_sem, this_measured_resp + this_measured_resp_sem, alpha=0.4)
        plt.plot(plot_x, posterior_response_chosen[:, i], label='posterior')
        plt.plot(plot_x, post_pred_response_chosen[:, i], label='posterior predictive')
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylim(ylim)
        plt.ylabel(cell_ids_chosen[i])
        plt.xlabel('time (s)')

        plt.legend()
        plt.title('average responses to stimulation of: ' + cell_ids_chosen[plot_inds[i][1]])

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

