from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np


def rms(data, axis=-1):
    return np.sqrt(np.mean(data**2, axis=axis))


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


def plot_model_params(model, cell_ids, cell_ids_chosen=None):
    colormap = mpl.colormaps['coolwarm']

    model_params = model.get_params()
    A_full = model_params['trained']['dynamics_weights'][:model.dynamics_dim, :]
    A = np.split(A_full, model.dynamics_lags, axis=1)

    # limit the matrix to the chosen neurons
    if cell_ids_chosen is not None:
        neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])


        for aa in range(len(A)):
            A[aa] = A[aa][neuron_inds_chosen, :]
            A[aa] = A[aa][:, neuron_inds_chosen]

    plot_x = A[0].shape[0]

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
    B_full = model_params['trained']['dynamics_input_weights'][:model.dynamics_dim, :]
    B = np.split(B_full, model.dynamics_input_lags, axis=1)
    B = [i[model.param_props['mask']['dynamics_input_weights'].numpy().astype(bool)] for i in B]
    B = [i[:, None] for i in B]
    B = np.concatenate(B, axis=1)
    B = B[neuron_inds_chosen, :]
    cmax = np.max(np.abs(B))

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


def get_stim_response(emissions, inputs, window=(-20, 60)):
    num_neurons = emissions[0].shape[1]

    # get structure which is responding neuron x stimulated neuron
    stim_responses = np.zeros((window[1] - window[0], num_neurons, num_neurons))
    num_stims = np.zeros((num_neurons, num_neurons))

    for e, i in zip(emissions, inputs):
        num_time = e.shape[0]
        stim_events = np.where(i)

        for time, target in zip(stim_events[0], stim_events[1]):
            if window[0] + time >= 0 and window[1] + time < num_time:
                responses = e[window[0]+time:window[1]+time, :]
                nan_responses = np.any(np.isnan(responses), axis=0)
                responses[:, nan_responses] = 0
                num_stims[~nan_responses, target] += 1
                stim_responses[:, :, target] += responses

    num_stims[num_stims == 0] = np.nan
    stim_responses = stim_responses / num_stims[None, :, :]

    return stim_responses


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


def plot_posterior(emissions, inputs, posterior, prior, cell_ids):
    sample_rate = 0.5
    colormap = mpl.colormaps['coolwarm']

    filt_shape = np.ones(5)
    for i in range(inputs.shape[1]):
        inputs[:, i] = np.convolve(inputs[:, i], filt_shape, mode='same')

    plot_y = np.arange(len(cell_ids))
    plot_x = np.arange(0, emissions.shape[0], emissions.shape[0]/10)
    cmax = np.nanmax(np.abs((emissions, posterior)))

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(inputs.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(plot_y, cell_ids)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.clim((-1, 1))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(emissions.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('measured data')
    plt.yticks(plot_y, cell_ids)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(posterior.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('model posterior, all data')
    plt.yticks(plot_y, cell_ids)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.tight_layout()

    plt.figure()
    cmax = np.nanmax(np.abs((prior, posterior)))

    plt.subplot(3, 1, 1)
    plt.imshow(inputs.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(plot_y, cell_ids)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.clim((-1, 1))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(prior.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('sampled model')
    plt.yticks(plot_y, cell_ids)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(posterior.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('model posterior, all data')
    plt.yticks(plot_y, cell_ids)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.tight_layout()

    plt.show()


def plot_missing_neuron(model, emissions, inputs, posterior, cell_ids, neuron_to_remove, time_window):
    # sample_rate = model.sample_rate
    sample_rate = 0.5

    emissions_torch, inputs_torch = model.standardize_inputs([emissions], [inputs])
    emissions_torch = emissions_torch[0]
    inputs_torch = inputs_torch[0]
    init_mean_torch = model.estimate_init_mean([emissions_torch])[0]
    init_cov_torch = model.estimate_init_cov([emissions_torch])[0]

    missing_neuron_ind = cell_ids.index(neuron_to_remove)

    # plot the posterior with a neuron missing
    # get the posterior with a neuron missing
    emissions_missing = emissions_torch
    emissions_missing[:, missing_neuron_ind] = np.nan
    posterior_missing = model.lgssm_smoother(emissions_missing, inputs_torch, init_mean_torch, init_cov_torch)[3]
    posterior_missing = (posterior_missing @ model.emissions_weights.T).detach().cpu().numpy()
    posterior_missing = posterior_missing[time_window[0]:time_window[1], missing_neuron_ind]

    posterior = posterior[time_window[0]:time_window[1], missing_neuron_ind]
    emissions = emissions[time_window[0]:time_window[1], missing_neuron_ind]

    plot_x = np.arange(emissions.shape[0]) * sample_rate
    plt.figure()
    plt.plot(plot_x, emissions)
    plt.plot(plot_x, posterior)
    plt.plot(plot_x, posterior_missing)
    plt.xlabel('time (s)')
    plt.legend(['measured', 'full posterior', 'posterior missing'])
    plt.title(neuron_to_remove + ' missing')
    plt.show()


def plot_stim_l2_norm(model, emissions, inputs_full, posterior, prior, cell_ids, cell_ids_chosen, window=(0, 120)):
    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    plot_x = np.arange(len(chosen_neuron_inds))
    colormap = mpl.colormaps['coolwarm']

    # go through emissions and get the average input response for each neuron
    measured_stim_responses = get_stim_response(emissions, inputs_full, window=window)
    prior_stim_responses = get_stim_response(prior, inputs_full, window=window)
    posterior_stim_responses = get_stim_response(posterior, inputs_full, window=window)
    model_weights = model.dynamics_weights[:model.dynamics_dim, :]

    # calculate the rms for each response. This deals with the fact that responses can vary and go negative
    measured_response_norm = rms(measured_stim_responses, axis=0)
    prior_response_norm = rms(prior_stim_responses, axis=0)
    posterior_response_norm = rms(posterior_stim_responses, axis=0)
    model_weights_norm = rms(stack_weights(model_weights, model.dynamics_lags, axis=1), axis=0)

    # pull out the neurons we care about
    measured_response_norm = measured_response_norm[chosen_neuron_inds, :][:, chosen_neuron_inds]
    prior_response_norm = prior_response_norm[chosen_neuron_inds, :][:, chosen_neuron_inds]
    posterior_response_norm = posterior_response_norm[chosen_neuron_inds, :][:, chosen_neuron_inds]
    model_weights_norm = model_weights_norm[chosen_neuron_inds, :][:, chosen_neuron_inds]

    # set the diagonal to 0 for visualization
    measured_response_norm[np.eye(measured_response_norm.shape[0], dtype=bool)] = 0
    prior_response_norm[np.eye(prior_response_norm.shape[0], dtype=bool)] = 0
    posterior_response_norm[np.eye(posterior_response_norm.shape[0], dtype=bool)] = 0
    model_weights_norm[np.eye(model_weights_norm.shape[0], dtype=bool)] = 0

    # normalize so that everything is on the same scale
    measured_response_norm = measured_response_norm / np.nanmax(measured_response_norm)
    prior_response_norm = prior_response_norm / np.max(prior_response_norm)
    posterior_response_norm = posterior_response_norm / np.max(posterior_response_norm)
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
    plt.imshow(prior_response_norm, interpolation='nearest', cmap=colormap)
    plt.title('prior response L2 norm')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 3)
    plt.imshow(posterior_response_norm, interpolation='nearest', cmap=colormap)
    plt.title('posterior response L2 norm')
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

    plt.show()


def plot_stim_response(emissions, inputs_full, posterior, prior, cell_ids, cell_ids_chosen, neuron_to_stim, window=(-60, 120)):
    # sample_rate = model.sample_rate
    sample_rate = 0.5
    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    plot_x = np.arange(window[0], window[1]) * sample_rate
    neuron_to_stim_ind = cell_ids_chosen.index(neuron_to_stim)

    # go through emissions and get the average input response for each neuron
    measured_stim_responses = get_stim_response(emissions, inputs_full, window=window)
    prior_stim_responses = get_stim_response(prior, inputs_full, window=window)
    posterior_stim_responses = get_stim_response(posterior, inputs_full, window=window)

    # pull out the neurons we care about
    measured_stim_responses = measured_stim_responses[:, chosen_neuron_inds, :]
    measured_stim_responses = measured_stim_responses[:, :, chosen_neuron_inds]
    prior_stim_responses = prior_stim_responses[:, chosen_neuron_inds, :]
    prior_stim_responses = prior_stim_responses[:, :, chosen_neuron_inds]
    posterior_stim_responses = posterior_stim_responses[:, chosen_neuron_inds, :]
    posterior_stim_responses = posterior_stim_responses[:, :, chosen_neuron_inds]

    # normalize so that everything is on the same scale
    # measured_response_norm = measured_stim_responses / np.max(measured_stim_responses)
    # prior_response_norm = prior_stim_responses / np.max(prior_stim_responses)
    # posterior_response_norm = posterior_stim_responses / np.max(posterior_stim_responses)

    ylim = (np.nanmin(measured_stim_responses), np.nanmax(measured_stim_responses))
    # ylim = (-1.5, 4.0)

    for i in range(measured_stim_responses.shape[1]):
        plt.figure()
        plt.plot(plot_x, measured_stim_responses[:, i, neuron_to_stim_ind])
        plt.plot(plot_x, posterior_stim_responses[:, i, neuron_to_stim_ind])
        plt.plot(plot_x, prior_stim_responses[:, i, neuron_to_stim_ind])
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylim(ylim)
        plt.ylabel(cell_ids_chosen[i])
        plt.xlabel('time (s)')

        plt.legend(['measured', 'smoothed', 'model'])
        plt.title('average responses to stimulation of: ' + neuron_to_stim)

    plt.show()


def plot_dynamics_eigs(model):
    A = model.dynamics_weights.numpy()

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

