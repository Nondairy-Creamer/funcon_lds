import numpy as np
import analysis_utilities as au
import wormneuroatlas as wa
import matplotlib as mpl
from matplotlib import pyplot as plt
colormap = mpl.colormaps['coolwarm']
plot_percent = 95


def plot_model_params(model, model_true=None, cell_ids_chosen=None):
    model_params = model.get_params()
    if model_true is not None:
        has_ground_truth = True
    else:
        has_ground_truth = False

    if cell_ids_chosen is None:
        cell_ids_chosen = model.cell_ids.copy()

    if has_ground_truth:
        model_params_true = model_true.get_params()
    cell_ids = model.cell_ids
    # limit the matrix to the chosen neurons
    neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])

    # plot the log likelihood
    plt.figure()
    plt.plot(model.log_likelihood, label='fit parameters')
    if has_ground_truth:
        plt.axhline(model_true.log_likelihood[0], color='k', linestyle=':' , label='true parameters')
        plt.legend()
    plt.xlabel('EM iteration')
    plt.ylabel('log likelihood')
    plt.show()

    # plot the dynamics weights
    A_full = model_params['trained']['dynamics_weights'][:model.dynamics_dim, :]
    A = np.split(A_full, model.dynamics_lags, axis=1)
    A = [i[np.ix_(neuron_inds_chosen, neuron_inds_chosen)] for i in A]
    # get rid of the diagonal
    for aa in range(len(A)):
        A[aa][np.eye(A[aa].shape[0], dtype=bool)] = np.nan
    abs_max = np.nanmax(np.abs(A))

    if has_ground_truth:
        A_full_true = model_params_true['trained']['dynamics_weights'][:model.dynamics_dim, :]
        A_true = np.split(A_full_true, model.dynamics_lags, axis=1)
        A_true = [i[np.ix_(neuron_inds_chosen, neuron_inds_chosen)] for i in A_true]
        # get rid of the diagonal
        for aa in range(len(A_true)):
            A_true[aa][np.eye(A_true[aa].shape[0], dtype=bool)] = np.nan

        abs_max_true = np.nanmax(np.abs(A_true))
        abs_max = np.nanmax((abs_max, abs_max_true))
    else:
        A_true = [None for i in range(len(A))]

    for i in range(len(A)):
        plot_matrix(A[i], A_true[i], labels_x=cell_ids_chosen, labels_y=cell_ids_chosen, abs_max=abs_max, title='A')

    # plot the B matrix
    B_full = model_params['trained']['dynamics_input_weights'][:model.dynamics_dim, :]
    B = np.split(B_full, model.dynamics_input_lags, axis=1)
    B = np.stack([np.diag(i) for i in B]).T
    B = B[neuron_inds_chosen, :]

    if has_ground_truth:
        B_full_true = model_params_true['trained']['dynamics_input_weights'][:model.dynamics_dim, :]
        B_true = np.split(B_full_true, model.dynamics_input_lags, axis=1)
        B_true = np.stack([np.diag(i) for i in B_true]).T
        B_true = B_true[neuron_inds_chosen, :]
    else:
        B_true = None

    plot_matrix(B, B_true, labels_y=cell_ids_chosen, title='B')

    # Plot the Q matrix
    Q = model_params['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim]
    Q = Q[np.ix_(neuron_inds_chosen, neuron_inds_chosen)]

    if has_ground_truth:
        Q_true = model_params_true['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim]
        Q_true = Q_true[np.ix_(neuron_inds_chosen, neuron_inds_chosen)]
    else:
        Q_true = None

    plot_matrix(Q, Q_true, labels_x=cell_ids_chosen, labels_y=cell_ids_chosen, title='Q')

    # Plot the R matrix
    R = model_params['trained']['emissions_cov'][:model.dynamics_dim, :model.dynamics_dim]
    R = R[np.ix_(neuron_inds_chosen, neuron_inds_chosen)]

    if has_ground_truth:
        R_true = model_params_true['trained']['emissions_cov'][:model.dynamics_dim, :model.dynamics_dim]
        R_true = R_true[np.ix_(neuron_inds_chosen, neuron_inds_chosen)]
    else:
        R_true = None

    plot_matrix(R, R_true, labels_x=cell_ids_chosen, labels_y=cell_ids_chosen, title='R')


def plot_matrix(param_trained, param_true=None, labels_x=None, labels_y=None, abs_max=None, title=''):
    # plot a specific model parameter, usually called by plot_model_params

    if abs_max is None:
        if param_true is not None:
            abs_max = np.nanmax([np.nanmax(np.abs(i)) for i in [param_trained, param_true, param_true - param_trained]])
        else:
            abs_max = np.nanmax([np.nanmax(np.abs(i)) for i in [param_trained]])

    if param_true is not None:
        plt.subplot(2, 2, 1)
    plt.imshow(param_trained, interpolation='Nearest', cmap=colormap)
    plt.title('fit weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()

    if labels_x is not None:
        plt.xticks(np.arange(param_trained.shape[1]), labels_x)
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    if labels_y is not None:
        plt.yticks(np.arange(param_trained.shape[0]), labels_y)

    if param_true is not None:
        plt.subplot(2, 2, 2)
        plt.imshow(param_true, interpolation='Nearest', cmap=colormap)
        plt.title('true weights, ' + title)
        plt.clim((-abs_max, abs_max))
        plt.colorbar()

        if labels_x is not None:
            plt.xticks(np.arange(param_trained.shape[1]), labels_x)
            ax = plt.gca()
            for label in ax.get_xticklabels():
                label.set_rotation(90)

        if labels_y is not None:
            plt.yticks(np.arange(param_trained.shape[0]), labels_y)

        plt.subplot(2, 2, 3)
        plt.imshow(param_true - param_trained, interpolation='Nearest', cmap=colormap)
        plt.title('true - fit')
        plt.clim((-abs_max, abs_max))
        plt.colorbar()

        if labels_x is not None:
            plt.xticks(np.arange(param_trained.shape[1]), labels_x)
            ax = plt.gca()
            for label in ax.get_xticklabels():
                label.set_rotation(90)

        if labels_y is not None:
            plt.yticks(np.arange(param_trained.shape[0]), labels_y)

    plt.tight_layout()
    plt.show()


def plot_posterior(data, posterior_dict, cell_ids_chosen, sample_rate=0.5, window_size=1000):
    emissions = data['emissions']
    inputs = data['inputs']
    posterior = posterior_dict['posterior']
    post_pred = posterior_dict['post_pred_noise']

    neuron_inds_chosen = np.array([data['cell_ids'].index(i) for i in cell_ids_chosen])

    # get all the inputs but with only the chosen neurons
    inputs_truncated = [i[:, neuron_inds_chosen] for i in inputs]
    data_ind_chosen, time_window = au.find_stim_events(inputs_truncated, window_size=window_size)

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
    data_ind, time_window = au.find_stim_events(chosen_inputs, window_size=window_size)

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


def plot_stim_l2_norm(model, data, posterior_dict, cell_ids_chosen, window=(-60, 120), sub_pre_stim=True):
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
    measured_stim_responses = au.get_stim_response(emissions, inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=False)[0]
    post_pred_stim_responses = au.get_stim_response(post_pred, inputs, window=window, sub_pre_stim=sub_pre_stim, return_pre=False)[0]
    model_weights = model.dynamics_weights[:model.dynamics_dim, :]

    # calculate the rms for each response. This deals with the fact that responses can vary and go negative
    measured_response_norm = au.rms(measured_stim_responses, axis=0)
    post_pred_response_norm = au.rms(post_pred_stim_responses, axis=0)
    model_weights_norm = au.rms(au.stack_weights(model_weights, model.dynamics_lags, axis=1), axis=0)
    correlation = au.nan_corr_data(emissions)
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
    m_c = au.nancorrcoef([measured_response_norm.reshape(-1), correlation_masked.reshape(-1)])[0, 1]
    m_pp = au.nancorrcoef([measured_response_norm.reshape(-1), post_pred_masked.reshape(-1)])[0, 1]
    m_w = au.nancorrcoef([measured_response_norm.reshape(-1), model_weights_masked.reshape(-1)])[0, 1]

    # compare each of the weights against the model weights
    w_c = au.nancorrcoef([model_weights_norm.reshape(-1), correlation_masked.reshape(-1)])[0, 1]
    w_pp = au.nancorrcoef([model_weights_norm.reshape(-1), post_pred_masked.reshape(-1)])[0, 1]

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

    connect_corr_score = au.nancorrcoef([connectome_to_corr, correlation_no_nan])[0, 1]
    connect_weights_score = au.nancorrcoef([connectome_to_weights, weights_no_nan])[0, 1]

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
                       window=(-60, 120), sample_rate=0.5, sub_pre_stim=True, num_plot=5):
    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']
    post_pred = posterior_dict['post_pred']
    posterior = posterior_dict['posterior']

    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    plot_x = np.arange(window[0], window[1]) * sample_rate
    neuron_to_stim_ind = cell_ids_chosen.index(neuron_to_stim)

    # go through emissions and get the average input response for each neuron
    measured_stim_responses, measured_stim_responses_sem = au.get_stim_response(emissions, inputs, window=window, sub_pre_stim=sub_pre_stim)[:2]
    posterior_stim_responses = au.get_stim_response(posterior, inputs, window=window, sub_pre_stim=sub_pre_stim)[0]
    post_pred_stim_responses = au.get_stim_response(post_pred, inputs, window=window, sub_pre_stim=sub_pre_stim)[0]

    # pull out the neurons we care about
    measured_stim_responses = measured_stim_responses[:, chosen_neuron_inds, :]
    measured_stim_responses = measured_stim_responses[:, :, chosen_neuron_inds]
    measured_stim_responses_sem = measured_stim_responses_sem[:, chosen_neuron_inds, :]
    measured_stim_responses_sem = measured_stim_responses_sem[:, :, chosen_neuron_inds]
    measured_stim_responses_l2 = au.rms(measured_stim_responses, axis=0)
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

    ylim = (np.nanpercentile([measured_response_chosen - measured_response_sem_chosen, post_pred_response_chosen, posterior_response_chosen], 1),
            np.nanpercentile([measured_response_chosen + measured_response_sem_chosen, post_pred_response_chosen, posterior_response_chosen], 99))

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


def plot_model_comparison(sorting_param, model_list, posterior_train_list, data_train_list, posterior_test_list, data_test_list):
    num_non_nan_train = [np.sum([np.sum(~np.isnan(j)) for j in i['emissions']]) for i in data_train_list]
    num_non_nan_test = [np.sum([np.sum(~np.isnan(j)) for j in i['emissions']]) for i in data_test_list]

    sorting_list = [getattr(i, sorting_param) for i in model_list]
    ll_over_train = [getattr(i, 'log_likelihood') for i in model_list]
    train_ll = [i['ll'] for i in posterior_train_list]
    test_ll = [i['ll'] for i in posterior_test_list]

    # normalize likelihood by number of measurements
    ll_over_train = [ll_over_train[i] / num_non_nan_train[i] for i in range(len(ll_over_train))]
    train_ll = [train_ll[i] / num_non_nan_train[i] for i in range(len(train_ll))]
    test_ll = [test_ll[i] / num_non_nan_test[i] for i in range(len(test_ll))]

    sorting_list_inds = np.argsort(sorting_list)
    sorting_list = np.sort(sorting_list)
    ll_over_train = [ll_over_train[i] for i in sorting_list_inds]
    train_ll = [train_ll[i] for i in sorting_list_inds]
    test_ll = [test_ll[i] for i in sorting_list_inds]

    best_model_ind = np.argmax(test_ll)

    for ii, i in enumerate(ll_over_train):
        plt.figure()
        plt.plot(i)
        plt.title(sorting_param + ': ' + str(sorting_list[ii]))
        plt.xlabel('Iterations of EM')
        plt.ylabel('mean log likelihood')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(sorting_list, train_ll, label='train', marker='o')
    plt.xlabel(sorting_param)
    plt.ylabel('mean log likelihood')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(sorting_list, test_ll, label='test', marker='o')
    plt.xlabel(sorting_param)
    plt.grid()
    plt.tight_layout()

    plt.show()

    return best_model_ind

