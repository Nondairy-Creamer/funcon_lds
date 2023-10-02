import numpy as np
import analysis_utilities as au
import matplotlib as mpl
from matplotlib import pyplot as plt
colormap = mpl.colormaps['coolwarm']
plot_percent = 95
score_fun = au.nan_corr
# score_fun = au.nan_r2


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
        plt.axhline(model_true.log_likelihood[0], color='k', linestyle=':', label='true parameters')
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
    # model_sampled = posterior_dict['model_sampled_noise']
    model_sampled = posterior_dict['post_pred']

    neuron_inds_chosen = np.array([data['cell_ids'].index(i) for i in cell_ids_chosen])

    # get all the inputs but with only the chosen neurons
    inputs_truncated = [i[:, neuron_inds_chosen] for i in inputs]
    data_ind_chosen, time_window = au.find_stim_events(inputs_truncated, window_size=window_size)

    emissions_chosen = emissions[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    inputs_chosen = inputs[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    posterior_chosen = posterior[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    model_sampled_chosen = model_sampled[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]

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
    cmax = np.nanpercentile(np.abs((model_sampled_chosen, posterior_chosen)), plot_percent)

    plt.subplot(3, 1, 1)
    plt.imshow(inputs_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x * sample_rate)
    plt.xlabel('time (s)')
    plt.clim((-1, 1))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(model_sampled_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
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

    # check for data that has already been saved
    if 'posterior_missing' in posterior_dict.keys():
        if neuron_to_remove not in posterior_dict['posterior_missing']:
            force_calc = True
        else:
            if data_ind != posterior_dict['posterior_missing'][neuron_to_remove]['data_ind'] or \
                    time_window != posterior_dict['posterior_missing'][neuron_to_remove]['time_window']:
                force_calc = True
    else:
        force_calc = True

    emissions = data['emissions'][data_ind]
    inputs = data['inputs'][data_ind]
    init_mean = posterior_dict['init_mean'][data_ind]
    init_cov = posterior_dict['init_cov'][data_ind]
    sample_rate = model.sample_rate

    missing_neuron_ind = cell_ids.index(neuron_to_remove)

    if force_calc:
        emissions_missing = emissions.copy()
        emissions_missing[:, missing_neuron_ind] = np.nan
        posterior_missing = model.lgssm_smoother(emissions_missing, inputs, init_mean, init_cov)[1]
        posterior_missing = (posterior_missing @ model.emissions_weights.T)
        posterior_missing = posterior_missing[time_window[0]:time_window[1], missing_neuron_ind]
    else:
        posterior_missing = posterior_dict['posterior_missing'][neuron_to_remove]

    emissions = emissions[time_window[0]:time_window[1], missing_neuron_ind]
    inputs = inputs[time_window[0]:time_window[1], :]
    stim_time, stim_inds = np.where(inputs == 1)
    stim_names = [cell_ids[i] for i in stim_inds]
    recon_score, recon_score_ci = score_fun(emissions, posterior_missing)

    plot_x = np.arange(emissions.shape[0]) * sample_rate
    plt.figure()
    plt.plot(plot_x, emissions, label='measured')
    plt.plot(plot_x, posterior_missing, label='posterior, ' + neuron_to_remove + ' unmeasured')
    ylim = plt.ylim()
    for si, s in enumerate(stim_time):
        plt.axvline(plot_x[s], color=[0.5, 0.5, 0.5])
        plt.text(plot_x[s], ylim[1], stim_names[si], rotation=45)
    plt.xlabel('time (s)')
    plt.ylabel(neuron_to_remove + ' activity, correlation = ' + str(recon_score))
    plt.legend()
    plt.show()

    posterior_dict['posterior_missing'][neuron_to_remove] = {'posterior': posterior_missing,
                                                             'emissions': emissions,
                                                             'data_ind': data_ind,
                                                             'time_window': time_window,
                                                             }

    return posterior_dict


def plot_irf_norm(model, measured_irf, model_sampled_irf, data_corr, cell_ids_chosen):
    cell_ids = model.cell_ids
    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]

    model_weights = au.ave_fun(au.stack_weights(model.dynamics_weights[:model.dynamics_dim, :], model.dynamics_lags, axis=1), axis=0)

    # set the diagonal to nan to do stats on everything else
    data_corr[np.eye(data_corr.shape[0], dtype=bool)] = np.nan
    measured_irf[np.eye(data_corr.shape[0], dtype=bool)] = np.nan
    model_sampled_irf[np.eye(data_corr.shape[0], dtype=bool)] = np.nan
    model_weights[np.eye(data_corr.shape[0], dtype=bool)] = np.nan

    # plot each of the estimated weight matricies
    data_corr_plt = data_corr.copy()
    measured_response_norm_plt = measured_irf.copy()
    model_sampled_response_norm_plt = model_sampled_irf.copy()
    model_weights_norm_plt = model_weights.copy()

    data_corr_plt = data_corr_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    measured_response_norm_plt = measured_response_norm_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    model_sampled_response_norm_plt = model_sampled_response_norm_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    model_weights_norm_plt = model_weights_norm_plt[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]

    data_corr_plt = data_corr_plt / np.nanmax(data_corr_plt)
    measured_response_norm_plt = measured_response_norm_plt / np.nanmax(measured_response_norm_plt)
    model_sampled_response_norm_plt = model_sampled_response_norm_plt / np.nanmax(model_sampled_response_norm_plt)
    model_weights_norm_plt = model_weights_norm_plt / np.nanmax(model_weights_norm_plt)

    # plot all the impulse responses
    to_plot = ~np.all(np.isnan(measured_irf), axis=0) & ~np.all(np.isnan(model_sampled_irf), axis=0) & ~np.all(np.isnan(data_corr), axis=0)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(measured_irf[np.ix_(to_plot, to_plot)], interpolation='nearest')
    plt.subplot(2, 2, 2)
    plt.imshow(model_sampled_irf[np.ix_(to_plot, to_plot)], interpolation='nearest')
    plt.subplot(2, 2, 3)
    plt.imshow(data_corr[np.ix_(to_plot, to_plot)], interpolation='nearest')
    plt.subplot(2, 2, 4)
    plt.imshow(model_weights[np.ix_(to_plot, to_plot)], interpolation='nearest')
    plt.tight_layout()

    # plot the impulse responses of the chosen neurons
    plt.figure()
    plot_x = np.arange(len(chosen_neuron_inds))

    ax = plt.subplot(2, 2, 1)
    plt.imshow(measured_response_norm_plt, interpolation='nearest', cmap=colormap)
    plt.title('measured response')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 2)
    plt.imshow(model_sampled_response_norm_plt, interpolation='nearest', cmap=colormap)
    plt.title('post pred response')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 3)
    plt.imshow(data_corr_plt, interpolation='nearest', cmap=colormap)
    plt.title('correlation')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    ax = plt.subplot(2, 2, 4)
    plt.imshow(model_weights_norm_plt, interpolation='nearest', cmap=colormap)
    plt.title('model weights')
    plt.xticks(plot_x, cell_ids_chosen)
    plt.yticks(plot_x, cell_ids_chosen)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    plt.clim((-1, 1))

    plt.tight_layout()

    plt.show()


def compare_irf_w_anatomy(model, measured_irf, model_sampled_irf, data_corr, cell_ids_chosen):
    cell_ids = model.cell_ids
    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]

    # get the magnitude of the model weights so we can compare them without sign to the connectome
    model_weights = au.ave_fun(np.abs(au.stack_weights(model.dynamics_weights[:model.dynamics_dim, :], model.dynamics_lags, axis=1)), axis=0)
    measured_irf = np.abs(measured_irf)
    model_sampled_irf = np.abs(model_sampled_irf)
    data_corr = np.abs(data_corr)

    # set the diagonal to nan to do stats on everything else
    model_weights[np.eye(model_weights.shape[0], dtype=bool)] = np.nan
    measured_irf[np.eye(measured_irf.shape[0], dtype=bool)] = np.nan
    model_sampled_irf[np.eye(model_sampled_irf.shape[0], dtype=bool)] = np.nan
    data_corr[np.eye(data_corr.shape[0], dtype=bool)] = np.nan

    # shape the weights into a vector and remove nans
    model_weights_col = model_weights.reshape(-1)
    measured_irf_col = measured_irf.reshape(-1)
    model_sampled_irf_col = model_sampled_irf.reshape(-1)
    data_corr_col = data_corr.reshape(-1)

    mask = ~np.isnan(data_corr_col) & ~np.isnan(model_sampled_irf_col) & ~np.isnan(measured_irf_col)
    model_weights_col = model_weights_col[mask]
    measured_irf_col = measured_irf_col[mask]
    model_sampled_irf_col = model_sampled_irf_col[mask]
    data_corr_col = data_corr_col[mask]

    chem_conn, gap_conn, pep_conn = au.get_anatomical_data(cell_ids)

    chem_conn_col = chem_conn.reshape(-1)[mask]
    gap_conn_col = gap_conn.reshape(-1)[mask]
    pep_conn_col = pep_conn.reshape(-1)[mask]

    # get the best linear combination of the two connectomes to the correlation matrix and model weights
    ctome_stacked = np.stack((chem_conn_col, gap_conn_col, pep_conn_col)).T
    ctome_model_weights = ctome_stacked @ np.linalg.lstsq(ctome_stacked, model_weights_col)[0]
    ctome_measured_irf = ctome_stacked @ np.linalg.lstsq(ctome_stacked, measured_irf_col)[0]
    ctome_model_sampled_irf = ctome_stacked @ np.linalg.lstsq(ctome_stacked, model_sampled_irf_col)[0]
    ctome_data_corr = ctome_stacked @ np.linalg.lstsq(ctome_stacked, data_corr_col)[0]

    # compare each of the weights against measured
    model_weights_score, model_weights_score_ci = score_fun(ctome_model_weights, model_weights_col)
    measured_irf_score, measured_irf_score_ci = score_fun(ctome_measured_irf, measured_irf_col)
    model_sampled_irf_score, model_sampled_irf_score_ci = score_fun(ctome_model_sampled_irf, model_sampled_irf_col)
    data_corr_score, data_corr_score_ci = score_fun(ctome_data_corr, data_corr_col)

    # make figure comparing the metrics to the connectome
    plt.figure()
    plot_x = np.arange(4)
    y_val = np.array([model_weights_score, measured_irf_score, model_sampled_irf_score, data_corr_score])
    y_val_ci = np.abs(np.stack([model_weights_score_ci, measured_irf_score_ci, model_sampled_irf_score_ci, data_corr_score_ci]).T - y_val)
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='.', color='k')
    plt.xticks(plot_x, ['model weights', 'measured irf', 'post pred', 'data corr'])
    plt.xlabel('correlation to connectome')

    plt.show()


def compare_irf_w_prediction(model, measured_irf, model_sampled_irf, data_corr, cell_ids_chosen):
    model_weights = au.ave_fun(au.stack_weights(model.dynamics_weights[:model.dynamics_dim, :], model.dynamics_lags, axis=1), axis=0)
    cell_ids = model.cell_ids
    neuron_inds_chosen = [cell_ids.index(i) for i in cell_ids_chosen]

    # set the diagonal to nan to do stats on everything else
    data_corr[np.eye(data_corr.shape[0], dtype=bool)] = np.nan
    measured_irf[np.eye(measured_irf.shape[0], dtype=bool)] = np.nan
    model_sampled_irf[np.eye(model_sampled_irf.shape[0], dtype=bool)] = np.nan
    model_weights[np.eye(model_weights.shape[0], dtype=bool)] = np.nan

    measured_irf_chosen = measured_irf[np.ix_(neuron_inds_chosen, neuron_inds_chosen)]
    model_sampled_irf_chosen = model_sampled_irf[np.ix_(neuron_inds_chosen, neuron_inds_chosen)]
    data_corr_chosen = data_corr[np.ix_(neuron_inds_chosen, neuron_inds_chosen)]

    # scatter the interactions just between the chosen neurons
    model_sampled_score, model_sampled_score_ci = score_fun(measured_irf_chosen, model_sampled_irf_chosen)
    data_corr_score, data_corr_score_ci = score_fun(measured_irf_chosen, data_corr_chosen)

    plt.figure()
    plt.scatter(measured_irf_chosen.reshape(-1), model_sampled_irf_chosen.reshape(-1))
    xlim = plt.xlim()
    plt.plot(xlim, xlim)
    plt.title('post pred, ' + str(model_sampled_score))
    plt.ylabel('post pred irf')
    plt.xlabel('measured irf')

    plt.figure()
    plt.scatter(measured_irf_chosen.reshape(-1), data_corr_chosen.reshape(-1))
    plt.ylabel('data correlation')
    plt.xlabel('measured_irf')
    plt.title('data corr, ' + str(data_corr_score))

    # get a mask of where the data correlation and the measured irf is nan so that we compare on all the same data
    mask = ~np.isnan(data_corr) & ~np.isnan(model_sampled_irf) & ~np.isnan(measured_irf)
    mask_column = mask.reshape(-1)
    measured_irf_col = measured_irf.reshape(-1)[mask_column]
    model_sampled_irf_col = model_sampled_irf.reshape(-1)[mask_column]
    model_weights_col = model_weights.reshape(-1)[mask_column]
    data_corr_col = data_corr.reshape(-1)[mask_column]

    data_corr_score, data_corr_score_ci = score_fun(measured_irf_col, data_corr_col)
    model_sampled_score, model_sampled_score_ci = score_fun(measured_irf_col, model_sampled_irf_col)
    model_weights_score, model_weights_score_ci = score_fun(measured_irf_col, model_weights_col)

    # compare the metrics to the measured IRF
    plt.figure()
    plot_x = np.arange(3)
    y_val = np.array([data_corr_score, model_sampled_score, model_weights_score])
    y_val_ci = np.abs(np.stack([data_corr_score_ci, model_sampled_score_ci, model_weights_score_ci]).T - y_val)
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, yerr=y_val_ci, fmt='.', color='k')
    plt.xticks(plot_x, ['data correlation', 'model sampled', 'model weights'])
    plt.ylabel('correlation to impulse responses')

    # show the scatter plots for the comparison to IRF
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(measured_irf_col, data_corr_col, s=2)
    plt.ylabel('data correlation')
    plt.xlabel('measured impulse response')
    plt.subplot(2, 2, 2)
    plt.scatter(measured_irf_col, model_sampled_irf_col, s=2)
    xlim = plt.xlim()
    plt.plot(xlim, xlim, color='k')
    plt.ylabel('model impulse responses')
    plt.xlabel('measured impulse response')
    plt.subplot(2, 2, 3)
    plt.scatter(model_sampled_irf_col, data_corr_col, s=2)
    plt.xlabel('model impulse responses')
    plt.ylabel('data correlation')
    plt.tight_layout()

    plt.show()


def plot_irf_traces(measured_irf, measured_irf_sem, posterior_irf, model_sampled_irf, cell_ids, cell_ids_chosen, window,
                    sample_rate=0.5, num_plot=5):

    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    plot_x = np.arange(window[0], window[1]) * sample_rate

    # pull out the neurons we care about
    measured_irf = measured_irf[:, chosen_neuron_inds, :][:, :, chosen_neuron_inds]
    measured_irf_sem = measured_irf_sem[:, chosen_neuron_inds, :][:, :, chosen_neuron_inds]
    model_sampled_irf = model_sampled_irf[:, chosen_neuron_inds, :][:, :, chosen_neuron_inds]
    posterior_irf = posterior_irf[:, chosen_neuron_inds, :][:, :, chosen_neuron_inds]

    # find the 5 highest responses to plot
    measured_stim_responses_l2 = au.ave_fun(model_sampled_irf, axis=0)
    measured_stim_responses_l2[np.eye(measured_stim_responses_l2.shape[0], dtype=bool)] = 0
    sorted_vals = np.sort(measured_stim_responses_l2.reshape(-1))
    plot_inds = []
    for m in range(num_plot):
        best = np.where(measured_stim_responses_l2 == sorted_vals[-(m+1)])
        plot_inds.append((best[0][0], best[1][0]))

    measured_response_chosen = np.zeros((measured_irf.shape[0], num_plot))
    measured_response_sem_chosen = np.zeros((measured_irf.shape[0], num_plot))
    model_sampled_response_chosen = np.zeros((measured_irf.shape[0], num_plot))
    posterior_response_chosen = np.zeros((measured_irf.shape[0], num_plot))

    plot_label = []
    for pi, p in enumerate(plot_inds):
        measured_response_sem_chosen[:, pi] = measured_irf_sem[:, p[0], p[1]]
        measured_response_chosen[:, pi] = measured_irf[:, p[0], p[1]]
        model_sampled_response_chosen[:, pi] = model_sampled_irf[:, p[0], p[1]]
        posterior_response_chosen[:, pi] = posterior_irf[:, p[0], p[1]]
        plot_label.append((cell_ids_chosen[p[0]], cell_ids_chosen[p[1]]))

    ylim = (np.nanpercentile([measured_response_chosen - measured_response_sem_chosen, model_sampled_response_chosen, posterior_response_chosen], 1),
            np.nanpercentile([measured_response_chosen + measured_response_sem_chosen, model_sampled_response_chosen, posterior_response_chosen], 99))

    for i in range(measured_response_chosen.shape[1]):
        plt.figure()
        this_measured_resp = measured_response_chosen[:, i]
        this_measured_resp_sem = measured_response_sem_chosen[:, i]
        plt.plot(plot_x, this_measured_resp, label='measured')
        plt.fill_between(plot_x, this_measured_resp - this_measured_resp_sem, this_measured_resp + this_measured_resp_sem, alpha=0.4)
        plt.plot(plot_x, posterior_response_chosen[:, i], label='posterior')
        plt.plot(plot_x, model_sampled_response_chosen[:, i], label='posterior predictive')
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylim(ylim)
        plt.ylabel(plot_label[i][0])
        plt.xlabel('time (s)')

        plt.legend()
        plt.title('average responses to stimulation of: ' + plot_label[i][1])

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


def plot_model_comparison(sorting_param, model_list, posterior_train_list, data_train_list, posterior_test_list,
                          data_test_list, plot_figs=True, best_model_ind=None):
    num_non_nan_train = [np.sum([np.sum(~np.isnan(j)) for j in i['emissions']]) for i in data_train_list]
    num_non_nan_test = [np.sum([np.sum(~np.isnan(j)) for j in i['emissions']]) for i in data_test_list]

    sorting_list = [getattr(i, sorting_param) for i in model_list]
    for i in range(len(sorting_list)):
        if sorting_list[i] is None:
            sorting_list[i] = -np.inf
    ll_over_train = [getattr(i, 'log_likelihood') for i in model_list]
    train_ll = [i['ll'] for i in posterior_train_list]
    test_ll = [i['ll'] for i in posterior_test_list]

    # normalize likelihood by number of measurements
    ll_over_train = [ll_over_train[i] / num_non_nan_train[i] for i in range(len(ll_over_train))]
    train_ll = [train_ll[i] / num_non_nan_train[i] for i in range(len(train_ll))]
    test_ll = [test_ll[i] / num_non_nan_test[i] for i in range(len(test_ll))]

    if best_model_ind is None:
        best_model_ind = np.argmax(test_ll)

    if plot_figs:
        sorting_list_inds = np.argsort(sorting_list)
        sorting_list = np.sort(sorting_list)
        ll_over_train = [ll_over_train[i] for i in sorting_list_inds]
        train_ll = [train_ll[i] for i in sorting_list_inds]
        test_ll = [test_ll[i] for i in sorting_list_inds]

        for ii, i in enumerate(ll_over_train):
            plt.figure()
            plt.plot(i)
            plt.title(sorting_param + ': ' + str(sorting_list[ii]))
            plt.xlabel('Iterations of EM')
            plt.ylabel('mean log likelihood')

        if sorting_list[0] == -np.inf:
            sorting_list[0] = 2 * sorting_list[1] - sorting_list[2]
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

