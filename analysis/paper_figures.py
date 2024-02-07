from matplotlib import pyplot as plt
import numpy as np
import wormneuroatlas as wa
import metrics as met
import lgssm_utilities as ssmu
import matplotlib as mpl
import analysis_utilities as au
colormap = mpl.colormaps['coolwarm']


def weight_prediction(weights, weight_name, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_irms_score = []
    model_irms_score_ci = []

    # get the baseline correlation and IRMs. This is the best the model could have done
    # across all data
    irms_baseline = met.nan_corr(weights['data']['train'][weight_name], weights['data']['test'][weight_name])[0]

    # get the comparison between model prediction and data irm/correlation
    for m in ['synap', 'synap_randC']:
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
        met.nan_corr(weights['models'][m][weight_name], weights['data']['test'][weight_name])
        model_irms_score.append(model_irms_to_measured_irms_test)
        model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val / irms_baseline)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels'], rotation=45)
    # plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'], rotation=45)
    plt.ylabel('explainable correlation to measured + ' + weight_name)
    plt.tight_layout()

    plt.savefig(fig_save_path / ('measured_vs_model_' + weight_name + '.pdf'))

    model_irms_score = []
    model_irms_score_ci = []

    for m in ['synap', 'unconstrained', 'synap_randA']:
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
        met.nan_corr(weights['models'][m][weight_name], weights['data']['test'][weight_name])
        model_irms_score.append(model_irms_to_measured_irms_test)
        model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val / irms_baseline)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    # plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'], rotation=45)
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained', 'model\n+ scrambled anatomy'], rotation=45)
    plt.ylabel('explainable correlation to measured + ' + weight_name)
    plt.tight_layout()

    plt.savefig(fig_save_path / ('measured_vs_model_' + weight_name + '.pdf'))

    plt.show()


def weight_prediction_sweep(weights, masks, weight_name, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data

    # compare data corr and data IRM to connectome
    n_stim_mask = masks['n_stim_mask']
    n_stim_sweep = masks['n_stim_sweep']

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_name = []
    model_irms_score_sweep = []
    model_irms_score_sweep_ci = []

    # sweep through the minimum number of stimulations allowed and calculate the score
    # as the number of required stimulations goes up the quality of correlation goes up
    irms_baseline_sweep = np.zeros(n_stim_sweep.shape[0])
    irms_baseline_sweep_ci = np.zeros((2, n_stim_sweep.shape[0]))

    # sweep across number of stims, for the IRF data
    for ni, n in enumerate(n_stim_sweep):
        data_train_irms = weights['data']['train'][weight_name].copy()
        data_test_irms = weights['data']['test'][weight_name].copy()

        data_train_irms[n_stim_mask[ni]] = np.nan
        data_test_irms[n_stim_mask[ni]] = np.nan

        irms_baseline_sweep[ni], irms_baseline_sweep_ci[:, ni] = met.nan_corr(data_train_irms, data_test_irms)

    # get the comparison between model prediction and data irm/correlation
    for m in weights['models']:
        # if m in ['synap', 'synap_randC', 'synap_randA']:
        if m in ['synap', 'synap_randC']:
            # for each model, calculate its score for both corr and IRM reconstruction across the n stim sweep
            model_name.append(m)
            model_irms_score_sweep.append(np.zeros(n_stim_sweep.shape[0]))
            model_irms_score_sweep_ci.append(np.zeros((2, n_stim_sweep.shape[0])))

            for ni, n in enumerate(n_stim_sweep):
                # mask the model predicted correlations and IRMs based on how many stimulation events were observed
                model_irms = weights['models'][m][weight_name].copy()
                model_irms[n_stim_mask[ni]] = np.nan

                model_irms_to_measured_irms, model_irms_to_measured_irms_ci = \
                    met.nan_corr(model_irms, weights['data']['test'][weight_name])
                model_irms_score_sweep[-1][ni] = model_irms_to_measured_irms
                model_irms_score_sweep_ci[-1][:, ni] = model_irms_to_measured_irms_ci

    # plot model reconstruction of IRMs
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(n_stim_sweep, irms_baseline_sweep, irms_baseline_sweep_ci, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs, mcs_ci, label=n)
    plt.xlabel('# of stimulation events')
    plt.ylabel('similarity to measured ' + weight_name)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(0, 0)
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs / irms_baseline_sweep, mcs_ci / irms_baseline_sweep, label=n)
    plt.xlabel('# of stimulation events')
    plt.ylabel('similarity to measured ' + weight_name)
    plt.tight_layout()
    plt.savefig(fig_save_path / ('measured_vs_model_' + weight_name + '_over_n.pdf'))

    plt.show()


def weights_vs_connectome(weights, masks, metric=met.f_measure, rng=np.random.default_rng(), fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans'], masks['corr_nans'], combine_masks=True)

    model_weights_conn, model_weights_conn_ci = met.metric_ci(metric, masks['synap'], weights['models']['synap']['dirms_binarized'], rng=rng)
    data_corr_conn, data_corr_conn_ci = met.metric_ci(metric, masks['synap'], weights['data']['train']['corr_binarized'], rng=rng)
    data_irm_conn, data_irm_conn_ci = met.metric_ci(metric, masks['synap'], weights['data']['train']['q'], rng=rng)
    conn_null = met.metric_null(metric, masks['synap'])

    # plot model weight similarity to connectome
    plt.figure()
    y_val = np.array([model_weights_conn, data_corr_conn, data_irm_conn])
    y_val_ci = np.stack([model_weights_conn_ci, data_corr_conn_ci, data_irm_conn_ci]).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.axhline(conn_null, color='k', linestyle='--')
    plt.xticks(plot_x, labels=['model weights', 'data correlation', 'data IRMs'], rotation=45)
    plt.ylabel('similarity to connectome')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'weights_vs_connectome.pdf')

    plt.show()


def plot_eigenvalues(models, cell_ids):
    # calculate the eignvalues / vectors of the dynamics matrix
    model = models['synap']
    A = model.dynamics_weights
    eig_vals, eig_vects = np.linalg.eig(A)

    sort_inds = np.argsort(np.abs(eig_vals))[::-1]

    eig_vals = eig_vals[sort_inds]
    eig_vects = eig_vects[:, sort_inds]
    eig_vects = np.stack(np.split(eig_vects, model.dynamics_lags, axis=0))
    eig_vects = np.mean(np.abs(eig_vects), axis=0)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(eig_vals))
    plt.xlabel('eigenvalue')

    plt.subplot(1, 2, 2)
    plt.plot(np.abs(eig_vals[:100]))
    plt.xlabel('eigenvalue')

    plt.figure()
    plt.scatter(np.real(eig_vals), np.imag(eig_vals))
    plt.xlabel('real[eigenvalues]')
    plt.ylabel('imag[eigenvalues]')

    ylim_max = np.max(eig_vects)
    ylim_plot = (0, ylim_max)

    num_eigvects_to_plot = 20
    num_eigvect_inds_to_plot = 20
    plot_x = np.arange(num_eigvect_inds_to_plot)

    for n in range(num_eigvects_to_plot):
        cell_ids_plot = cell_ids['all'].copy()
        this_eig_vect = eig_vects[:, n]
        cell_sort_inds = np.argsort(this_eig_vect)[::-1]
        this_eig_vect = this_eig_vect[cell_sort_inds]
        cell_ids_plot = [cell_ids_plot[i] for i in cell_sort_inds[:num_eigvect_inds_to_plot]]

        plt.figure()
        plt.scatter(plot_x, this_eig_vect[:num_eigvect_inds_to_plot])
        plt.ylim(ylim_plot)
        plt.xticks(plot_x, cell_ids_plot, rotation=90)

    plt.show()

    a=1


def unconstrained_vs_constrained_model(weights, fig_save_path=None):
    data_corr = weights['data']['test']['corr']
    data_irms = weights['data']['test']['irms']

    # compare unconstrained and constrained model IRMs to measured IRMs
    model_corr_score = []
    model_corr_score_ci = []
    model_irms_score = []
    model_irms_score_ci = []
    for m in weights['models']:
        if m in ['synap', 'unconstrained']:
            model_corr_to_measured_corr, model_corr_to_measured_corr_ci = met.nan_corr(weights['models'][m]['corr'], data_corr)
            model_corr_score.append(model_corr_to_measured_corr)
            model_corr_score_ci.append(model_corr_to_measured_corr_ci)

            model_irms_to_measured_irms, model_irms_to_measured_irms_ci = met.nan_corr(weights['models'][m]['irms'], data_irms)
            model_irms_score.append(model_irms_to_measured_irms)
            model_irms_score_ci.append(model_irms_to_measured_irms_ci)

    irms_baseline = met.nan_corr(weights['data']['train']['irms'], weights['data']['test']['irms'])[0]

    # plot model reconstruction of IRMs
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.subplot(1, 2, 1)
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.axhline(irms_baseline, color='k', linestyle='--')
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained'], rotation=45)
    plt.ylabel('similarity to measured IRMs')

    plt.subplot(1, 2, 2)
    plt.bar(plot_x, y_val / irms_baseline)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained'], rotation=45)
    plt.ylabel('normalized similarity to measured IRMs')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'sim_to_irm_uncon.pdf')

    plt.show()


def plot_irms(weights, cell_ids, use_chosen_ids=False, fig_save_path=None):
    plot_x = np.arange(len(cell_ids['all']))
    plot_x = np.arange(len(cell_ids['chosen']))
    font_size = 8

    model_name = 'synap_randA'
    neurons_to_mask = ['AIAL']
    model_weights = weights['models'][model_name]['eirms'].copy()
    model_irms = weights['models'][model_name]['irms'].copy()
    model_corr = weights['models'][model_name]['corr'].copy()
    data_irms = weights['data']['train']['irms'].copy()
    data_corr = weights['data']['train']['corr'].copy()

    model_weights[np.eye(model_weights.shape[0], dtype=bool)] = np.nan
    model_irms[np.eye(model_irms.shape[0], dtype=bool)] = np.nan
    model_corr[np.eye(model_corr.shape[0], dtype=bool)] = np.nan
    data_irms[np.eye(data_irms.shape[0], dtype=bool)] = np.nan
    data_corr[np.eye(data_corr.shape[0], dtype=bool)] = np.nan

    for n in neurons_to_mask:
        neuron_ind = cell_ids['all'].index(n)
        model_weights[neuron_ind, :] = 0
        model_weights[neuron_ind, neuron_ind] = np.nan
        model_irms[neuron_ind, :] = 0
        model_irms[neuron_ind, neuron_ind] = np.nan

    weight_lims = np.nanmax(np.abs(model_weights))
    irm_lims = np.nanmax(np.abs((model_irms, data_irms)))
    corr_lims = np.nanmax(np.abs((model_corr, data_corr)))

    # get the neurons to plot
    if use_chosen_ids:
        neuron_inds = [cell_ids['all'].index(i) for i in cell_ids['chosen']]
        model_weights = model_weights[np.ix_(neuron_inds, neuron_inds)]
        model_irms = model_irms[np.ix_(neuron_inds, neuron_inds)]
        model_corr = model_corr[np.ix_(neuron_inds, neuron_inds)]

        data_irms = data_irms[np.ix_(neuron_inds, neuron_inds)]
        data_corr = data_corr[np.ix_(neuron_inds, neuron_inds)]
        cell_ids_plot = cell_ids['chosen']
    else:
        cell_ids_plot = cell_ids['all']

    plt.figure()
    plt.imshow(model_weights, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    color_limits = np.nanmax(np.abs(model_weights))
    color_limits = (-color_limits, color_limits)
    plt.clim(color_limits)
    plt.savefig(fig_save_path / 'full_model_weights.pdf')

    plt.figure()
    plt.imshow(model_irms, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    color_limits = np.nanmax(np.abs(model_irms))
    color_limits = (-color_limits, color_limits)
    plt.clim(color_limits)
    plt.savefig(fig_save_path / 'full_model_irms.pdf')

    plt.figure()
    plt.imshow(model_corr, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    color_limits = np.nanmax(np.abs(model_corr))
    color_limits = (-color_limits, color_limits)
    plt.clim(color_limits)
    plt.savefig(fig_save_path / 'full_model_corr.pdf')

    plt.figure()
    plt.imshow(data_irms, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    color_limits = np.nanmax(np.abs(data_irms))
    color_limits = (-color_limits, color_limits)
    plt.clim(color_limits)
    plt.savefig(fig_save_path / 'full_data_irms.pdf')

    plt.figure()
    plt.imshow(data_corr, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    color_limits = np.nanmax(np.abs(data_corr))
    color_limits = (-color_limits, color_limits)
    plt.clim(color_limits)
    plt.savefig(fig_save_path / 'full_data_corr.pdf')

    plt.show()


def plot_irfs(weights, masks, cell_ids, window, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs = ssmu.remove_nan_irfs(weights, cell_ids, chosen_mask=None)

    data_irfs = no_nan_irfs['data_irfs']
    data_irfs_sem = no_nan_irfs['data_irfs_sem']
    model_irfs = no_nan_irfs['model_irfs']
    model_dirfs = no_nan_irfs['model_dirfs']
    model_irms = no_nan_irfs['model_irms']
    cell_ids = no_nan_irfs['cell_ids']

    # get the highest model dirm vs model irm diff
    irm_dirm_mag = np.abs(model_irms)
    irm_dirm_mag_inds = np.argsort(irm_dirm_mag)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs.shape[0])

    for ii, i in enumerate(range(num_plot)):
        plt.figure()
        plot_ind = irm_dirm_mag_inds[i]
        this_irf = data_irfs[:, plot_ind]
        this_irf_sem = data_irfs_sem[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity')

        plt.savefig(fig_save_path / ('trace_' + str(ii)))

    plt.show()


def plot_irfs_train_test(weights, masks, cell_ids, window, chosen_mask=None, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs_train = ssmu.remove_nan_irfs(weights, cell_ids, data_type='train', chosen_mask=chosen_mask)
    no_nan_irfs_test = ssmu.remove_nan_irfs(weights, cell_ids, data_type='test', chosen_mask=chosen_mask)

    data_irfs_train = no_nan_irfs_train['data_irfs']
    data_irfs_sem_train = no_nan_irfs_test['data_irfs_sem']
    data_irfs_test = no_nan_irfs_test['data_irfs']
    data_irfs_sem_test = no_nan_irfs_test['data_irfs_sem']
    model_irfs = no_nan_irfs_test['model_irfs']
    model_irms = no_nan_irfs_test['model_irms']
    cell_ids = no_nan_irfs_test['cell_ids']

    # get the highest model dirm vs model irm diff
    irm_dirm_mag = np.abs(model_irms)
    irm_dirm_mag_inds = np.argsort(irm_dirm_mag)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs_test.shape[0])

    for i in range(num_plot):
        plot_ind = irm_dirm_mag_inds[i]

        plt.figure()
        plt.subplot(2, 1, 1)
        this_irf = data_irfs_train[:, plot_ind]
        this_irf_sem = data_irfs_sem_train[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylabel('cell activity (train set)')

        plt.subplot(2, 1, 2)
        this_irf = data_irfs_test[:, plot_ind]
        this_irf_sem = data_irfs_sem_test[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity (test set)')

    plt.show()


def plot_dirfs(weights, masks, cell_ids, window, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs = ssmu.remove_nan_irfs(weights, cell_ids, chosen_mask=masks['synap'])

    data_irfs = no_nan_irfs['data_irfs']
    data_irfs_sem = no_nan_irfs['data_irfs_sem']
    model_irfs = no_nan_irfs['model_irfs']
    model_dirfs = no_nan_irfs['model_dirfs']
    model_eirfs = no_nan_irfs['model_eirfs']
    model_irms = no_nan_irfs['model_irms']
    cell_ids = no_nan_irfs['cell_ids']

    # get the highest model dirm vs model irm diff
    irm_dirm_mag = np.abs(model_irms)
    irm_dirm_mag_inds = np.argsort(irm_dirm_mag)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs.shape[0])

    for i in range(num_plot):
        plt.figure()
        plot_ind = irm_dirm_mag_inds[i]
        this_irf = data_irfs[:, plot_ind]
        this_irf_sem = data_irfs_sem[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        # plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity')

    plt.show()


def plot_dirfs_train_test(weights, masks, cell_ids, window, chosen_mask=None, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs_train = ssmu.remove_nan_irfs(weights, cell_ids, data_type='train', chosen_mask=chosen_mask)
    no_nan_irfs_test = ssmu.remove_nan_irfs(weights, cell_ids, data_type='test', chosen_mask=chosen_mask)

    data_irfs_train = no_nan_irfs_train['data_irfs']
    data_irfs_sem_train = no_nan_irfs_test['data_irfs_sem']
    data_irfs_test = no_nan_irfs_test['data_irfs']
    data_irfs_sem_test = no_nan_irfs_test['data_irfs_sem']
    model_irfs = no_nan_irfs_test['model_irfs']
    model_dirfs = no_nan_irfs_test['model_dirfs']
    model_rdirfs = no_nan_irfs_test['model_rdirfs']
    model_eirfs = no_nan_irfs_test['model_eirfs']
    model_irms = no_nan_irfs_test['model_irms']
    cell_ids = no_nan_irfs_test['cell_ids']

    # get the highest model dirm vs model irm diff
    irm_dirm_mag = np.abs(model_irms)
    irm_dirm_mag_inds = np.argsort(irm_dirm_mag)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs_test.shape[0])

    for i in range(num_plot):
        plot_ind = irm_dirm_mag_inds[i]

        plt.figure()
        plt.subplot(2, 1, 1)
        this_irf = data_irfs_train[:, plot_ind]
        this_irf_sem = data_irfs_sem_train[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        plt.plot(plot_x, model_rdirfs[:, plot_ind], label='model rdirf')
        plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylabel('cell activity (train set)')

        plt.subplot(2, 1, 2)
        this_irf = data_irfs_test[:, plot_ind]
        this_irf_sem = data_irfs_sem_test[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        plt.plot(plot_x, model_rdirfs[:, plot_ind], label='model rdirf')
        plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity (test set)')

    plt.show()


def plot_dirfs_train_test_swap(weights, masks, cell_ids, window, chosen_mask=None, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs_train = ssmu.remove_nan_irfs(weights, cell_ids, data_type='train', chosen_mask=chosen_mask)
    no_nan_irfs_test = ssmu.remove_nan_irfs(weights, cell_ids, data_type='test', chosen_mask=chosen_mask)

    # find places where model dirf and irf flip sign
    pos_to_neg = (no_nan_irfs_test['model_dirms'] > 0) & (no_nan_irfs_test['model_irms'] < 0)
    neg_to_pos = (no_nan_irfs_test['model_dirms'] < 0) & (no_nan_irfs_test['model_irms'] > 0)
    swapped = pos_to_neg | neg_to_pos

    data_irfs_train = no_nan_irfs_train['data_irfs'][:, swapped]
    data_irfs_sem_train = no_nan_irfs_test['data_irfs_sem'][:, swapped]
    data_irfs_test = no_nan_irfs_test['data_irfs'][:, swapped]
    data_irfs_sem_test = no_nan_irfs_test['data_irfs_sem'][:, swapped]
    model_irfs = no_nan_irfs_test['model_irfs'][:, swapped]
    model_dirfs = no_nan_irfs_test['model_dirfs'][:, swapped]
    model_rdirfs = no_nan_irfs_test['model_rdirfs'][:, swapped]
    model_eirfs = no_nan_irfs_test['model_eirfs'][:, swapped]
    model_irms = no_nan_irfs_test['model_irms'][swapped]
    model_dirms = no_nan_irfs_test['model_dirms'][swapped]
    cell_ids = no_nan_irfs_test['cell_ids'][:, swapped]

    # get the highest model dirm vs model irm diff
    irm_dirm_mag = np.abs(model_irms - model_dirms)
    irm_dirm_mag_inds = np.argsort(irm_dirm_mag)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs_test.shape[0])

    for i in range(num_plot):
        plot_ind = irm_dirm_mag_inds[i]

        plt.figure()
        plt.subplot(2, 1, 1)
        this_irf = data_irfs_train[:, plot_ind]
        this_irf_sem = data_irfs_sem_train[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        plt.plot(plot_x, model_rdirfs[:, plot_ind], label='model rdirf')
        plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylabel('cell activity (train set)')

        plt.subplot(2, 1, 2)
        this_irf = data_irfs_test[:, plot_ind]
        this_irf_sem = data_irfs_sem_test[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        plt.plot(plot_x, model_rdirfs[:, plot_ind], label='model rdirf')
        plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity (test set)')

    plt.show()


def plot_dirfs_gt_irfs(weights, masks, cell_ids, window, chosen_mask=None, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs_train = ssmu.remove_nan_irfs(weights, cell_ids, data_type='train', chosen_mask=chosen_mask)
    no_nan_irfs_test = ssmu.remove_nan_irfs(weights, cell_ids, data_type='test', chosen_mask=chosen_mask)

    # find places where model dirf and irf flip sign
    dirfs_gt_irfs = no_nan_irfs_test['model_dirms'] > no_nan_irfs_test['model_irms']
    dirfs_gt_irfs = dirfs_gt_irfs & (no_nan_irfs_test['model_dirms'] > 0)

    data_irfs_train = no_nan_irfs_train['data_irfs'][:, dirfs_gt_irfs]
    data_irfs_sem_train = no_nan_irfs_test['data_irfs_sem'][:, dirfs_gt_irfs]
    data_irfs_test = no_nan_irfs_test['data_irfs'][:, dirfs_gt_irfs]
    data_irfs_sem_test = no_nan_irfs_test['data_irfs_sem'][:, dirfs_gt_irfs]
    model_irfs = no_nan_irfs_test['model_irfs'][:, dirfs_gt_irfs]
    model_dirfs = no_nan_irfs_test['model_dirfs'][:, dirfs_gt_irfs]
    model_rdirfs = no_nan_irfs_test['model_rdirfs'][:, dirfs_gt_irfs]
    model_eirfs = no_nan_irfs_test['model_eirfs'][:, dirfs_gt_irfs]
    model_irms = no_nan_irfs_test['model_irms'][dirfs_gt_irfs]
    model_dirms = no_nan_irfs_test['model_dirms'][dirfs_gt_irfs]
    cell_ids = no_nan_irfs_test['cell_ids'][:, dirfs_gt_irfs]

    # get the highest model dirm vs model irm diff
    irm_dirm_mag = np.abs(model_dirms)
    irm_dirm_mag_inds = np.argsort(irm_dirm_mag)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs_test.shape[0])

    for i in range(num_plot):
        plot_ind = irm_dirm_mag_inds[i]

        plt.figure()
        # plt.subplot(2, 1, 1)
        # this_irf = data_irfs_train[:, plot_ind]
        # this_irf_sem = data_irfs_sem_train[:, plot_ind]
        #
        # plt.plot(plot_x, this_irf, label='data irf')
        # plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)
        #
        # plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        # plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        # # plt.plot(plot_x, model_rdirfs[:, plot_ind], label='model rdirf')
        # # plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        # plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        # plt.legend()
        # plt.axvline(0, color='k', linestyle='--')
        # plt.axhline(0, color='k', linestyle='--')
        # plt.ylabel('cell activity (train set)')

        # plt.subplot(2, 1, 2)
        this_irf = data_irfs_test[:, plot_ind]
        this_irf_sem = data_irfs_sem_test[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        # plt.plot(plot_x, model_rdirfs[:, plot_ind], label='model rdirf')
        # plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity (test set)')

    plt.show()


def plot_dirm_diff(weights, masks, cell_ids, window, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs = ssmu.remove_nan_irfs(weights, cell_ids, chosen_mask=masks['synap'])

    data_irfs = no_nan_irfs['data_irfs']
    data_irfs_sem = no_nan_irfs['data_irfs_sem']
    model_irfs = no_nan_irfs['model_irfs']
    model_dirfs = no_nan_irfs['model_dirfs']
    model_rdirfs = no_nan_irfs['model_rdirfs']
    model_eirfs = no_nan_irfs['model_eirfs']
    model_irms = no_nan_irfs['model_irms']
    model_dirms = no_nan_irfs['model_dirms']
    cell_ids = no_nan_irfs['cell_ids']

    # get the highest model dirm vs model irm diff
    irm_dirm_mag = np.abs(model_irms - model_dirms)
    irm_dirm_mag_inds = np.argsort(irm_dirm_mag)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs.shape[0])

    for i in range(num_plot):
        plt.figure()
        plot_ind = irm_dirm_mag_inds[i]
        this_irf = data_irfs[:, plot_ind]
        this_irf_sem = data_irfs_sem[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        # plt.plot(plot_x, model_rdirfs[:, plot_ind], label='model rdirf')
        # plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity')

    plt.show()


def irm_vs_dirm(weights, masks, cell_ids):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    irms = weights['models']['synap']['irms']
    dirms = weights['models']['synap']['dirms']

    irm_dirm_ratio = irms / dirms
    irm_dirm_ratio_ave = np.nanmean(irm_dirm_ratio)
    irm_dirm_ratio_ave_sem = np.nanstd(irm_dirm_ratio, ddof=1) / np.sqrt(np.sum(~np.isnan(irm_dirm_ratio)))

    plt.figure()
    plt.bar(1, irm_dirm_ratio_ave)
    plt.errorbar(1, irm_dirm_ratio_ave, irm_dirm_ratio_ave_sem)
    plt.show()

    a=1


def predict_chem_synapse_sign(weights, masks, cell_ids, metric=met.accuracy, rng=np.random.default_rng(), fig_save_path=None):
    # get the connections associated with chem but not gap junctions
    # and the connections associated with gap but not chemical junctions
    chem_no_gap = ~masks['gap'] & masks['chem']

    model_synap_dirms_chem = weights['models']['synap']['eirms'][chem_no_gap]
    model_uncon_dirms_chem = weights['models']['unconstrained']['eirms'][chem_no_gap]
    data_irms_chem = weights['data']['test']['irms'][chem_no_gap]

    # binarize the synapses into greater than / less than 0
    # note that in python 3 > nan is False annoyingly. set it back to nan
    nan_loc_chem = np.isnan(model_synap_dirms_chem)
    model_synap_dirms_chem = (model_synap_dirms_chem > 0).astype(float)
    model_uncon_dirms_chem = (model_uncon_dirms_chem > 0).astype(float)
    data_irms_chem = (data_irms_chem > 0).astype(float)

    model_synap_dirms_chem[nan_loc_chem] = np.nan
    model_uncon_dirms_chem[nan_loc_chem] = np.nan
    data_irms_chem[nan_loc_chem] = np.nan

    # get the sign of the chemical synapses
    watlas = wa.NeuroAtlas()
    chem_sign_out = watlas.get_chemical_synapse_sign()

    cmplx = np.logical_and(np.any(chem_sign_out == -1, axis=0),
                           np.any(chem_sign_out == 1, axis=0))
    chem_sign = np.nansum(chem_sign_out, axis=0)
    chem_sign[cmplx] = 0
    chem_mask = chem_sign == 0

    chem_sign[chem_sign > 0] = 1
    chem_sign[chem_sign < 0] = 0
    chem_sign[chem_mask] = np.nan

    atlas_ids = list(watlas.neuron_ids)
    atlas_ids[atlas_ids.index('AWCON')] = 'AWCL'
    atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCR'
    cell_inds = np.array([atlas_ids.index(i) for i in cell_ids['all']])
    chem_sign = chem_sign[np.ix_(cell_inds, cell_inds)]
    chem_sign[masks['irm_nans']] = np.nan
    chem_sign = chem_sign[chem_no_gap]

    # prediction accuracy
    chem_sign_predict_model_synap, chem_sign_predict_model_synap_ci = met.metric_ci(metric, chem_sign, model_synap_dirms_chem, rng=rng)
    chem_sign_predict_model_uncon, chem_sign_predict_model_uncon_ci = met.metric_ci(metric, chem_sign, model_uncon_dirms_chem, rng=rng)
    chem_sign_predict_data_dirms, chem_sign_predict_data_dirms_ci = met.metric_ci(metric, chem_sign, data_irms_chem, rng=rng)
    chem_sign_predict_null = met.metric_null(metric, chem_sign)

    plt.figure()
    y_val = np.array([chem_sign_predict_model_synap, chem_sign_predict_model_uncon, chem_sign_predict_data_dirms])
    y_val_ci = np.stack([chem_sign_predict_model_synap_ci, chem_sign_predict_model_uncon_ci, chem_sign_predict_data_dirms_ci]).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.axhline(chem_sign_predict_null, color='k', linestyle='--')
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained', 'data IRMs'], rotation=45)
    plt.ylabel('% correct')
    plt.title('similarity to known synapse sign')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'sim_to_chem_synapse_sign.pdf')

    plt.show()


def predict_gap_synapse_sign(weights, masks, metric=met.accuracy, rng=np.random.default_rng(), fig_save_path=None):
    # get the connections associated with chem but not gap junctions
    # and the connections associated with gap but not chemical junctions
    # TODO: when I was just masking based on number of stimulations, but not based on the nans in the data
    # the fraction of positive gap junctions in the data went down... seems like a bug
    gap_no_chem = masks['gap'] & ~masks['chem']

    model_synap_dirms_gap = weights['models']['synap']['eirms'][gap_no_chem]
    data_irms_gap = weights['data']['test']['irms'][gap_no_chem]

    # binarize the synapses into greater than / less than 0
    # note that in python 3 > nan is False annoyingly. set it back to nan
    nan_loc_gap = np.isnan(model_synap_dirms_gap)
    model_synap_dirms_gap = (model_synap_dirms_gap > 0).astype(float)
    data_irms_gap = (data_irms_gap > 0).astype(float)

    model_synap_dirms_gap[nan_loc_gap] = np.nan
    data_irms_gap[nan_loc_gap] = np.nan

    # calculate the rate of positive synapses for chemical vs gap junction
    model_synap_dirms_gap_pr, model_synap_dirms_gap_pr_ci = met.metric_ci(metric, np.ones_like(model_synap_dirms_gap), model_synap_dirms_gap, rng=rng)
    data_irms_gap_pr, data_irms_gap_pr_ci = met.metric_ci(metric, np.ones_like(data_irms_gap), data_irms_gap, rng=rng)

    plt.figure()
    y_val = np.array([model_synap_dirms_gap_pr, data_irms_gap_pr])
    y_val_ci = np.stack([model_synap_dirms_gap_pr_ci, data_irms_gap_pr_ci]).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'data'], rotation=45)
    plt.ylabel('% positive synapse')
    plt.title('predicted sign of gap junctions')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'sim_to_gap_synapse_sign.pdf')

    plt.show()


def unconstrained_model_vs_connectome(weights, masks, fig_save_path=None):
    model_synap_dirms_conn = met.f_measure(masks['synap'], weights['models']['synap']['dirms_binarized'])
    model_uncon_dirms_conn = met.f_measure(masks['synap'], weights['models']['unconstrained']['dirms_binarized'])
    model_synap_dirms = weights['models']['synap']['dirms']
    model_uncon_dirms = weights['models']['unconstrained']['dirms']

    plt.figure()
    y_val = np.array([model_synap_dirms_conn, model_uncon_dirms_conn])
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained'], rotation=45)
    plt.ylabel('similarity to anatomical connections')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'sim_to_conn.pdf')

    # plot model similarity to synapse count
    plt.figure()
    anatomy_mat = weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn']
    weights_to_sc = []
    weights_to_sc_ci = []

    wts, wts_ci = met.nan_corr(anatomy_mat[masks['synap']], model_synap_dirms[masks['synap']])
    weights_to_sc.append(wts)
    weights_to_sc_ci.append(wts_ci)

    wts, wts_ci = met.nan_corr(anatomy_mat[masks['synap']], model_uncon_dirms[masks['synap']])
    weights_to_sc.append(wts)
    weights_to_sc_ci.append(wts_ci)

    y_val = np.array(weights_to_sc)
    y_val_ci = np.stack(weights_to_sc_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained'], rotation=45)
    plt.ylabel('similarity to synapse count')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'sim_to_synap_count.pdf')

    plt.show()


def corr_zimmer_paper(weights, models, cell_ids):
    model = models['synap']

    # including only the neurons we have in the model
    cell_ids_selected = ['AVAL', 'AVAR', 'RIML', 'RIMR', 'AIBL', 'AIBR', 'AVEL', 'AVER',
                         'SABD', 'SABVL', 'URYDL', 'URYDR', 'URYVR', 'URYVL', 'SABVR',
                         'RIVL', 'RIVR', 'SMDVL', 'SMDVR', 'SMDDL', 'SMDDR', 'ALA', 'ASKL', 'ASKR', 'PHAL', 'PHAR',
                         'DVC', 'AVFL', 'AVFR', 'AVBL', 'AVBR', 'RID', 'RIBL', 'RIBR', 'PVNL',
                         'DVA', 'SIADL', 'SIAVR', 'SIADR', 'RMEV', 'RMED', 'RMEL', 'RIS', 'PLML',
                         'PVNR', 'RMER']

    neurons_to_silence = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'PVCL', 'PVCR', 'RIML', 'RIMR']
    # neurons_to_silence = ['PVQR', 'RIPR', 'RIPL', 'M2R', 'M2L']
    # neurons_to_silence = ['AVBL', 'AVBR', 'RIBL', 'RIBR', 'AIBL', 'AIBR']
    model = models['synap']
    model_silenced = ssmu.get_silenced_model(model, neurons_to_silence)

    # get the indicies of the selected neurons to show
    cell_plot_inds = np.zeros(len(cell_ids_selected), dtype=int)
    for ci, c in enumerate(cell_ids_selected):
        cell_plot_inds[ci] = cell_ids['all'].index(c)

    # predict the covarian
    model_corr = ssmu.predict_model_corr_coef(model)
    model_silenced_corr = ssmu.predict_model_corr_coef(model_silenced)

    # select the neurons you want to predict from the larger matrix
    data_corr_plot = weights['data']['train']['corr'][np.ix_(cell_plot_inds, cell_plot_inds)]
    model_corr_plot = model_corr[np.ix_(cell_plot_inds, cell_plot_inds)]
    model_corr_silenced_plot = model_silenced_corr[np.ix_(cell_plot_inds, cell_plot_inds)]

    # set diagonals to nan for visualization
    data_corr_plot[np.eye(data_corr_plot.shape[0], dtype=bool)] = np.nan
    model_corr_plot[np.eye(model_corr_plot.shape[0], dtype=bool)] = np.nan
    model_corr_silenced_plot[np.eye(model_corr_silenced_plot.shape[0], dtype=bool)] = np.nan

    cell_ids_plot = cell_ids_selected.copy()
    for i in range(len(cell_ids_plot)):
        if cell_ids_plot[i] in neurons_to_silence:
            cell_ids_plot[i] = '*' + cell_ids_plot[i]

    plot_x = np.arange(len(cell_ids_plot))

    cmax = np.nanmax(np.abs((data_corr_plot, model_corr_plot)))
    plot_clim = (-cmax, cmax)

    plt.figure()
    plt.imshow(data_corr_plot, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, cell_ids_plot, size=8, rotation=90)
    plt.yticks(plot_x, cell_ids_plot, size=8)
    plt.clim(plot_clim)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(model_corr_plot, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, cell_ids_plot, size=5, rotation=90)
    plt.yticks(plot_x, cell_ids_plot, size=5)
    plt.clim(plot_clim)

    plt.subplot(1, 2, 2)
    plt.imshow(model_corr_silenced_plot, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, cell_ids_plot, size=5, rotation=90)
    plt.yticks(plot_x, cell_ids_plot, size=5)
    plt.clim(plot_clim)

    plt.figure()
    plt.imshow(np.abs(model_corr_plot - model_corr_silenced_plot), interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, cell_ids_plot, size=8, rotation=90)
    plt.yticks(plot_x, cell_ids_plot, size=8)
    plt.clim(plot_clim)
    plt.colorbar()

    num_bins = 100
    plt.figure()
    plt.hist(model_corr.reshape(-1), bins=num_bins, density=True, label='model', fc=(1, 0, 0, 0.5))
    plt.hist(model_silenced_corr.reshape(-1), bins=num_bins, density=True, label='silenced model', fc=(0, 0, 1, 0.5))
    plt.legend()
    plt.title('matrix of correlation coefficients')
    plt.xlabel('correlation coefficient')
    plt.ylabel('probability density')

    plt.show()

