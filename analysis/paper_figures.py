from matplotlib import pyplot as plt
import numpy as np
import wormneuroatlas as wa
import metrics as met
import lgssm_utilities as ssmu
import matplotlib as mpl
import analysis_utilities as au
import scipy.cluster.hierarchy as sch


# colormap = mpl.colormaps['RdBu_r']
colormap = mpl.colormaps['coolwarm']
# colormap.set_bad(color=[0.8, 0.8, 0.8])

plot_color = {'data': np.array([27, 158, 119]) / 255,
              'synap': np.array([217, 95, 2]) / 255,
              'unconstrained': np.array([117, 112, 179]) / 255,
              'synap_randA': np.array([231, 41, 138]) / 255,
              # 'synap_randC': np.array([102, 166, 30]) / 255,
              'synap_randC': np.array([128, 128, 128]) / 255,
              'anatomy': np.array([64, 64, 64]) / 255,
              }


def weight_prediction(weights, masks, weight_name, fig_save_path=None):
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
    for m in [weights['models']['synap'][weight_name], weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn'], weights['models']['synap_randC'][weight_name]]:
    # for m in [weights['models']['synap'][weight_name], masks['synap'], weights['models']['synap_randC'][weight_name]]:
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
            met.nan_corr(m, weights['data']['test'][weight_name])
        model_irms_score.append(model_irms_to_measured_irms_test)
        model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['synap'], plot_color['anatomy'], plot_color['synap_randC']]
    plt.bar(plot_x, y_val / irms_baseline, color=bar_colors)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'connectome', 'model\n+ scrambled labels'], rotation=45)
    # plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'], rotation=45)
    plt.ylabel('% explainable correlation to measured ' + weight_name)
    plt.tight_layout()

    plt.savefig(fig_save_path / ('measured_vs_model_randC_' + weight_name + '.pdf'))

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
    bar_colors = [plot_color['synap'], plot_color['unconstrained'], plot_color['synap_randA']]
    plt.bar(plot_x, y_val / irms_baseline, color=bar_colors)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    # plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'], rotation=45)
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained', 'model\n+ scrambled anatomy'], rotation=45)
    plt.ylabel('% explainable correlation to measured ' + weight_name)
    plt.tight_layout()

    plt.savefig(fig_save_path / ('measured_vs_model_randA_' + weight_name + '.pdf'))

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
    plt.errorbar(n_stim_sweep, irms_baseline_sweep, irms_baseline_sweep_ci, label='explainable correlation', color=plot_color['data'])
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs, mcs_ci, label=n, color=plot_color[n])
    plt.xlabel('# of stimulation events')
    plt.ylabel('correlation to measured ' + weight_name)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(0, 0)
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs / irms_baseline_sweep, mcs_ci / irms_baseline_sweep, label=n, color=plot_color[n])
    plt.xlabel('# of stimulation events')
    plt.ylabel('% explainable correlation to measured ' + weight_name)
    plt.tight_layout()
    plt.savefig(fig_save_path / ('measured_vs_model_' + weight_name + '_over_n.pdf'))

    plt.show()


def weights_vs_connectome(weights, masks, metric=met.f_measure, rng=np.random.default_rng(), fig_save_path=None):
    # weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans'], masks['corr_nans'], combine_masks=True)

    model_weights_conn, model_weights_conn_ci = met.metric_ci(metric, masks['synap'], weights['models']['synap']['eirms_binarized'], rng=rng)
    data_corr_conn, data_corr_conn_ci = met.nan_corr(masks['synap'], weights['data']['train']['corr_binarized'])
    data_irm_conn, data_irm_conn_ci = met.nan_corr(masks['synap'], weights['data']['train']['q'])
    conn_null = met.metric_null(metric, masks['synap'])

    model_weights = np.abs(weights['models']['synap']['eirms'][masks['synap']])
    synapse_counts = (weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn'])[masks['synap']]
    weights_counts_corr, weights_counts_corr_ci = met.nan_corr(model_weights, synapse_counts)

    plt.figure()
    plt.scatter(model_weights.reshape(-1), synapse_counts.reshape(-1))
    plt.title(weights_counts_corr)
    plt.show()

    # plot model weight similarity to connectome
    plt.figure()
    y_val = np.array([data_corr_conn, data_irm_conn])
    y_val_ci = np.stack([data_corr_conn_ci, data_irm_conn_ci]).T
    # y_val = np.array([data_corr_conn, data_irm_conn, model_weights_conn])
    # y_val_ci = np.stack([data_corr_conn_ci, data_irm_conn_ci, model_weights_conn_ci]).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.ylim([0, 1])
    # plt.axhline(conn_null, color='k', linestyle='--')
    plt.xticks(plot_x, labels=['data correlation', 'data STAMs'])
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

    # sort_inds = np.argsort(np.abs(eig_vals))
    sort_inds = np.argsort(np.abs(eig_vals))[::-1]

    eig_vals = eig_vals[sort_inds]
    eig_vects = eig_vects[:, sort_inds]
    remove_complex_conj = np.where(np.iscomplex(eig_vals))[0][::2]
    eig_vals = np.delete(eig_vals, remove_complex_conj)
    eig_vects = np.delete(eig_vects, remove_complex_conj, axis=1)
    eig_vects_stacked = np.stack(np.split(eig_vects, model.dynamics_lags, axis=0))
    eig_vects_comb_real = np.mean(np.real(eig_vects_stacked), axis=0)
    eig_vects_comb_abs = np.mean(np.abs(eig_vects_stacked), axis=0)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(eig_vals))
    plt.xlabel('eigenvalue')

    plt.subplot(1, 2, 2)
    plt.plot(np.abs(eig_vals[:100]))
    plt.xlabel('eigenvalue')

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(np.real(eig_vals), np.imag(eig_vals))
    plt.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)))
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_aspect('equal', 'box')
    plt.xlabel('real[eigenvalues]')
    plt.ylabel('imag[eigenvalues]')

    ylim_max = np.max(np.abs(eig_vects_comb_real))
    ylim_plot = (-ylim_max, ylim_max)

    num_eigvects_to_plot = 10
    num_eigvect_inds_to_plot = 20
    plot_x = np.arange(num_eigvect_inds_to_plot)

    for n in range(num_eigvects_to_plot):
        cell_ids_plot = cell_ids['all'].copy()
        this_eig_vect_comb_real = eig_vects_comb_real[:, n]
        this_eig_vect_comb_abs = eig_vects_comb_abs[:, n]
        this_eig_vect_stacked = eig_vects_stacked[:, :, n]

        cell_sort_inds = np.argsort(this_eig_vect_comb_abs)[::-1]

        this_eig_vect_comb_abs = this_eig_vect_comb_abs[cell_sort_inds]
        this_eig_vect_comb_real = this_eig_vect_comb_real[cell_sort_inds]
        this_eig_vect_stacked = this_eig_vect_stacked[:, cell_sort_inds]

        cell_ids_plot = [cell_ids_plot[i] for i in cell_sort_inds[:num_eigvect_inds_to_plot]]

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.scatter(plot_x, this_eig_vect_comb_abs[:num_eigvect_inds_to_plot])
        plt.ylim(ylim_plot)
        # plt.xticks(plot_x, cell_ids_plot, rotation=90)

        plt.subplot(3, 1, 2)
        plt.scatter(plot_x, this_eig_vect_comb_real[:num_eigvect_inds_to_plot])
        plt.ylim(ylim_plot)
        # plt.xticks(plot_x, cell_ids_plot, rotation=90)

        plt.subplot(3, 1, 3)
        plt.imshow(np.real(this_eig_vect_stacked[:, :num_eigvect_inds_to_plot]))
        plt.xticks(plot_x, cell_ids_plot, rotation=90)
        plt.ylabel('time lags')

    plt.show()


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


def plot_irms(weights, cell_ids, num_neurons=None, fig_save_path=None):
    font_size = 8

    data_irms = weights['data']['test']['irms'].copy()
    data_corr = weights['data']['test']['corr'].copy()

    not_all_nan = ~np.all(np.isnan(data_corr), axis=0)
    data_irms = data_irms[:, not_all_nan][not_all_nan, :]
    data_corr = data_corr[:, not_all_nan][not_all_nan, :]
    cell_ids_all = cell_ids['all'].copy()
    cell_ids_all = [cell_ids_all[i] for i in range(len(cell_ids_all)) if not_all_nan[i]]

    # get the neurons to plot
    if num_neurons is None:
        cell_ids_plot = cell_ids_all
        neuron_inds = [i for i in range(len(cell_ids_all))]
    else:
        cell_ids_plot = sorted(cell_ids['sorted'][:num_neurons])
        neuron_inds = [cell_ids_all.index(i) for i in cell_ids_plot]

    data_irms[np.eye(data_irms.shape[0], dtype=bool)] = np.nan
    data_corr[np.eye(data_corr.shape[0], dtype=bool)] = np.nan
    data_irms = data_irms[np.ix_(neuron_inds, neuron_inds)]
    data_corr = data_corr[np.ix_(neuron_inds, neuron_inds)]

    # get euclidian distances for correlation matrix
    # corr_dist = au.condensed_distance(data_corr)
    # link = sch.linkage(corr_dist)
    # dendo = sch.dendrogram(link)
    # new_order = dendo['leaves']
    # new_order = np.arange(data_corr.shape[0])

    # data_irms = data_irms[np.ix_(new_order, new_order)]
    # data_corr = data_corr[np.ix_(new_order, new_order)]
    # cell_ids_plot = [cell_ids_plot[i] for i in new_order]

    plot_x = np.arange(len(cell_ids_plot))

    plt.figure()
    plt.imshow(data_irms, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    color_limits = np.nanmax(np.abs(data_irms))
    color_limits = (-color_limits, color_limits)
    plt.clim(color_limits)
    plt.title('data IRMs')
    if num_neurons is None:
        plt.savefig(fig_save_path / 'full_data_irms.pdf')
    else:
        plt.savefig(fig_save_path / 'sampled_data_irms.pdf')

    plt.figure()
    plt.imshow(data_corr, interpolation='nearest', cmap=colormap)
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    color_limits = np.nanmax(np.abs(data_corr))
    color_limits = (-color_limits, color_limits)
    plt.clim(color_limits)
    plt.title('data correlation matrix')
    if num_neurons is None:
        plt.savefig(fig_save_path / 'full_data_corr.pdf')
    else:
        plt.savefig(fig_save_path / 'sampled_data_corr.pdf')

    for m in ['synap', 'unconstrained', 'synap_randA']:
        neurons_to_mask = ['AIAL']
        model_irms = weights['models'][m]['irms'].copy()
        model_corr = weights['models'][m]['corr'].copy()

        model_irms = model_irms[:, not_all_nan][not_all_nan, :]
        model_corr = model_corr[:, not_all_nan][not_all_nan, :]
        # model_irms = model_irms[np.ix_(new_order, new_order)]
        # model_corr = model_corr[np.ix_(new_order, new_order)]

        model_irms[np.eye(model_irms.shape[0], dtype=bool)] = np.nan
        model_corr[np.eye(model_corr.shape[0], dtype=bool)] = np.nan

        for n in neurons_to_mask:
            neuron_ind = cell_ids['all'].index(n)
            model_irms[neuron_ind, :] = 0
            model_irms[neuron_ind, neuron_ind] = np.nan

        model_irms = model_irms[np.ix_(neuron_inds, neuron_inds)]
        model_corr = model_corr[np.ix_(neuron_inds, neuron_inds)]

        plt.figure()
        plt.imshow(model_irms, interpolation='nearest', cmap=colormap)
        plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
        plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
        color_limits = np.nanmax(np.abs(model_irms))
        color_limits = (-color_limits, color_limits)
        plt.clim(color_limits)
        plt.title(m + ' IRMs')
        if num_neurons is None:
            plt.savefig(fig_save_path / ('full_data_irms_' + m + '.pdf'))
        else:
            plt.savefig(fig_save_path / ('sampled_data_irms_' + m + '.pdf'))

        plt.figure()
        plt.imshow(model_corr, interpolation='nearest', cmap=colormap)
        plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
        plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
        color_limits = np.nanmax(np.abs(model_corr))
        color_limits = (-color_limits, color_limits)
        plt.clim(color_limits)
        plt.title(m + ' correlation matrix')
        if num_neurons is None:
            plt.savefig(fig_save_path / ('full_data_corr_' + m + '.pdf'))
        else:
            plt.savefig(fig_save_path / ('sampled_data_corr_' + m + '.pdf'))

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
    irm_corr = np.zeros(data_irfs.shape[1])
    for i in range(data_irfs.shape[1]):
        irm_corr[i] = met.nan_corr(model_irfs[:, i], data_irfs[:, i])[0]
    irm_dirm_mag_inds = np.argsort(irm_corr)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs.shape[0])

    for ii, i in enumerate(range(num_plot)):
        plt.figure()
        plot_ind = irm_dirm_mag_inds[i]
        this_irf = data_irfs[:, plot_ind]
        this_irf_sem = data_irfs_sem[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.title(cell_ids[plot_ind, 1] + ' -> ' + cell_ids[plot_ind, 0])
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


def plot_dirfs(weights, masks, cell_ids, window, chosen_mask=None, num_plot=10, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs = ssmu.remove_nan_irfs(weights, cell_ids, chosen_mask=chosen_mask)

    data_irfs = no_nan_irfs['data_irfs']
    data_irfs_sem = no_nan_irfs['data_irfs_sem']
    model_irfs = no_nan_irfs['model_irfs']
    model_dirfs = no_nan_irfs['model_dirfs']
    model_eirfs = no_nan_irfs['model_eirfs']
    model_irms = no_nan_irfs['model_irms']
    cell_ids = no_nan_irfs['cell_ids']

    # get the IRFs with the highest correlation to the data
    irm_corr = np.zeros(data_irfs.shape[1])
    for i in range(data_irfs.shape[1]):
        irm_corr[i] = met.nan_r2(model_irfs[:, i], data_irfs[:, i])
    irm_dirm_mag_inds = au.nan_argsort(irm_corr)[::-1]

    plot_x = np.linspace(-window[0], window[1], data_irfs.shape[0])

    offset = 0
    # offset = int(irm_dirm_mag_inds.shape[0] / 2)
    for i in range(offset, offset + num_plot):
        plt.figure()
        plot_ind = irm_dirm_mag_inds[i]
        this_irf = data_irfs[:, plot_ind]
        this_irf_sem = data_irfs_sem[:, plot_ind]

        plt.plot(plot_x, this_irf, label='data irf')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], label='model irf')
        plt.plot(plot_x, model_dirfs[:, plot_ind], label='model dirf')
        # plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.title(cell_ids[plot_ind, 1] + ' -> ' + cell_ids[plot_ind, 0])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity')

        plt.savefig(fig_save_path / ('dirfs_' + str(i) + '.pdf'))

        plt.show()

        a=1

    plt.show()


def plot_specific_dirfs(weights, masks, cell_ids, pairs, window, fig_save_path=None):
    weights = ssmu.mask_weights_to_nan(weights, masks['irm_nans_num_stim'], masks['corr_nans_num_stim'])

    no_nan_irfs = ssmu.remove_nan_irfs(weights, cell_ids)

    data_irfs = no_nan_irfs['data_irfs']
    data_irfs_sem = no_nan_irfs['data_irfs_sem']
    model_irfs = no_nan_irfs['model_irfs']
    model_dirfs = no_nan_irfs['model_dirfs']
    model_eirfs = no_nan_irfs['model_eirfs']
    model_irms = no_nan_irfs['model_irms']
    cell_ids = no_nan_irfs['cell_ids']

    chosen_inds = np.zeros(pairs.shape[0], dtype=int)
    for pi, p in enumerate(pairs):
        chosen_inds[pi] = np.where(np.all(cell_ids == p, axis=1))[0]

    plot_x = np.linspace(-window[0], window[1], data_irfs.shape[0])

    for i, plot_ind in enumerate(chosen_inds):
        plt.figure()
        this_irf = data_irfs[:, plot_ind]
        this_irf_sem = data_irfs_sem[:, plot_ind]

        plt.plot(plot_x, this_irf, color=plot_color['data'], label='data STA')
        plt.fill_between(plot_x, this_irf - this_irf_sem, this_irf + this_irf_sem, color=plot_color['data'], alpha=0.4)

        plt.plot(plot_x, model_irfs[:, plot_ind], color=plot_color['synap'], label='model STA')
        plt.plot(plot_x, model_dirfs[:, plot_ind], color=plot_color['synap'], linestyle='dashed', label='model direct STA')
        # plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        stim_string = cell_ids[plot_ind, 1] + ' -> ' + cell_ids[plot_ind, 0]
        plt.title(stim_string)
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity')

        plt.savefig(fig_save_path / ('dirfs_' + stim_string + '.pdf'))

    plt.show()
    a=1


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
    irm_corr = np.zeros(data_irfs_test.shape[1])
    for i in range(data_irfs_test.shape[1]):
        irm_corr[i] = met.nan_r2(model_irfs[:, i], data_irfs_test[:, i])
    irm_dirm_mag_inds = au.nan_argsort(irm_corr)[::-1]

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
    chem_sign_predict_data_dirms, chem_sign_predict_data_dirms_ci = met.metric_ci(metric, chem_sign, data_irms_chem, rng=rng)

    # get the chance level
    chem_prob = np.nanmean(chem_sign)
    model_prob = np.nanmean(model_synap_dirms_chem)
    data_prob = np.nanmean(data_irms_chem)

    chem_sign_predict_model_synap -= chem_prob * model_prob + (1 - chem_prob) * (1 - model_prob)
    chem_sign_predict_data_dirms -= chem_prob * data_prob + (1 - chem_prob) * (1 - data_prob)

    # calculate a two-sample bootstrap test
    n_boot = 10000
    booted_diff = np.zeros(n_boot)

    # get rid of nans
    chem_sign = chem_sign.reshape(-1).astype(float)
    model_synap_dirms_chem = model_synap_dirms_chem.reshape(-1).astype(float)
    data_irms_chem = data_irms_chem.reshape(-1).astype(float)

    nan_loc = np.isnan(chem_sign) | np.isnan(model_synap_dirms_chem) | np.isnan(data_irms_chem)
    chem_sign = chem_sign[~nan_loc]
    model_synap_dirms_chem = model_synap_dirms_chem[~nan_loc]
    data_irms_chem = data_irms_chem[~nan_loc]

    for n in range(n_boot):
        sample_inds = rng.integers(0, high=chem_sign.shape[0], size=chem_sign.shape[0])
        chem_sign_resampled = chem_sign[sample_inds]
        model_synap_dirms_chem_resampled = model_synap_dirms_chem[sample_inds]
        data_irms_chem_resampled = data_irms_chem[sample_inds]

        # calculate the chance level prob
        chem_sign_prob = np.mean(chem_sign_resampled)
        model_dirms_prob = np.mean(model_synap_dirms_chem_resampled)
        data_dirms_prob = np.mean(data_irms_chem_resampled)

        model_dirms_baseline = chem_sign_prob * data_dirms_prob + (1 - chem_sign_prob) * (1 - model_dirms_prob)
        data_dirms_baseline = chem_sign_prob * data_dirms_prob + (1 - chem_sign_prob) * (1 - data_dirms_prob)

        model_accuracy = metric(model_synap_dirms_chem_resampled, chem_sign_resampled)
        data_accuracy = metric(data_irms_chem_resampled, chem_sign_resampled)

        booted_diff[n] = (model_accuracy - model_dirms_baseline) - (data_accuracy - data_dirms_baseline)

    if np.median(booted_diff) < 0:
        booted_diff *= -1

    p = 2 * np.mean(booted_diff < 0)

    plt.figure()
    plt.hist(booted_diff, bins=50)

    plt.figure()
    y_val = np.array([chem_sign_predict_data_dirms, chem_sign_predict_model_synap])
    y_val_ci = np.stack([chem_sign_predict_data_dirms_ci, chem_sign_predict_model_synap_ci]).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['data IRMs', 'model'], rotation=45)
    plt.ylabel('% correct above random chance')
    plt.title('similarity to known synapse sign\n p=' + str(p))
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
    model_synap_dirms_conn = met.f_measure(masks['synap'], weights['models']['synap']['eirms_binarized'])
    model_uncon_dirms_conn = met.f_measure(masks['synap'], weights['models']['unconstrained']['eirms_binarized'])
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


def plot_missing_neuron(data, posterior_dict, sample_rate=2, fig_save_path=None):
    cell_ids = data['cell_ids']
    posterior_missing = posterior_dict['posterior_missing']
    emissions = data['emissions']
    rng = np.random.default_rng(0)

    # calculate the correlation of the missing to measured neurons
    missing_corr = np.zeros((len(emissions), emissions[0].shape[1]))
    for ei, pi in zip(range(len(emissions)), range(len(posterior_missing))):
        for n in range(emissions[ei].shape[1]):
            if np.mean(~np.isnan(emissions[ei][:, n])) > 0.5:
                missing_corr[ei, n] = met.nan_corr(emissions[ei][:, n], posterior_missing[ei][:, n])[0]
            else:
                missing_corr[ei, n] = np.nan

    # calculate a null distribution
    missing_corr_null = np.zeros((len(emissions), emissions[0].shape[1]))
    for ei, pi in zip(range(len(emissions)), range(len(posterior_missing))):
        random_assignment = rng.permutation(emissions[ei].shape[1])

        for n in range(emissions[ei].shape[1]):
            if np.mean(~np.isnan(emissions[ei][:, n])) > 0.5:
                missing_corr_null[ei, n] = met.nan_corr(emissions[ei][:, n], posterior_missing[ei][:, random_assignment[n]])[0]
            else:
                missing_corr_null[ei, n] = np.nan

    # get the p value that the reconstructed neuron accuracy is significantly different than the null
    p = au.bootstrap_p(missing_corr - missing_corr_null, n_boot=1000)

    plt.figure()
    plt.hist(missing_corr_null.reshape(-1), label='null', alpha=0.5, color='k')
    plt.hist(missing_corr.reshape(-1), label='missing data', alpha=0.5, color=plot_color['synap'])
    plt.title('p = ' + str(p))
    plt.legend()
    plt.xlabel('correlation')
    plt.ylabel('count')

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'recon_histogram.pdf')

    sorted_corr_inds = au.nan_argsort(missing_corr.reshape(-1))
    best_offset = -3  # AVER
    median_offset = 5  # URYDL
    best_neuron = np.unravel_index(sorted_corr_inds[-1 + best_offset], missing_corr.shape)
    median_neuron = np.unravel_index(sorted_corr_inds[int(sorted_corr_inds.shape[0] / 2) + median_offset], missing_corr.shape)

    best_data_ind = best_neuron[0]
    best_neuron_ind = best_neuron[1]
    median_data_ind = median_neuron[0]
    median_neuron_ind = median_neuron[1]

    plot_x = np.arange(emissions[best_data_ind].shape[0]) / sample_rate
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('best held out neuron: ' + cell_ids[best_neuron_ind])
    plt.plot(plot_x, emissions[best_data_ind][:, best_neuron_ind], label='data', color=plot_color['data'])
    plt.plot(plot_x, posterior_missing[best_data_ind][:, best_neuron_ind], label='posterior', color=plot_color['synap'])
    plt.xlabel('time (s)')
    plt.ylabel('neural activity')

    plot_x = np.arange(emissions[median_data_ind].shape[0]) / sample_rate
    # plot_x = np.arange(1000) * sample_rate
    plt.subplot(2, 1, 2)
    plt.title('median held out neuron: ' + cell_ids[median_neuron_ind])
    plt.plot(plot_x, emissions[median_data_ind][:, median_neuron_ind], label='data', color=plot_color['data'])
    plt.plot(plot_x, posterior_missing[median_data_ind][:, median_neuron_ind], label='posterior', color=plot_color['synap'])
    plt.ylim(plt.ylim()[0], 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('neural activity')

    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'recon_examples.pdf')

    plt.show()

    return


def plot_sampled_model(data, posterior_dict, cell_ids, sample_rate=2, num_neurons=10,
                       window_size=1000, fig_save_path=None):
    emissions = data['emissions']
    inputs = data['inputs']
    posterior = posterior_dict['posterior']
    model_sampled = posterior_dict['model_sampled_noise']

    cell_ids_chosen = sorted(cell_ids['sorted'][:num_neurons])
    neuron_inds_chosen = np.array([cell_ids['all'].index(i) for i in cell_ids_chosen])

    # get all the inputs but with only the chosen neurons
    inputs_truncated = [i[:, neuron_inds_chosen] for i in inputs]
    emissions_truncated = [e[:, neuron_inds_chosen] for e in emissions]
    data_ind_chosen, time_window = au.get_example_data_set(inputs_truncated, emissions_truncated, window_size=window_size)

    emissions_chosen = emissions[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    inputs_chosen = inputs[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    all_inputs = inputs[data_ind_chosen][time_window[0]:time_window[1], :]
    posterior_chosen = posterior[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    model_sampled_chosen = model_sampled[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]

    stim_events = np.where(np.sum(all_inputs, axis=1) > 0)[0]
    stim_ids = [cell_ids['all'][np.where(all_inputs[i, :])[0][0]] for i in stim_events]

    filt_shape = np.ones(5)
    for i in range(inputs_chosen.shape[1]):
        inputs_chosen[:, i] = np.convolve(inputs_chosen[:, i], filt_shape, mode='same')

    plot_y = np.arange(len(cell_ids_chosen))
    plot_x = np.arange(0, emissions_chosen.shape[0], 60 * sample_rate)

    plt.figure()
    cmax = np.nanpercentile(np.abs((model_sampled_chosen, posterior_chosen)), plot_percent)

    plt.subplot(3, 1, 1)
    plt.imshow(inputs_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x / sample_rate)
    plt.xlabel('time (s)')
    plt.clim((-1, 1))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(emissions_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('data')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x / sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(model_sampled_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('sampled model')
    plt.yticks(plot_y, cell_ids_chosen)
    plt.xticks(plot_x, plot_x / sample_rate)
    plt.xlabel('time (s)')
    plt.colorbar()

    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'emissions_vs_sampled.pdf')

    # plot the sampled model as time traces
    data_offset = -np.arange(emissions_chosen.shape[1])
    emissions_chosen = emissions_chosen + data_offset[None, :]
    model_sampled_chosen = model_sampled_chosen + data_offset[None, :]

    plt.figure()
    for stim_time, stim_name in zip(stim_events, stim_ids):
        plt.axvline(stim_time, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.text(stim_time, 1.5, stim_name, rotation=90)
    plt.plot(emissions_chosen)
    plt.ylim([data_offset[-1] - 1, 1])
    plt.yticks(data_offset, cell_ids_chosen)
    plt.xticks(plot_x, (plot_x / sample_rate).astype(int))
    plt.xlabel('time (s)')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'emissions_time_traces.pdf')

    plt.figure()
    for stim_time, stim_name in zip(stim_events, stim_ids):
        plt.axvline(stim_time, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.text(stim_time, 1.5, stim_name, rotation=90)
    plt.plot(model_sampled_chosen)
    plt.ylim([data_offset[-1] - 1, 1])
    plt.yticks(data_offset, cell_ids_chosen)
    plt.xticks(plot_x, (plot_x / sample_rate).astype(int))
    plt.xlabel('time (s)')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'model_sampled_time_traces.pdf')

    plt.show()

    return

