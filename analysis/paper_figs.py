from matplotlib import pyplot as plt
import numpy as np
import wormneuroatlas as wa
import metrics as met
import lgssm_utilities as ssmu


def corr_irm_recon(weights, masks, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_corr_score = []
    model_corr_score_ci = []
    model_irms_score = []
    model_irms_score_ci = []

    # get the baseline correlation and IRMs. This is the best the model could have done
    # across all data
    corr_baseline = met.nan_corr(weights['data']['train']['corr'], weights['data']['test']['corr'])[0]

    irms_baseline = met.nan_corr(weights['data']['train']['irms'], weights['data']['test']['irms'])[0]

    model_corr_to_measured_corr_train, model_corr_to_measured_corr_train_ci = \
        met.nan_corr(weights['models']['synap']['corr'], weights['data']['train']['corr'])
    model_corr_score.append(model_corr_to_measured_corr_train)
    model_corr_score_ci.append(model_corr_to_measured_corr_train_ci)

    model_irms_to_measured_irms_train, model_irms_to_measured_irms_train_ci = \
        met.nan_corr(weights['models']['synap']['irms'], weights['data']['train']['irms'])
    model_irms_score.append(model_irms_to_measured_irms_train)
    model_irms_score_ci.append(model_irms_to_measured_irms_train_ci)

    # get the comparison between model prediction and data irm/correlation
    for m in weights['models']:
        if m in ['synap', 'synap_randC', 'synap_randA']:
            # first, get the correlation for each model using all the data. Then compare with only part of the data
            model_corr_to_measured_corr_test, model_corr_to_measured_corr_test_ci = \
                met.nan_corr(weights['models'][m]['corr'], weights['data']['test']['corr'])
            model_corr_score.append(model_corr_to_measured_corr_test)
            model_corr_score_ci.append(model_corr_to_measured_corr_test_ci)

            model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
                met.nan_corr(weights['models'][m]['irms'], weights['data']['test']['irms'])
            model_irms_score.append(model_irms_to_measured_irms_test)
            model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)



    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(model_corr_score)
    y_val_ci = np.stack(model_corr_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val / corr_baseline)
    plt.errorbar(plot_x, y_val / corr_baseline, y_val_ci / corr_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model (train)', 'model (test)', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    plt.ylabel('explainable correlation to data correlation')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()

    plt.savefig(fig_save_path / 'measured_vs_model_corr.pdf')

    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val / irms_baseline)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model (train)', 'model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    plt.ylabel('explainable correlation to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.savefig(fig_save_path / 'measured_vs_model_irms.pdf')
    plt.tight_layout()

    plt.show()


def corr_irm_recon_sweep(weights, masks, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data

    # compare data corr and data IRM to connectome
    n_stim_mask = masks['n_stim_mask']
    n_stim_sweep = masks['n_stim_sweep']
    n_obs_mask = masks['n_obs_mask']
    n_obs_sweep = masks['n_obs_sweep']

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_name = []

    model_corr_score_sweep = []
    model_corr_score_sweep_ci = []
    model_irms_score_sweep = []
    model_irms_score_sweep_ci = []

    # sweep through the minimum number of stimulations allowed and calculate the score
    # as the number of required stimulations goes up the quality of correlation goes up
    corr_baseline_sweep = np.zeros(n_obs_sweep.shape[0])
    corr_baseline_sweep_ci = np.zeros((2, n_obs_sweep.shape[0]))
    irms_baseline_sweep = np.zeros(n_stim_sweep.shape[0])
    irms_baseline_sweep_ci = np.zeros((2, n_stim_sweep.shape[0]))

    # sweep across number of stims, for the IRF data
    for ni, n in enumerate(n_stim_sweep):
        data_train_irms = weights['data']['train']['irms'].copy()
        data_test_irms = weights['data']['test']['irms'].copy()

        data_train_irms[n_stim_mask[ni]] = np.nan
        data_test_irms[n_stim_mask[ni]] = np.nan

        irms_baseline_sweep[ni], irms_baseline_sweep_ci[:, ni] = met.nan_corr(data_train_irms, data_test_irms)

    # sweep across number of observations, for the correlation data
    for ni, n in enumerate(n_obs_sweep):
        data_train_corr = weights['data']['train']['corr'].copy()
        data_test_corr = weights['data']['test']['corr'].copy()

        data_train_corr[n_obs_mask[ni]] = np.nan
        data_test_corr[n_obs_mask[ni]] = np.nan

        corr_baseline_sweep[ni], corr_baseline_sweep_ci[:, ni] = met.nan_corr(data_train_corr, data_test_corr)

    # get the comparison between model prediction and data irm/correlation
    for m in weights['models']:
        if m in ['synap', 'synap_randC', 'synap_randA']:
            # for each model, calculate its score for both corr and IRM reconstruction across the n stim sweep
            model_name.append(m)
            model_corr_score_sweep.append(np.zeros(n_obs_sweep.shape[0]))
            model_corr_score_sweep_ci.append(np.zeros((2, n_obs_sweep.shape[0])))
            model_irms_score_sweep.append(np.zeros(n_stim_sweep.shape[0]))
            model_irms_score_sweep_ci.append(np.zeros((2, n_stim_sweep.shape[0])))

            for ni, n in enumerate(n_stim_sweep):
                # mask the model predicted correlations and IRMs based on how many stimulation events were observed
                model_irms = weights['models'][m]['irms'].copy()
                model_irms[n_stim_mask[ni]] = np.nan

                model_irms_to_measured_irms, model_irms_to_measured_irms_ci = \
                    met.nan_corr(model_irms, weights['data']['test']['irms'])
                model_irms_score_sweep[-1][ni] = model_irms_to_measured_irms
                model_irms_score_sweep_ci[-1][:, ni] = model_irms_to_measured_irms_ci

            for ni, n in enumerate(n_obs_sweep):
                # mask the model predicted correlations and IRMs based on how many stimulation events were observed
                model_corr = weights['models'][m]['corr'].copy()
                model_corr[n_obs_mask[ni]] = np.nan

                model_corr_to_measured_corr, model_corr_to_measured_corr_ci = \
                    met.nan_corr(model_corr, weights['data']['test']['corr'])
                model_corr_score_sweep[-1][ni] = model_corr_to_measured_corr
                model_corr_score_sweep_ci[-1][:, ni] = model_corr_to_measured_corr_ci

    # plot model reconstruction of correlations
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(n_obs_sweep, corr_baseline_sweep, corr_baseline_sweep_ci, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_corr_score_sweep, model_corr_score_sweep_ci):
        plt.errorbar(n_obs_sweep, mcs, mcs_ci, label=n)
    # plt.ylim([0, plt.ylim()[1]])
    plt.xlabel('# of pairs observed')
    plt.ylabel('similarity to measured correlation')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(0, 0)
    for n, mcs, mcs_ci in zip(model_name, model_corr_score_sweep, model_corr_score_sweep_ci):
        plt.errorbar(n_obs_sweep, mcs / corr_baseline_sweep, mcs_ci / corr_baseline_sweep, label=n)
    # plt.ylim([0, 1])
    plt.xlabel('# of pairs observed')
    plt.ylabel('similarity to measured correlation')
    plt.tight_layout()
    plt.savefig(fig_save_path / 'measured_vs_model_corr_over_n.pdf')

    # plot model reconstruction of IRMs
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(n_stim_sweep, irms_baseline_sweep, irms_baseline_sweep_ci, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs, mcs_ci, label=n)
    # plt.ylim([0, plt.ylim()[1]])
    plt.xlabel('# of stimulation events')
    plt.ylabel('similarity to measured IRMs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(0, 0)
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs / irms_baseline_sweep, mcs_ci / irms_baseline_sweep, label=n)
    # plt.ylim([0, 1])
    plt.xlabel('# of stimulation events')
    plt.ylabel('similarity to measured IRMs')
    plt.tight_layout()
    plt.savefig(fig_save_path / 'measured_vs_model_irms_over_n.pdf')

    plt.show()


def weights_vs_connectome(weights, masks, metric=met.f_measure, rng=np.random.default_rng(), fig_save_path=None):
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
    plt.xticks(plot_x, labels=['model weights', 'data correlation', 'data IRMs'])
    plt.ylabel('similarity to connectome')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'weights_vs_connectome.pdf')

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
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained'])
    plt.ylabel('similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    plt.subplot(1, 2, 2)
    plt.bar(plot_x, y_val / irms_baseline)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained'])
    plt.ylabel('normalized similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'sim_to_irm_uncon.pdf')

    plt.show()


def plot_irfs(weights, cell_ids, window, num_plot=10, fig_save_path=None):
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

    for i in range(num_plot):
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

    plt.show()


def plot_irfs_train_test(weights, cell_ids, window, chosen_mask=None, num_plot=10, fig_save_path=None):
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
        plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity')

    plt.show()


def plot_dirfs_train_test(weights, cell_ids, window, chosen_mask=None, num_plot=10, fig_save_path=None):
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


def plot_dirm_diff(weights, masks, cell_ids, window, num_plot=10, fig_save_path=None):
    no_nan_irfs = ssmu.remove_nan_irfs(weights, cell_ids, chosen_mask=masks['synap'])

    data_irfs = no_nan_irfs['data_irfs']
    data_irfs_sem = no_nan_irfs['data_irfs_sem']
    model_irfs = no_nan_irfs['model_irfs']
    model_dirfs = no_nan_irfs['model_dirfs']
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
        plt.plot(plot_x, model_eirfs[:, plot_ind], label='model eirf')
        plt.title(cell_ids[1, plot_ind] + ' -> ' + cell_ids[0, plot_ind])
        plt.legend()
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('cell activity')

    plt.show()


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
    chem_sign[masks['nan']] = np.nan
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
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained', 'data IRMs'])
    plt.ylabel('% correct')
    plt.title('similarity to known synapse sign')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
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
    plt.xticks(plot_x, labels=['model', 'data'])
    plt.ylabel('% positive synapse')
    plt.title('predicted sign of gap junctions')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
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
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
    plt.ylabel('similarity to anatomical connections')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
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
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
    plt.ylabel('similarity to synapse count')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'sim_to_synap_count.pdf')

    plt.show()
