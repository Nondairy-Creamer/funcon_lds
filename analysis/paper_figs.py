from pathlib import Path
from matplotlib import pyplot as plt
import analysis_utilities as au
import analysis_methods as am
import numpy as np
import pickle


def corr_irm_recon(weights, weights_masked, masks, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data

    # compare data corr and data IRM to connectome
    n_stim_mask = masks['n_stim_mask']
    n_stim_sweep = masks['n_stim_sweep']
    num_neuron_pairs = [np.sum(~i) for i in n_stim_mask]

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_name = []

    model_corr_score = []
    model_corr_score_ci = []
    model_irms_score = []
    model_irms_score_ci = []

    model_corr_score_sweep = []
    model_corr_score_sweep_ci = []
    model_irms_score_sweep = []
    model_irms_score_sweep_ci = []

    # sweep through the minimum number of stimulations allowed and calculate the score
    # as the number of required stimulations goes up the quality of correlation goes up
    corr_baseline_sweep = np.zeros(n_stim_sweep.shape[0])
    corr_baseline_sweep_ci = np.zeros((2, n_stim_sweep.shape[0]))
    irms_baseline_sweep = np.zeros(n_stim_sweep.shape[0])
    irms_baseline_sweep_ci = np.zeros((2, n_stim_sweep.shape[0]))

    # get the baseline correlation and IRMs. This is the best the model could have done
    # across all data
    corr_baseline = au.nan_corr(weights_masked['data']['train']['corr'],
                                weights_masked['data']['test']['corr'])[0]

    irms_baseline = au.nan_corr(weights_masked['data']['train']['irms'],
                                weights_masked['data']['test']['irms'])[0]

    # sweep across number of stims
    for ni, n in enumerate(n_stim_sweep):
        data_train_corr = weights['data']['train']['corr'].copy()
        data_train_irms = weights['data']['train']['irms'].copy()
        data_test_corr = weights['data']['test']['corr'].copy()
        data_test_irms = weights['data']['test']['irms'].copy()

        data_train_corr[n_stim_mask[ni]] = np.nan
        data_train_irms[n_stim_mask[ni]] = np.nan
        data_test_corr[n_stim_mask[ni]] = np.nan
        data_test_irms[n_stim_mask[ni]] = np.nan

        corr_baseline_sweep[ni], corr_baseline_sweep_ci[:, ni] = au.nan_corr(data_train_corr, data_test_corr)
        irms_baseline_sweep[ni], irms_baseline_sweep_ci[:, ni] = au.nan_corr(data_train_irms, data_test_irms)

    # get the comparison between model prediction and data irm/correlation
    for m in weights['models']:
        if m in ['synap', 'synap_randC', 'synap_randA']:
            # first, get the correlation for each model using all the data. Then compare with only part of the data
            model_corr_to_measured_corr, model_corr_to_measured_corr_ci = \
                au.nan_corr(weights_masked['models'][m]['corr'], weights_masked['data']['test']['corr'])
            model_corr_score.append(model_corr_to_measured_corr)
            model_corr_score_ci.append(model_corr_to_measured_corr_ci)

            model_irms_to_measured_irms, model_irms_to_measured_irms_ci = \
                au.nan_corr(weights_masked['models'][m]['irms'], weights_masked['data']['test']['irms'])
            model_irms_score.append(model_irms_to_measured_irms)
            model_irms_score_ci.append(model_irms_to_measured_irms_ci)

            # for each model, calculate its score for both corr and IRM reconstruction across the n stim sweep
            model_name.append(m)
            model_corr_score_sweep.append(np.zeros(n_stim_sweep.shape[0]))
            model_corr_score_sweep_ci.append(np.zeros((2, n_stim_sweep.shape[0])))
            model_irms_score_sweep.append(np.zeros(n_stim_sweep.shape[0]))
            model_irms_score_sweep_ci.append(np.zeros((2, n_stim_sweep.shape[0])))

            for ni, n in enumerate(n_stim_sweep):
                # mask the model predicted correlations and IRMs based on how many stimulation events were observed
                model_corr = weights['models'][m]['corr'].copy()
                model_irms = weights['models'][m]['irms'].copy()
                model_corr[n_stim_mask[ni]] = np.nan
                model_irms[n_stim_mask[ni]] = np.nan

                model_corr_to_measured_corr, model_corr_to_measured_corr_ci = \
                    au.nan_corr(model_corr, weights['data']['test']['corr'])
                model_corr_score_sweep[-1][ni] = model_corr_to_measured_corr
                model_corr_score_sweep_ci[-1][:, ni] = model_corr_to_measured_corr_ci

                model_irms_to_measured_irms, model_irms_to_measured_irms_ci = \
                    au.nan_corr(model_irms, weights['data']['test']['irms'])
                model_irms_score_sweep[-1][ni] = model_irms_to_measured_irms
                model_irms_score_sweep_ci[-1][:, ni] = model_irms_to_measured_irms_ci

    # plot model reconstruction of correlations
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(n_stim_sweep, corr_baseline_sweep, corr_baseline_sweep_ci, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_corr_score_sweep, model_corr_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs, mcs_ci, label=n)
    plt.ylim([0, plt.ylim()[1]])
    plt.xlabel('minimum # of stimulation events')
    plt.ylabel('similarity to measured correlation')
    # plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(n_stim_sweep, corr_baseline_sweep / corr_baseline_sweep, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_corr_score_sweep, model_corr_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs / corr_baseline_sweep, mcs_ci / corr_baseline_sweep, label=n)
    plt.ylim([0, 1])
    plt.xlabel('minimum # of stimulation events')
    plt.ylabel('similarity to measured correlation')
    plt.legend()

    plt.figure()
    y_val = np.array(model_corr_score)
    y_val_ci = np.stack(model_corr_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val / corr_baseline)
    plt.errorbar(plot_x, y_val / corr_baseline, y_val_ci / corr_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    plt.ylabel('normalized similarity to measured correlation')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_corr.pdf')

    # plot model reconstruction of IRMs
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(n_stim_sweep, irms_baseline_sweep, irms_baseline_sweep_ci, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs, mcs_ci, label=n)
    plt.ylim([0, plt.ylim()[1]])
    plt.xlabel('minimum # of stimulation events')
    plt.ylabel('similarity to measured IRMs')
    # plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(n_stim_sweep, irms_baseline_sweep / irms_baseline_sweep, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs / irms_baseline_sweep, mcs_ci / irms_baseline_sweep, label=n)
    plt.ylim([0, 1])
    plt.xlabel('minimum # of stimulation events')
    plt.ylabel('similarity to measured IRMs')
    plt.legend()

    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val / irms_baseline)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    plt.ylabel('normalized similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_irm.pdf')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(n_stim_sweep, num_neuron_pairs)
    plt.ylabel('number of neuron pairs')
    plt.xlabel('minimum # of stimulation events')

    plt.subplot(1, 2, 2)
    plt.plot(n_stim_sweep[n_stim_sweep.shape[0]//2:], num_neuron_pairs[n_stim_sweep.shape[0]//2:])
    plt.ylabel('number of neuron pairs')
    plt.xlabel('minimum # of stimulation events')

    plt.show()


def figure_1(weights, weights_masked, masks, cell_ids, fig_save_path=None):
    # compare data corr and data IRM to connectome
    data_corr = weights_masked['data']['train']['corr']
    data_irms = weights_masked['data']['test']['irms']

    model_weights_conn = au.f_measure(masks['synap'], weights_masked['models']['synap']['dirms_binarized'])
    data_corr_conn = au.f_measure(masks['synap'], weights_masked['data']['train']['corr_binarized'])
    data_irm_conn = au.f_measure(masks['synap'], weights_masked['data']['train']['q'])
    conn_null = au.f_measure_null(masks['synap'])

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_corr_score = []
    model_corr_score_ci = []
    model_irms_score = []
    model_irms_score_ci = []
    num_stim_sweep = np.arange(2, 7)
    model_name = []

    # sweep through the minimum number of stimulations allowed and calculate the score
    # as the number of required stimulations goes up the quality of correlation goes up
    n_stim_mask = masks['n_stim_mask']
    corr_baseline = np.zeros(num_stim_sweep.shape[0])
    irms_baseline = np.zeros(num_stim_sweep.shape[0])
    # get the mask for number of stims
    for ni, n in enumerate(num_stim_sweep):
        data_train_corr = weights_masked['data']['train']['corr'].copy()
        data_train_irms = weights_masked['data']['train']['irms'].copy()
        data_train_corr[n_stim_mask[ni]] = np.nan
        data_train_irms[n_stim_mask[ni]] = np.nan

        corr_baseline[ni] = au.nan_corr(data_train_corr, weights_masked['data']['test']['corr'])[0]
        irms_baseline[ni] = au.nan_corr(data_train_irms, weights_masked['data']['test']['irms'])[0]

    for m in weights['models']:
        if m in ['synap', 'synap_randC', 'synap_randA']:
            model_corr_score.append(np.zeros(num_stim_sweep.shape[0]))
            model_corr_score_ci.append(np.zeros((2, num_stim_sweep.shape[0])))
            model_irms_score.append(np.zeros(num_stim_sweep.shape[0]))
            model_irms_score_ci.append(np.zeros((2, num_stim_sweep.shape[0])))

            model_name.append(m)

            model_corr_to_measured_corr, model_corr_to_measured_corr_ci = au.nan_corr(weights['models'][m]['corr'], data_corr)
            model_corr_score.append(model_corr_to_measured_corr)
            model_corr_score_ci.append(model_corr_to_measured_corr_ci)


            for ni, n in enumerate(num_stim_sweep):
                model_corr = weights_masked['models'][m]['corr'].copy()
                model_irms = weights_masked['models'][m]['irms'].copy()
                model_corr[n_stim_mask[ni]] = np.nan
                model_irms[n_stim_mask[ni]] = np.nan

                model_corr_to_measured_corr, model_corr_to_measured_corr_ci = au.nan_corr(model_corr, data_corr)
                model_corr_score[-1][ni] = model_corr_to_measured_corr
                model_corr_score_ci[-1][:, ni] = model_corr_to_measured_corr_ci

                model_irms_to_measured_irms, model_irms_to_measured_irms_ci = au.nan_corr(model_irms, data_irms)
                model_irms_score[-1][ni] = model_irms_to_measured_irms
                model_irms_score_ci[-1][:, ni] = model_irms_to_measured_irms_ci


    # plotting
    # am.plot_irf(measured_irf=weights['data']['test']['irfs'], measured_irf_sem=weights['data']['test']['irfs_sem'],
    #             model_irf=weights['models']['synap']['irfs'],
    #             cell_ids=cell_ids['all'], cell_ids_chosen=cell_ids['chosen'], window=window, num_plot=num_traces_plot,
    #             fig_save_path=fig_save_path)
    am.plot_irm(model_weights=weights_masked['models']['synap']['dirms'], measured_irm=data_irms, model_irm=weights_masked['models']['synap']['irms'],
                data_corr=data_corr, cell_ids=cell_ids['all'], cell_ids_chosen=cell_ids['chosen'], fig_save_path=fig_save_path)

    # plot model weight similarity to connectome
    plt.figure()
    y_val = np.array([model_weights_conn, data_corr_conn, data_irm_conn])
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.axhline(conn_null, color='k', linestyle='--')
    plt.xticks(plot_x, labels=['model weights', 'data correlation', 'data IRMs'])
    plt.ylabel('similarity to connectome')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_conn.pdf')

    # plot model reconstruction of correlations
    plt.figure()
    y_val = np.array(model_corr_score)
    y_val_ci = np.stack(model_corr_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.subplot(1, 2, 1)
    plt.plot(num_stim_sweep, corr_baseline, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_corr_score, model_corr_score_ci):
        plt.errorbar(num_stim_sweep, mcs, mcs_ci, label=n)
    plt.xlabel('# of stimulation events')
    plt.ylabel('similarity to measured correlation')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    # plt.subplot(1, 2, 2)
    # plt.bar(plot_x, y_val / corr_baseline)
    # plt.errorbar(plot_x, y_val / corr_baseline, y_val_ci / corr_baseline, fmt='none', color='k')
    # plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    # plt.ylabel('normalized similarity to measured correlation')
    # ax = plt.gca()
    # for label in ax.get_xticklabels():
    #     label.set_rotation(45)

    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_corr.pdf')

    # plot model reconstruction of IRMs
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.subplot(1, 2, 1)
    plt.plot(num_stim_sweep, irms_baseline, label='explainable correlation')
    for n, mcs, mcs_ci in zip(model_name, model_irms_score, model_irms_score_ci):
        plt.errorbar(num_stim_sweep, mcs, mcs_ci, label=n)
    plt.xlabel('# of stimulation events')
    plt.ylabel('similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    plt.show()

    plt.subplot(1, 2, 2)
    plt.bar(plot_x, y_val / irms_baseline)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    plt.ylabel('normalized similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_irm.pdf')

    plt.show()


def figure_2(weights, masks, cell_ids, fig_save_path=None, window=(30, 60), num_traces_plot=5, num_rand=100):
    data_corr = weights['data']['train']['corr']
    data_irms = weights['data']['test']['irms']
    rng = np.random.default_rng(0)

    # compare model IRMs to measured IRMs
    model_corr_score = []
    model_corr_score_ci = []
    model_irms_score = []
    model_irms_score_ci = []
    for m in weights['models']:
        if m in ['synap', 'unconstrained']:
            model_corr_to_measured_corr, model_corr_to_measured_corr_ci = au.nan_corr(weights['models'][m]['corr'], data_corr)
            model_corr_score.append(model_corr_to_measured_corr)
            model_corr_score_ci.append(model_corr_to_measured_corr_ci)

            model_irms_to_measured_irms, model_irms_to_measured_irms_ci = au.nan_corr(weights['models'][m]['irms'], data_irms)
            model_irms_score.append(model_irms_to_measured_irms)
            model_irms_score_ci.append(model_irms_to_measured_irms_ci)
    irms_baseline = au.nan_corr(weights['data']['train']['irms'], weights['data']['test']['irms'])[0]

    # compare model dIRMs to synpase count
    model_dirms_synap = weights['models']['synap']['dirms'][masks['synap']]
    synapse_count = (weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn'])[masks['synap']]
    dirms_to_synapse, dirms_to_synapse_ci = au.nan_corr(model_dirms_synap, synapse_count)

    # compare dirms to predicted synapse sign
    # get estimate of synapse sign
    import wormneuroatlas as wa
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
    chem_sign = chem_sign[masks['chem']]

    model_synap_dirms_chem = weights['models']['synap']['dirms'][masks['chem']]
    model_uncon_dirms_chem = weights['models']['unconstrained']['dirms'][masks['chem']]
    data_dirms_chem = weights['data']['test']['irms'][masks['chem']]

    nan_loc = np.isnan(model_synap_dirms_chem)
    model_synap_dirms_chem = (model_synap_dirms_chem > 0).astype(float)
    model_uncon_dirms_chem = (model_uncon_dirms_chem > 0).astype(float)
    data_dirms_chem = (data_dirms_chem > 0).astype(float)

    model_synap_dirms_chem[nan_loc] = np.nan
    model_uncon_dirms_chem[nan_loc] = np.nan
    data_dirms_chem[nan_loc] = np.nan

    # prediction accuracy
    chem_sign_predict_model_synap = au.f_measure(chem_sign, model_synap_dirms_chem)
    chem_sign_predict_model_uncon = au.f_measure(chem_sign, model_uncon_dirms_chem)
    chem_sign_predict_data_dirms = au.f_measure(chem_sign, data_dirms_chem)
    chem_sign_predict_null = au.f_measure_null(chem_sign)

    # compare model weights with binary connections
    # binarize model weights
    model_synap_weights_conn = au.f_measure(masks['synap'], weights['models']['synap']['dirms_binarized'])
    model_uncon_weights_conn = au.f_measure(masks['synap'], weights['models']['unconstrained']['dirms_binarized'])

    # plotting
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
    plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels'])
    plt.ylabel('normalized similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_irm_uncon.pdf')

    plt.figure()
    y_val = np.array([chem_sign_predict_model_synap, chem_sign_predict_model_uncon, chem_sign_predict_data_dirms])
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.axhline(chem_sign_predict_null, color='k', linestyle='--')
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained', 'data IRMs'])
    plt.ylabel('similarity to known synapse sign')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_synapse_sign.pdf')

    # plot model vs unconstrained model IRMs
    plt.figure()
    model_synap_irms = weights['models']['synap']['irms']
    model_uncon_irms = weights['models']['unconstrained']['irms']
    model_irm_similarity = au.nan_corr(model_synap_irms.reshape(-1),
                                       model_uncon_irms.reshape(-1))[0]
    plt.scatter(model_synap_irms.reshape(-1), model_uncon_irms.reshape(-1))
    plt.xlabel('model IRMs')
    plt.ylabel('unconstrained model IRMs')
    plt.title('similarity = ' + str(model_irm_similarity))
    plt.savefig(fig_save_path / 'model_to_model_irm_scatter.pdf')

    # plot model vs unconstrained model weights
    plt.figure()
    model_synap_dirms = weights['models']['synap']['irms'][masks['synap']]
    model_uncon_dirms = weights['models']['unconstrained']['irms'][masks['synap']]
    model_irm_similarity = au.nan_corr(model_synap_dirms.reshape(-1),
                                       model_uncon_dirms.reshape(-1))[0]
    plt.scatter(model_synap_dirms, model_uncon_dirms)
    plt.xlabel('model weights')
    plt.ylabel('unconstrained model weights')
    plt.title('model weights where there are anatomical connections\nsimilarity = ' + str(model_irm_similarity))
    plt.savefig(fig_save_path / 'model_to_model_weights_scatter.pdf')

    plt.figure()
    y_val = np.array([model_synap_weights_conn, model_uncon_weights_conn])
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
    plt.ylabel('similarity to anatomical connections')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_conn.pdf')

    # plot model similarity to synapse count
    plt.figure()
    anatomy_mat = weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn']
    weights_to_sc = []
    weights_to_sc_ci = []

    wts, wts_ci = au.nan_corr(anatomy_mat[masks['synap']], model_synap_dirms)
    weights_to_sc.append(wts)
    weights_to_sc_ci.append(wts_ci)

    wts, wts_ci = au.nan_corr(anatomy_mat[masks['synap']], model_uncon_dirms)
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
    plt.savefig(fig_save_path / 'sim_to_synap_count.pdf')

    plt.show()


def figure_3(weights, masks, cell_ids):
    # compare model IRMs to measured IRMs
    model_irms_score = []
    model_irms_score_ci = []
    for mi in model_irms:
        model_irms_to_measured_irms, model_irms_to_measured_irms_ci = au.nan_corr(mi, data_irms)
        model_irms_score.append(model_irms_to_measured_irms)
        model_irms_score_ci.append(model_irms_to_measured_irms_ci)

    # compare model weights to synpase count
    anat_mask = (chem_conn + gap_conn) > 0
    anatomy_list = [chem_conn[anat_mask], gap_conn[anat_mask]]
    # sub sample the weights down to where there are known connections
    model_weights_anat = []
    for mw in model_weights:
        split_weights = np.split(mw, mw.shape[0])
        model_weights_anat.append([np.abs(i[0, anat_mask]) for i in split_weights])

    num_models = len(model_weights_anat)
    weights_to_sc = []
    weights_to_sc_ci = []
    anatomy_comb = []
    weights_comb = []
    for i in range(num_models):
        this_weights_to_sc, this_weights_to_sc_ci, this_anatomy_comb, this_weights_comb = \
            au.compare_matrix_sets(anatomy_list, model_weights_anat[i], positive_weights=True)
        weights_to_sc.append(this_weights_to_sc)
        weights_to_sc_ci.append(this_weights_to_sc_ci)
        anatomy_comb.append(this_anatomy_comb)
        weights_comb.append(this_weights_comb)

    # compare model weights with binary connections
    # binarize model weights
    std_factor = 3
    weights_bin = [np.sum(np.abs(i), axis=0) for i in model_weights]
    weights_bin = [(i > (np.nanstd(i) / std_factor)).astype(float) for i in weights_bin]
    for i in range(len(weights_bin)):
        weights_bin[i][nan_mask] = np.nan

    weights_to_anat = []
    weights_to_anat_ci = []

    for wb in weights_bin:
        this_weights_to_anat, this_weights_to_anat_ci = au.nan_corr(anat_mask, wb)
        weights_to_anat.append(this_weights_to_anat)
        weights_to_anat_ci.append(this_weights_to_anat_ci)

    # plotting
    am.plot_missing_neuron(data=data_test, posterior_dict=posterior_dicts[0], sample_rate=models[0].sample_rate,
                           fig_save_path=fig_save_path)


