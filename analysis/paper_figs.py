from pathlib import Path
from matplotlib import pyplot as plt
import analysis_utilities as au
import analysis_methods as am
import numpy as np
import pickle


def figure_1(weights, masks, cell_ids, fig_save_path=None, window=(30, 60), num_traces_plot=5):
    # set all the weights to nan with the nan mask
    weights['data']['irms'][masks['nan']] = np.nan
    weights['data']['corr'][masks['nan']] = np.nan

    weights['anatomy']['chem_conn'][masks['nan']] = np.nan
    weights['anatomy']['gap_conn'][masks['nan']] = np.nan
    weights['anatomy']['pep_conn'][masks['nan']] = np.nan

    for i in weights['models']:
        weights['models'][i]['weights'][:, masks['nan']] = np.nan
        weights['models'][i]['irms'][masks['nan']] = np.nan
        weights['models'][i]['dirms'][masks['nan']] = np.nan
        weights['models'][i]['corr'][masks['nan']] = np.nan

    # compare data corr and data IRM to connectome
    data_corr = weights['data']['corr']
    data_irms = weights['data']['irms']
    weights_mag = weights['models']['synap']['dirms']
    anatomy_bin = masks['synap']
    std_factor = 3
    data_corr_bin = (np.abs(data_corr) > (np.nanstd(data_corr) / std_factor)).astype(float)
    data_irms_bin = (np.abs(data_irms) > (np.nanstd(data_irms) / std_factor)).astype(float)
    model_weights_bin = (np.abs(weights_mag) > (np.nanstd(weights_mag) / std_factor)).astype(float)

    data_corr_bin[masks['nan']] = np.nan
    data_irms_bin[masks['nan']] = np.nan
    model_weights_bin[masks['nan']] = np.nan

    data_corr_conn, data_corr_conn_ci = au.nan_corr(anatomy_bin, data_corr)
    data_irm_conn, data_irm_conn_ci = au.nan_corr(anatomy_bin, data_irms)
    model_weights_conn, model_weights_conn_ci = au.nan_corr(anatomy_bin, model_weights_bin)

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_corr_score = []
    model_corr_score_ci = []
    model_irms_score = []
    model_irms_score_ci = []
    for m in weights['models']:
        model_corr_to_measured_corr, model_corr_to_measured_corr_ci = au.nan_corr(weights['models'][m]['corr'], data_corr)
        model_corr_score.append(model_corr_to_measured_corr)
        model_corr_score_ci.append(model_corr_to_measured_corr_ci)

        model_irms_to_measured_irms, model_irms_to_measured_irms_ci = au.nan_corr(weights['models'][m]['irms'], data_irms)
        model_irms_score.append(model_irms_to_measured_irms)
        model_irms_score_ci.append(model_irms_to_measured_irms_ci)

    # plotting
    am.plot_irf(measured_irf=weights['data']['irfs'], measured_irf_sem=weights['data']['irfs_sem'], model_irf=weights['models']['synap']['irfs'],
                cell_ids=cell_ids['all'], cell_ids_chosen=cell_ids['chosen'], window=window, num_plot=num_traces_plot,
                fig_save_path=fig_save_path)
    am.plot_irm(model_weights=weights['models']['synap']['dirms'], measured_irm=data_irms, model_irm=weights['models']['synap']['irms'],
                data_corr=data_corr, cell_ids=cell_ids['all'], cell_ids_chosen=cell_ids['chosen'], fig_save_path=fig_save_path)

    plt.figure()
    y_val = np.array([model_weights_conn, data_corr_conn, data_irm_conn, ])
    y_val_ci = np.stack([model_weights_conn_ci, data_corr_conn_ci, data_irm_conn_ci]).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model weights', 'data correlation', 'data IRMs'])
    plt.ylabel('similarity to connectome')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_conn.pdf')

    plt.figure()
    y_val = np.array(model_corr_score)
    y_val_ci = np.stack(model_corr_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    plt.ylabel('similarity to measured correlation')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_corr.pdf')

    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ scrambled labels', 'model\n+ scrambled anatomy'])
    plt.ylabel('similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_irm.pdf')

    plt.show()


def figure_2(weights, masks, cell_ids):
    # compare model IRMs to measured IRMs
    model_irms_score = []
    model_irms_score_ci = []
    for mi in model_irms:
        model_irms_to_measured_irms, model_irms_to_measured_irms_ci = au.nan_corr(mi, data_irms)
        model_irms_score.append(model_irms_to_measured_irms)
        model_irms_score_ci.append(model_irms_to_measured_irms_ci)

    # compare model weights to synpase count
    # TODO change back to chem_conn
    chem_mask = chem_conn > 0
    anat_mask = (chem_conn + gap_conn) > 0
    anatomy_list = [chem_conn[anat_mask], gap_conn[anat_mask]]
    # sub sample the weights down to where there are known connections
    model_weights_anat = []
    model_weights_anat_abs = []
    model_weights_chem = []
    for mw in model_weights:
        split_weights = np.split(mw, mw.shape[0])
        model_weights_anat_abs.append([np.abs(i[0, anat_mask]) for i in split_weights])
        model_weights_anat.append([i[0, anat_mask] for i in split_weights])
        model_weights_chem.append([i[0, chem_mask] for i in split_weights])

    num_models = len(model_weights_anat_abs)
    weights_to_sc = []
    weights_to_sc_ci = []
    anatomy_comb = []
    weights_comb = []
    model_to_model = []
    for i in range(num_models):
        this_weights_to_sc, this_weights_to_sc_ci, this_anatomy_comb, this_weights_comb = \
            au.compare_matrix_sets(anatomy_list, model_weights_anat_abs[i], positive_weights=True)
        weights_to_sc.append(this_weights_to_sc)
        weights_to_sc_ci.append(this_weights_to_sc_ci)
        anatomy_comb.append(this_anatomy_comb)
        weights_comb.append(this_weights_comb)

    model_to_model_sim, model_to_model_sim_ci, constrained_weights, unconstrained_weights = \
        au.compare_matrix_sets(model_weights_anat[0], model_weights_anat[1], positive_weights=False)

    # compare to predicted synapse sign
    watlas = wa.NeuroAtlas()
    signed_connections = watlas.get_anatomical_connectome(signed=True)
    chem_sign = watlas.get_chemical_synapse_sign()

    cmplx = np.logical_and(np.any(chem_sign == -1, axis=0),
                           np.any(chem_sign == 1, axis=0))
    chem_sign = np.nansum(chem_sign, axis=0)
    chem_sign[cmplx] = 0
    chem_sign[chem_sign == 0] = np.nan
    chem_sign[chem_sign > 1] = 1
    chem_sign[chem_sign < -1] = -1

    atlas_ids = list(watlas.neuron_ids)
    atlas_ids[atlas_ids.index('AWCON')] = 'AWCL'
    atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCR'
    cell_inds = np.array([atlas_ids.index(i) for i in cell_ids])
    chem_sign = chem_sign[np.ix_(cell_inds, cell_inds)]
    chem_sign[nan_mask] = np.nan
    # TODO: decide if there should be a negative sign here
    chem_sign = -chem_sign[chem_mask]

    mss = []
    mss_ci = []
    for mwc in range(len(model_weights_chem)):
        for m in range(len(model_weights_chem[mwc])):
            model_weights_chem[mwc][m] = (model_weights_chem[mwc][m] > 0).astype(float) - (
                        model_weights_chem[mwc][m] < 0).astype(float)

        model_sign_sim, model_sign_sim_ci = \
            au.compare_matrix_sets(chem_sign, model_weights_chem[mwc], positive_weights=False)[:2]

        mss.append(model_sign_sim)
        mss_ci.append(model_sign_sim_ci)

    irm_sign, irm_sign_ci = au.nan_corr(chem_sign, data_irms[chem_mask])
    mss.append(irm_sign)
    mss_ci.append(irm_sign_ci)

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
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
    plt.ylabel('similarity to measured IRMs')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_irms.pdf')

    plt.figure()
    y_val = np.array(mss)
    y_val_ci = np.stack(mss_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained', 'data IRMs'])
    plt.ylabel('similarity to known synapse sign')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_synapse_sign.pdf')

    # plot model vs unconstrained model IRMs
    plt.figure()
    model_irm_similarity = au.nan_corr(model_irms[0].reshape(-1), model_irms[1].reshape(-1))[0]
    plt.scatter(model_irms[0].reshape(-1), model_irms[1].reshape(-1))
    plt.xlabel('model IRMs')
    plt.ylabel('unconstrained model IRMs')
    plt.title('similarity = ' + str(model_irm_similarity))
    plt.savefig(fig_save_path / 'model_to_model_irm_scatter.pdf')

    # plot model vs unconstrained model weights
    plt.figure()
    plt.scatter(constrained_weights, unconstrained_weights)
    plt.xlabel('model weights')
    plt.ylabel('unconstrained model weights')
    plt.title('model weights where there are anatomical connections\nsimilarity = ' + str(model_to_model_sim))
    plt.savefig(fig_save_path / 'model_to_model_weights_scatter.pdf')

    plt.figure()
    y_val = np.array(weights_to_anat)
    y_val_ci = np.stack(weights_to_anat_ci).T
    plot_x = np.arange(y_val.shape[0])
    plt.bar(plot_x, y_val)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\nunconstrained'])
    plt.ylabel('similarity to anatomical connections')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.savefig(fig_save_path / 'sim_to_conn.pdf')

    plt.figure()
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

    for i in range(num_models):
        plt.figure()
        plt.scatter(anatomy_comb[i], weights_comb[i])
        plt.xlabel('synapse count')
        plt.ylabel('|model weights|')

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


