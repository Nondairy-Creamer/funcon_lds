from matplotlib import pyplot as plt
import numpy as np
# import wormneuroatlas as wa
import metrics as met
import lgssm_utilities as ssmu
import matplotlib as mpl
import analysis_utilities as au
import scipy
from copy import deepcopy
import csv


# colormap = mpl.colormaps['RdBu_r']
colormap = mpl.colormaps['coolwarm']
# colormap.set_bad(color=[0.8, 0.8, 0.8])
plot_percent = 95

plot_color = {'data': np.array([217, 95, 2]) / 255,
              'synap': np.array([27, 158, 119]) / 255,
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

    train_weights = weights['data']['train'][weight_name].copy()
    test_weights = weights['data']['test'][weight_name].copy()
    num_neurons = train_weights.shape[0]

    test_weights[np.eye(num_neurons, dtype=bool)] = np.nan
    train_weights[np.eye(num_neurons, dtype=bool)] = np.nan

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    model_irms_score = []
    model_irms_score_ci = []

    # get the baseline correlation and IRMs. This is the best the model could have done
    # across all data
    irms_baseline = met.nan_corr(test_weights, train_weights)[0]

    weights_to_compare = [weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn'],
                          weights['models']['synap_randC'][weight_name],
                          weights['models']['synap'][weight_name],
                          ]

    for ii, i in enumerate(weights_to_compare):
        weights_to_compare[ii][np.eye(i.shape[0], dtype=bool)] = np.nan

    # get the comparison between model prediction and data irm/correlation
    for mi, m in enumerate(weights_to_compare):
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
            met.nan_corr(m, test_weights)

        model_irms_score.append(model_irms_to_measured_irms_test)
        model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

    # plot average reconstruction over all data without normalization
    y_limits = [0, 1.3]
    plt.figure()
    y_val = np.array([model_irms_score[-1]])
    y_val_ci = np.stack([model_irms_score_ci[-1]]).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['synap']]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.axhline(irms_baseline, color='k', linestyle='--')
    plt.xticks(plot_x, labels=['model'], rotation=45)
    plt.ylabel('correlation to measured ' + weight_name)
    plt.ylim(y_limits)
    plt.tight_layout()
    plt.savefig(fig_save_path / ('measured_vs_model_randC_' + weight_name + '_raw.pdf'))

    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['anatomy'], plot_color['synap_randC'],  plot_color['synap']]
    plt.bar(plot_x, y_val / irms_baseline, color=bar_colors)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['connectome', 'model\n+ scrambled labels', 'model'], rotation=45)
    plt.ylabel('relative correlation to measured ' + weight_name)
    plt.ylim(y_limits)
    plt.tight_layout()

    plt.savefig(fig_save_path / ('measured_vs_model_randC_' + weight_name + '.pdf'))

    plt.show()


def weight_prediction_direct_vs_poly(weights, masks, cell_ids, weight_name, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data

    #TODO this function is not finished. need to normalize each different prediction by its own train-test

    # get the 53 pairs from andy's paper that are speculated to be extrasynaptic
    extrasyn_pairs = [['ADLR', 'VB1'],
                      ['AIMR', 'RMDDR'],
                      ['ASHR', 'AVDR'],
                      ['AVAR', 'SAAVL'],
                      ['AVDL', 'AVDR'],
                      ['AVDR', 'AWBL'],
                      ['AVDR', 'RIVR'],
                      ['AVEL', 'M3L'],
                      ['AVJL', 'AVDR'],
                      ['AVJR', 'CEPDL'],
                      ['AVKL', 'M3L'],
                      ['AWBL', 'IL2DR'],
                      ['AWBR', 'AVDR'],
                      ['AWBR', 'AWCL'],
                      ['AWBR', 'RMEL'],

                      ['AWBR', 'URXR'],
                      ['CEPDL', 'RMDL'],
                      ['CEPVL', 'RMDL'],
                      ['FLPR', 'AVDR'],
                      ['FLPR', 'M3L'],
                      ['I1L', 'ASHL'],
                      ['I1L', 'RMDVR'],
                      ['I1R', 'FLPR'],
                      ['I1R', 'M3L'],
                      ['I2L', 'M3L'],
                      ['I2R', 'I3'],
                      ['I3', 'M3L'],
                      ['IL1VL', 'RIVR'],
                      ['IL2DR', 'M3L'],
                      ['IL2R', 'M3L'],

                      ['M1', 'M3L'],
                      ['M2R', 'M3L'],
                      ['M3R', 'IL1DL'],
                      ['OLLR', 'I3'],
                      ['OLLR', 'IL1DL'],
                      ['OLQDR', 'IL1DL'],
                      ['RIVR', 'AWBL'],
                      ['RMDDL', 'RMDDR'],
                      ['RMDDR', 'AVER'],
                      ['RMDDR', 'RID'],
                      ['RMDL', 'RMDDL'],
                      ['RMDR', 'AWBL'],
                      ['RMDR', 'RMDVR'],
                      ['RMDVL', 'CEPVL'],
                      ['RMEL', 'IL1DL'],

                      ['RMEL', 'RMDVR'],
                      ['RMER', 'IL1VL'],
                      ['RMER', 'M3L'],
                      ['URBL', 'AVKL'],
                      ['URXL', 'AVDR'],
                      ['URXL', 'IL1DL'],
                      ['URYVL', 'M3L'],
                      ['VB1', 'AWBR'],
                      ]

    extrasyn_mask = np.zeros_like(masks['synap']) == 1
    for ep in extrasyn_pairs:
        if (ep[0] not in cell_ids['all']) or (ep[1] not in cell_ids['all']):
            continue
        resp_ind = cell_ids['all'].index(ep[0])
        stim_ind = cell_ids['all'].index(ep[1])

        extrasyn_mask[resp_ind, stim_ind] = True

    train_weights = weights['data']['train'][weight_name].copy()
    test_weights = weights['data']['test'][weight_name].copy()
    num_neurons = train_weights.shape[0]

    test_weights[np.eye(num_neurons, dtype=bool)] = np.nan
    train_weights[np.eye(num_neurons, dtype=bool)] = np.nan

    # compare model corr to measured corr and compare model IRMs to measured IRMs
    synap_irms_score = []
    synap_irms_score_ci = []
    uncon_irms_score = []
    uncon_irms_score_ci = []

    # get the baseline correlation and IRMs. This is the best the model could have done
    # across all data
    irms_baseline_connected = met.nan_corr(test_weights[masks['synap']], train_weights[masks['synap']])[0]
    irms_baseline_unconnected = met.nan_corr(test_weights[~masks['synap']], train_weights[~masks['synap']])[0]
    irms_baseline_extrasyn = met.nan_corr(test_weights[extrasyn_mask], train_weights[extrasyn_mask])[0]
    irms_baseline = np.array([irms_baseline_connected, irms_baseline_unconnected, irms_baseline_extrasyn])

    synap_connected = weights['models']['synap'][weight_name].copy()
    synap_connected[~masks['synap']] = np.nan
    synap_unconnected = weights['models']['synap'][weight_name].copy()
    synap_unconnected[masks['synap']] = np.nan
    synap_extrasyn = weights['models']['synap'][weight_name].copy()
    synap_extrasyn[~extrasyn_mask] = np.nan

    unconstrained_connected = weights['models']['unconstrained'][weight_name].copy()
    unconstrained_connected[~masks['synap']] = np.nan
    unconstrained_unconnected = weights['models']['unconstrained'][weight_name].copy()
    unconstrained_unconnected[masks['synap']] = np.nan
    unconstrained_extrasyn = weights['models']['unconstrained'][weight_name].copy()
    unconstrained_extrasyn[~extrasyn_mask] = np.nan

    weights_to_compare_synap = [synap_connected,
                                synap_unconnected,
                                synap_extrasyn
                                ]

    weights_to_compare_uncon = [unconstrained_connected,
                                unconstrained_unconnected,
                                unconstrained_extrasyn
                                ]

    # p_val, mean_corr, _ = met.two_sample_boostrap_corr_p(test_weights, synap_extrasyn, unconstrained_extrasyn, n_boot=10000)
    p_val = met.two_sample_corr_p(test_weights, synap_extrasyn, unconstrained_extrasyn)

    for ii, i in enumerate(weights_to_compare_synap):
        weights_to_compare_synap[ii][np.eye(i.shape[0], dtype=bool)] = np.nan

    for ii, i in enumerate(weights_to_compare_uncon):
        weights_to_compare_uncon[ii][np.eye(i.shape[0], dtype=bool)] = np.nan

    # get the comparison between model prediction and data irm/correlation
    for mi, m in enumerate(weights_to_compare_synap):
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
            met.nan_corr(m, test_weights)

        synap_irms_score.append(model_irms_to_measured_irms_test)
        synap_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

    # get the comparison between model prediction and data irm/correlation
    for mi, m in enumerate(weights_to_compare_uncon):
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
            met.nan_corr(m, test_weights)

        uncon_irms_score.append(model_irms_to_measured_irms_test)
        uncon_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

    # plot the 56 neurons for connectome-constrained and unconstrained
    nan_loc = np.isnan(test_weights) | np.isnan(synap_extrasyn) | np.isnan(unconstrained_extrasyn)
    target = test_weights[~nan_loc]
    data_1 = synap_extrasyn[~nan_loc]
    data_2 = unconstrained_extrasyn[~nan_loc]
    corr_1 = met.nan_corr(target, data_1)[0]
    corr_2 = met.nan_corr(target, data_2)[0]

    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.scatter(data_1, target, color=plot_color['synap'])
    plt.title('correlation = ' + str(corr_1)[:6])
    plt.xlabel('model STAMs')
    plt.ylabel('measured STAM')
    plt.legend(['connectome-constrained'])
    plt.xlim([-2, 3])
    plt.ylim([-2, 12])

    ax = plt.subplot(1, 2, 2)
    plt.scatter(data_2, target, color=plot_color['unconstrained'])
    plt.title('correlation = ' + str(corr_2)[:6])
    plt.xlabel('model STAM')
    plt.legend(['unconstrained'])
    plt.xlim([-2, 3])
    plt.ylim([-2, 12])

    plt.savefig(fig_save_path / ('55_neurons.pdf'))

    plt.show()

    # plot average reconstruction over all data without normalization
    plt.figure()
    y_val = np.array(synap_irms_score)
    y_val_ci = np.stack(synap_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['anatomy'], plot_color['synap_randC'],  plot_color['synap']]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['synap connected', 'synap unconnected', 'synap extrasyn'], rotation=45)
    plt.ylabel('correlation to measured ' + weight_name)
    plt.tight_layout()

    plt.figure()
    y_val = np.array(uncon_irms_score)
    y_val_ci = np.stack(uncon_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['anatomy'], plot_color['synap_randC'],  plot_color['synap']]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['unconstrained connected', 'unconstrained unconnected', 'unconstrained extrasyn'], rotation=45)
    plt.ylabel('correlation to measured ' + weight_name)
    plt.tight_layout()

    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(synap_irms_score) / irms_baseline
    y_val_ci = np.stack(synap_irms_score_ci).T / irms_baseline
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['anatomy'], plot_color['synap_randC'],  plot_color['synap']]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['synap connected', 'synap unconnected', 'synap extrasyn'], rotation=45)
    plt.ylabel('relative correlation to measured ' + weight_name)
    plt.tight_layout()

    # plot average reconstruction over all data
    plt.figure()
    y_val = np.array(uncon_irms_score) / irms_baseline
    y_val_ci = np.stack(uncon_irms_score_ci).T / irms_baseline
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['anatomy'], plot_color['synap_randC'],  plot_color['synap']]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['unconstrained connected', 'unconstrained unconnected', 'unconstrained extrasyn'], rotation=45)
    plt.ylabel('relative correlation to measured ' + weight_name)
    plt.tight_layout()

    plt.show()

    return


def compare_model_irms(weights, masks, weight_name, cell_ids, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data
    train_weights = weights['data']['train'][weight_name].copy()
    test_weights = weights['data']['test'][weight_name].copy()
    num_neurons = train_weights.shape[0]

    test_weights[np.eye(num_neurons, dtype=bool)] = np.nan
    train_weights[np.eye(num_neurons, dtype=bool)] = np.nan

    # get the baseline correlation and IRMs. This is the best the model could have done
    # across all data
    irms_baseline = met.nan_corr(test_weights, train_weights)[0]

    model_irms_score = []
    model_irms_score_ci = []

    for m in ['synap', 'unconstrained', 'synap_randA']:
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = (
            met.nan_corr(weights['models'][m][weight_name], test_weights))
        model_irms_score.append(model_irms_to_measured_irms_test)
        model_irms_score_ci.append(model_irms_to_measured_irms_test_ci)

    # plot average reconstruction over all data
    y_limits = [0, 1.1]
    plt.figure()
    y_val = np.array(model_irms_score)
    y_val_ci = np.stack(model_irms_score_ci).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['synap'], plot_color['unconstrained'], plot_color['synap_randA']]
    plt.bar(plot_x, y_val / irms_baseline, color=bar_colors)
    plt.errorbar(plot_x, y_val / irms_baseline, y_val_ci / irms_baseline, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained', 'model\n+ scrambled anatomy'], rotation=45)
    plt.ylabel('% explainable correlation to measured ' + weight_name)
    plt.title('AND connectome constraint')
    plt.ylim(y_limits)
    plt.tight_layout()

    plt.show()

    plt.savefig(fig_save_path / ('measured_vs_model_randA_' + weight_name + '.pdf'))


def connected_unconnected_irms(weights, masks, weight_name, cell_ids, fig_save_path=None):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data
    train_weights = weights['data']['train'][weight_name].copy()
    test_weights = weights['data']['test'][weight_name].copy()
    num_neurons = train_weights.shape[0]

    test_weights[np.eye(num_neurons, dtype=bool)] = np.nan
    train_weights[np.eye(num_neurons, dtype=bool)] = np.nan

    # get the baseline correlation and IRMs. This is the best the model could have done
    # across all data
    chosen_mask_c = masks['synap']
    chosen_mask_c[np.eye(num_neurons, dtype=bool)] = False
    chosen_mask_uc = ~masks['synap']
    chosen_mask_uc[np.eye(num_neurons, dtype=bool)] = False
    irms_baseline_c, irms_baseline_c_ci = met.nan_corr(test_weights[chosen_mask_c], train_weights[chosen_mask_c])[:2]
    irms_baseline_uc, irms_baseline_uc_ci = met.nan_corr(test_weights[chosen_mask_uc], train_weights[chosen_mask_uc])[:2]

    model_irms_score_c = []
    model_irms_score_c_ci = []
    model_irms_score_uc = []
    model_irms_score_uc_ci = []
    for m in ['synap', 'unconstrained', 'synap_randA']:
        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
            met.nan_corr(weights['models'][m][weight_name][chosen_mask_c], test_weights[chosen_mask_c])
        model_irms_score_c.append(model_irms_to_measured_irms_test)
        model_irms_score_c_ci.append(model_irms_to_measured_irms_test_ci)

        model_irms_to_measured_irms_test, model_irms_to_measured_irms_test_ci = \
            met.nan_corr(weights['models'][m][weight_name][chosen_mask_uc], test_weights[chosen_mask_uc])
        model_irms_score_uc.append(model_irms_to_measured_irms_test)
        model_irms_score_uc_ci.append(model_irms_to_measured_irms_test_ci)

    # plot average reconstruction over all data
    y_limits = [0, 1.1]
    plt.figure()
    y_val = np.array(model_irms_score_c)
    y_val_ci = np.stack(model_irms_score_c_ci).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['synap'], plot_color['unconstrained'], plot_color['synap_randA']]
    plt.bar(plot_x, y_val / irms_baseline_c, color=bar_colors)
    plt.errorbar(plot_x, y_val / irms_baseline_c, y_val_ci / irms_baseline_c, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained', 'model\n+ scrambled anatomy'], rotation=45)
    plt.ylabel('relative correlation to measured ' + weight_name)
    plt.ylim(y_limits)
    plt.tight_layout()

    y_limits = [0, 1.1]
    plt.figure()
    y_val = np.array(model_irms_score_uc)
    y_val_ci = np.stack(model_irms_score_uc_ci).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['synap'], plot_color['unconstrained'], plot_color['synap_randA']]
    plt.bar(plot_x, y_val / irms_baseline_uc, color=bar_colors)
    plt.errorbar(plot_x, y_val / irms_baseline_uc, y_val_ci / irms_baseline_uc, fmt='none', color='k')
    plt.xticks(plot_x, labels=['model', 'model\n+ unconstrained', 'model\n+ scrambled anatomy'], rotation=45)
    plt.ylabel('relative correlation to measured ' + weight_name)
    plt.ylim(y_limits)
    plt.tight_layout()

    y_limits = [0, 0.5]
    plt.figure()
    y_val = np.array([irms_baseline_c, irms_baseline_uc])
    y_val_ci = np.stack([irms_baseline_c_ci, irms_baseline_uc_ci]).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['data'], plot_color['data']]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, labels=['connected pairs', 'unconnected pairs'], rotation=45)
    plt.ylabel('train-test correlation')
    plt.ylim(y_limits)
    plt.tight_layout()
    plt.savefig(fig_save_path / ('train_test_connect_unconnect.pdf'))

    plt.show()
    a=1


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

    num_meas = []
    # get the comparison between model prediction and data irm/correlation
    for m in weights['models']:
        if m in ['synap', 'unconstrained', 'synap_randA']:
        # if m in ['synap']:
            # for each model, calculate its score for both corr and IRM reconstruction across the n stim sweep
            model_name.append(m)
            model_irms_score_sweep.append(np.zeros(n_stim_sweep.shape[0]))
            model_irms_score_sweep_ci.append(np.zeros((2, n_stim_sweep.shape[0])))

            for ni, n in enumerate(n_stim_sweep):
                # mask the model predicted correlations and IRMs based on how many stimulation events were observed
                model_irms = weights['models'][m][weight_name].copy()
                model_irms[n_stim_mask[ni]] = np.nan

                if m == 'synap':
                    num_meas.append(np.sum(~np.isnan(model_irms)))

                model_irms_to_measured_irms, model_irms_to_measured_irms_ci = \
                    met.nan_corr(model_irms, weights['data']['test'][weight_name])
                model_irms_score_sweep[-1][ni] = model_irms_to_measured_irms
                model_irms_score_sweep_ci[-1][:, ni] = model_irms_to_measured_irms_ci




    # plot model reconstruction of IRMs
    y_limits = [0, 1.3]
    x_limits = [0, 14]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(n_stim_sweep, irms_baseline_sweep, irms_baseline_sweep_ci, label='explainable correlation', color=plot_color['data'])
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs, mcs_ci, label=n, color=plot_color[n])
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.xlabel('# of stimulation events')
    plt.ylabel('correlation to measured ' + weight_name)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(0, 0)
    for n, mcs, mcs_ci in zip(model_name, model_irms_score_sweep, model_irms_score_sweep_ci):
        plt.errorbar(n_stim_sweep, mcs / irms_baseline_sweep, mcs_ci / irms_baseline_sweep, label=n, color=plot_color[n])
    plt.xlabel('# of stimulation events')
    plt.ylabel('% explainable correlation to measured ' + weight_name)
    plt.xlim(x_limits)
    plt.ylim(y_limits)

    plt.tight_layout()
    plt.savefig(fig_save_path / ('measured_vs_model_' + weight_name + '_over_n.pdf'))

    plt.figure()
    plt.plot(n_stim_sweep, num_meas, marker='o')
    current_y = plt.ylim()
    plt.ylim((0, current_y[1]))
    plt.xlim(x_limits)
    plt.xlabel('# of stimulation events')
    plt.ylabel('number of neuron pairs')
    plt.savefig(fig_save_path / 'num_stim_events.pdf')

    plt.show()


def direct_vs_indirect(weights, masks, fig_save_path=None, rng=np.random.default_rng()):
    # this figure will demonstrate that the model can reconstruct the observed data correlation and IRMs
    # first we will sweep across the data and restrict to neuron pairs where a stimulation event was recorded N times
    # we will demonstrate that ratio between the best possible correlation and our model correlation remains constant
    # this suggests that this ratio is independent of number of data

    data_test_irms = weights['data']['test']['irms'].copy()
    data_train_irms = weights['data']['train']['irms'].copy()
    model_irms = weights['models']['synap']['irms'].copy()
    model_dirms = weights['models']['synap']['dirms'].copy()

    data_test_irfs = weights['data']['test']['irfs'].copy()
    data_train_irfs = weights['data']['train']['irfs'].copy()
    model_irfs = weights['models']['synap']['irfs'].copy()
    model_dirfs = weights['models']['synap']['dirfs'].copy()

    num_neurons = model_irms.shape[0]

    test_train_corr = met.nan_corr(data_test_irms, data_train_irms)[0]
    model_irms_corr, model_irms_corr_ci = met.nan_corr(data_test_irms, model_irms)
    model_dirms_corr, model_dirms_corr_ci = met.nan_corr(data_test_irms, model_dirms)

    n_boot = 1000
    alpha = 0.05
    p = met.two_sample_boostrap_corr_p(data_test_irms, model_irms, model_dirms, alpha=alpha, n_boot=n_boot, rng=rng)[0]

    y_val = np.array([model_irms_corr, model_dirms_corr]) / test_train_corr
    y_val_ci = np.stack([model_irms_corr_ci, model_dirms_corr_ci]).T / test_train_corr
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['synap'], plot_color['synap']]

    plt.figure()
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.ylabel('relative correlation')
    plt.xticks(plot_x, ['STAMs', 'direct STAMs'])
    plt.title('p = ' + str(p) + ', number of bootstrap samples = ' + str(n_boot))
    plt.tight_layout()

    plt.savefig(fig_save_path / 'direct_vs_indirect.pdf')

    # # find fraction of IRFs that are improved by including indirect information
    # model_irfs_score = np.zeros((num_neurons, num_neurons))
    # model_dirfs_score = np.zeros((num_neurons, num_neurons))
    # irm_nan_loc = np.isnan(model_irms)
    # for i in range(num_neurons):
    #     for j in range(num_neurons):
    #         model_irfs_score[i, j] = np.nansum((data_test_irfs[30:, i, j] - model_irfs[30:, i, j])**2)
    #         model_dirfs_score[i, j] = np.nansum((data_test_irfs[30:, i, j] - model_dirfs[30:, i, j])**2)
    #
    # model_irfs_score[irm_nan_loc] = np.nan
    # model_dirfs_score[irm_nan_loc] = np.nan
    # irf_dirf_diff = model_dirfs_score - model_irfs_score
    # irf_dirf_diff = irf_dirf_diff[~np.isnan(irf_dirf_diff)]
    # print(np.mean(irf_dirf_diff < 0))
    #
    # plt.figure()
    # axis_max = np.nanpercentile(irf_dirf_diff, 80)
    # plot_lim = (-axis_max, axis_max)
    # bin_edges = np.linspace(plot_lim[0], plot_lim[1], 101)
    # plt.hist(irf_dirf_diff, density=True, bins=bin_edges)
    # plt.xlabel('squared error')
    # plt.ylabel('density')
    # plt.title('direct STA - STA')
    # plt.xlim(plot_lim)

    plt.show()

    return


def weights_vs_connectome(weights, masks, fig_save_path=None):
    # pull out the weights we will compare to the synapse counts
    gap_counts = weights['anatomy']['gap_conn']
    chem_counts = weights['anatomy']['chem_conn']
    # chem_size = weights['anatomy']['chem_size']
    gap_mask = masks['gap']
    chem_mask = masks['chem']
    diag_bool = np.eye(gap_mask.shape[0], dtype=bool)
    gap_mask[diag_bool] = False
    chem_mask[diag_bool] = False

    data_irms = np.abs(weights['data']['train']['irms'])
    model_uncon_weights = np.abs(weights['models']['unconstrained']['weights'])
    model_synap_weights = np.abs(weights['models']['synap']['weights'])

    # sample the weights using the connectome
    synapse_counts_gap = gap_counts[gap_mask]
    data_irms_gap = data_irms[gap_mask]
    model_uncon_weights_gap = model_uncon_weights[gap_mask]
    model_synap_weights_gap = model_synap_weights[gap_mask]

    synapse_counts_chem = chem_counts[chem_mask]
    # synapse_size_chem = chem_size[chem_mask]
    data_irms_chem = data_irms[chem_mask]
    model_uncon_weights_chem = model_uncon_weights[chem_mask]
    model_synap_weights_chem = model_synap_weights[chem_mask]

    # compare all the weights to the synapse counts
    data_irms_gap_corr, data_irms_gap_corr_ci = met.nan_corr(synapse_counts_gap, data_irms_gap)
    model_uncon_gap_corr, model_uncon_gap_corr_ci = met.nan_corr(synapse_counts_gap, model_uncon_weights_gap)
    model_synap_gap_corr, model_synap_gap_corr_ci = met.nan_corr(synapse_counts_gap, model_synap_weights_gap)

    data_irms_chem_corr, data_irms_chem_corr_ci = met.nan_corr(synapse_counts_chem, data_irms_chem)
    model_uncon_chem_corr, model_uncon_chem_corr_ci = met.nan_corr(synapse_counts_chem, model_uncon_weights_chem)
    model_synap_chem_corr, model_synap_chem_corr_ci = met.nan_corr(synapse_counts_chem, model_synap_weights_chem)

    # data_irms_chem_size_corr, data_irms_chem_size_corr_ci = met.nan_corr(synapse_size_chem, data_irms_chem)
    # model_uncon_chem_size_corr, model_uncon_chem_size_corr_ci = met.nan_corr(synapse_size_chem, model_uncon_weights_chem)
    # model_synap_chem_size_corr, model_synap_chem_size_corr_ci = met.nan_corr(synapse_size_chem, model_synap_weights_chem)

    plt.figure()
    plt.scatter(np.log(model_synap_weights_gap), np.log(synapse_counts_gap), color=plot_color['synap'])
    # plt.hexbin(np.log(model_synap_weights_gap), np.log(synapse_counts_gap), gridsize=10)
    plt.title('correlation = ' + str(model_synap_gap_corr))
    plt.xlabel('log model weights')
    plt.ylabel('log gap junction counts')

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'model_vs_gap_scatter.pdf')

    plt.figure()
    plt.scatter(np.log(model_synap_weights_chem), np.log(synapse_counts_chem), color=plot_color['synap'])
    # plt.hexbin(np.log(model_synap_weights_chem), np.log(synapse_counts_chem), gridsize=10)
    plt.title('correlation = ' + str(model_synap_chem_corr))
    plt.xlabel('log model weights')
    plt.ylabel('log chemical synapse counts')

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'model_vs_chem_scatter.pdf')

    y_val = np.array([data_irms_gap_corr, model_uncon_gap_corr, model_synap_gap_corr])
    y_val_ci = np.stack([data_irms_gap_corr_ci, model_uncon_gap_corr_ci, model_synap_gap_corr_ci]).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['anatomy'], plot_color['unconstrained'],  plot_color['synap']]
    plt.figure()
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.ylabel('correlation')
    plt.title('electrical synapse count')
    plt.xticks(plot_x, ['data STAMs', 'unconstrained', 'model'], rotation=45)
    plt.ylim((-0.1, 0.3))
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'weights_vs_gap.pdf')

    y_val = np.array([data_irms_chem_corr, model_uncon_chem_corr, model_synap_chem_corr])
    y_val_ci = np.stack([data_irms_chem_corr_ci, model_uncon_chem_corr_ci, model_synap_chem_corr_ci]).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color['anatomy'], plot_color['unconstrained'],  plot_color['synap']]
    plt.figure()
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.xticks(plot_x, ['data STAMs', 'unconstrained', 'model'], rotation=45)
    plt.ylabel('correlation')
    plt.title('chemical synapse count')
    plt.ylim((-0.1, 0.3))
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'weights_vs_chem.pdf')

    plt.show()

    return


def uncon_vs_connectome(weights, masks, fig_save_path=None):
    # model_weights = np.abs(weights['models']['unconstrained']['eirms'])
    model_weights = np.abs(weights['models']['unconstrained']['weights'])
    data_irms = np.abs(weights['data']['train']['irms'])
    nan_loc_data = np.isnan(data_irms)
    nan_loc_model = np.isnan(model_weights)

    connectome_mask = masks['synap']
    connectome_mask[np.eye(connectome_mask.shape[0], dtype=bool)] = False
    sparsity = np.mean(connectome_mask)
    cutoffs = np.linspace(0, 99, 100)

    aprf_data = np.zeros((cutoffs.shape[0], 4))
    aprf_uncon = np.zeros((cutoffs.shape[0], 4))

    for ci, c in enumerate(cutoffs):
        data_irms_cutoff = np.nanpercentile(data_irms, c)
        data_mask = (data_irms > data_irms_cutoff).astype(float)
        data_mask[nan_loc_data] = np.nan

        aprf_data[ci, 0] = met.accuracy(connectome_mask, data_mask)
        aprf_data[ci, 1] = met.precision(connectome_mask, data_mask)
        aprf_data[ci, 2] = met.recall(connectome_mask, data_mask)
        aprf_data[ci, 3] = met.f_measure(connectome_mask, data_mask)

        model_weight_cutoff = np.nanpercentile(model_weights, c)
        model_mask = (model_weights > model_weight_cutoff).astype(float)
        data_mask[nan_loc_model] = np.nan

        aprf_uncon[ci, 0] = met.accuracy(connectome_mask, model_mask)
        aprf_uncon[ci, 1] = met.precision(connectome_mask, model_mask)
        aprf_uncon[ci, 2] = met.recall(connectome_mask, model_mask)
        aprf_uncon[ci, 3] = met.f_measure(connectome_mask, model_mask)

    # plot the sparsity sweep of the data
    plt.figure()
    plt.plot(aprf_data[:, 3])
    plt.axvline(100 * (1 - sparsity), color='k', linestyle='--', label='connectome')
    # plt.legend(['accuracy', 'precision', 'recall', 'f measure'])
    plt.xlabel('sparsity')
    plt.ylabel('f measure, similarity to connectome')
    plt.savefig(fig_save_path / 'data_sparsity_sweep.pdf')

    plt.figure()
    plt.plot(aprf_uncon)
    plt.axvline(100 * (1 - sparsity), color='k', linestyle='--')
    plt.legend(['accuracy', 'precision', 'recall', 'f measure'])

    # plot weight magnitude for connected and unconnected neurons
    plt.figure()
    plt.hist(np.abs(model_weights[connectome_mask].reshape(-1)), bins=100, density=True, label='connected pairs', alpha=0.5)
    plt.hist(np.abs(model_weights[~connectome_mask].reshape(-1)), bins=100, density=True, label='unconnected pairs', alpha=0.5)
    plt.xlim(0, 0.2)
    plt.xlabel('|model weight|')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(fig_save_path / 'weights_connected_vs_unconnected.pdf')

    plt.show()

    return


def uncon_vs_synap(models, fig_save_path=None):
    synap_weights = models['synap'].dynamics_weights
    uncon_weights = models['unconstrained'].dynamics_weights

    synap_eig, synap_eig_vect = np.linalg.eig(synap_weights)
    uncon_eig, uncon_eig_vect = np.linalg.eig(uncon_weights)

    plt.figure()
    plt.plot(np.sort(np.abs(synap_eig))[::-1], label='constrained')
    plt.plot(np.sort(np.abs(uncon_eig))[::-1], label='unconstrained')
    plt.legend()

    # plot the eigenvalues real vs imaginary components
    x_circ = np.cos(np.linspace(0, 2 * np.pi, 100))
    y_circ = np.sin(np.linspace(0, 2 * np.pi, 100))

    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.scatter(np.real(synap_eig), np.imag(synap_eig))
    plt.plot(x_circ, y_circ)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_aspect('equal', 'box')
    plt.title('constrained')

    ax = plt.subplot(1, 2, 2)
    plt.scatter(np.real(uncon_eig), np.imag(uncon_eig))
    plt.title('unconstrained')
    plt.plot(x_circ, y_circ)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_aspect('equal', 'box')

    plt.show()

    return


def plot_model_eig(models, fig_save_path=None):
    model_weights = models['synap'].dynamics_weights.copy()
    cell_ids = models['synap'].cell_ids

    synap_eig_val, synap_eig_vect = np.linalg.eig(model_weights)

    eig_sort_inds = np.argsort(np.abs(synap_eig_val))[::-1]

    synap_eig_val = synap_eig_val[eig_sort_inds]
    synap_eig_vect = synap_eig_vect[:, eig_sort_inds]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(synap_eig_val), color=plot_color['synap'], label='constrained')
    plt.xlabel('eigenvalue #')
    plt.ylabel('eigenvalue magnitude')

    # plot the eigenvalues real vs imaginary components
    x_circ = np.cos(np.linspace(0, 2 * np.pi, 100))
    y_circ = np.sin(np.linspace(0, 2 * np.pi, 100))

    ax = plt.subplot(1, 2, 2)
    plt.scatter(np.real(synap_eig_val), np.imag(synap_eig_val), color=plot_color['synap'])
    plt.plot(x_circ, y_circ)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_aspect('equal', 'box')
    plt.title('constrained')
    plt.tight_layout()

    plt.savefig(fig_save_path / 'model_eigenvalues.pdf')

    vect_cutoff = 95  # % percent
    complex_vect_ind = np.where(np.imag(synap_eig_val) > 0.25)[0][0]
    chosen_vect_inds = [0, 1, complex_vect_ind]

    for vi, v in enumerate(chosen_vect_inds):
        chosen_vect_var = np.abs(synap_eig_vect[:, v])**2

        cutoff = np.percentile(chosen_vect_var, vect_cutoff)
        cutoff = 0.005

        large_weights_ind = np.where(chosen_vect_var > cutoff)[0]
        large_weights_cell_id = [cell_ids[i] for i in large_weights_ind]

        plot_x = np.arange(synap_eig_vect.shape[0])
        plt.figure()
        plt.scatter(plot_x, chosen_vect_var, color=plot_color['synap'])
        plt.ylabel('|eigenvector|^2')
        plt.xlabel('eigenvector component')
        for ind, id in zip(large_weights_ind, large_weights_cell_id):
            plt.text(ind + 2, chosen_vect_var[ind] + 0.007, id)

        plt.savefig(fig_save_path / ('model_eigenvector_' + str(vi) + '.pdf'))

        sub_weights = model_weights.copy()
        sub_weights = sub_weights[:, large_weights_ind][large_weights_ind, :]
        sub_weights[np.eye(sub_weights.shape[0], dtype=bool)] = np.nan
        cmax = np.nanmax(np.abs(sub_weights))

        plot_x = np.arange(sub_weights.shape[0])
        plt.figure()
        plt.imshow(sub_weights, interpolation='nearest', cmap=colormap)
        plt.clim((-cmax, cmax))
        plt.colorbar()
        plt.xticks(plot_x, large_weights_cell_id, rotation=90)
        plt.yticks(plot_x, large_weights_cell_id)

        plt.savefig(fig_save_path / ('model_eigenvector_' + str(vi) + '_network.pdf'))

    plt.show()

    return


def compare_model_vs_connectome_eig(models, masks, data, cell_ids, num_vect_plot=5, neuron_freq=0.1):
    anatomy = au.load_anatomical_data(cell_ids=cell_ids['all'])

    model = models['synap']
    chem_conn = anatomy['chem_conn']
    gap_conn = anatomy['gap_conn']

    is_gaba = au.get_neurotransmitters(cell_ids['all'])
    chem_with_gaba = chem_conn.copy()
    chem_with_gaba[:, is_gaba] *= -1
    laplacian = gap_conn - np.diag(np.sum(gap_conn, axis=0))

    model_A = model.dynamics_weights
    anatomy_A = np.diag(model_A.diagonal()) + (laplacian / 1000 + chem_with_gaba / 100)
    anatomy_A = scipy.linalg.expm(anatomy_A)

    eig_vals_model, eig_vects_model = np.linalg.eig(model_A)
    eig_vals_anatomy, eig_vects_anatomy = np.linalg.eig(anatomy_A)

    sorted_eig_vals_model = np.sort(np.abs(eig_vals_model))[::-1]
    sorted_eig_vals_anatomy = np.sort(np.abs(eig_vals_anatomy))[::-1]

    plt.figure()
    plt.plot(sorted_eig_vals_model, label='model')
    plt.plot(sorted_eig_vals_anatomy, label='anatomy')
    plt.legend()

    x_circ = np.cos(np.linspace(0, 2 * np.pi, 100))
    y_circ = np.sin(np.linspace(0, 2 * np.pi, 100))

    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.scatter(np.real(eig_vals_model), np.imag(eig_vals_model))
    plt.plot(x_circ, y_circ)
    ax.set_aspect('equal')
    plt.title('model')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    ax = plt.subplot(1, 2, 2)
    plt.scatter(np.real(eig_vals_anatomy), np.imag(eig_vals_anatomy))
    plt.plot(x_circ, y_circ)
    ax.set_aspect('equal')
    plt.title('anatomy')
    # plt.xlim((-1, 1))
    # plt.ylim((-1, 1))

    plt.show()


def plot_eigenvalues_find_enrichment(models, masks, data, cell_ids, num_vect_plot=5, neuron_freq=0.1):
    model = models['synap']
    cell_types, cell_type_labels = au.get_neuron_types(cell_ids['all'])

    # remove neurons that were measured infrequently
    measured_neurons = np.stack([np.mean(np.isnan(i), axis=0) <= 0.5 for i in data])
    measured_freq = np.mean(measured_neurons, axis=0)
    neurons_to_keep = measured_freq >= neuron_freq
    neurons_to_keep_tiled = np.tile(neurons_to_keep, model.dynamics_lags)

    # restrict the A matrix to neurons that were measured a lot
    A = model.dynamics_weights[:, neurons_to_keep_tiled][neurons_to_keep_tiled, :]
    cell_ids_all = [cell_ids['all'][i] for i in range(len(neurons_to_keep)) if neurons_to_keep[i]]
    cell_types = cell_types[neurons_to_keep, :]
    cell_type_count = np.sum(cell_types, axis=0)

    # calculate the eignvalues / vectors of the dynamics matrix
    eig_vals, eig_vects = np.linalg.eig(A)

    # sort to get the largest
    sort_inds = np.argsort(np.abs(eig_vals))[::-1]

    eig_vals = eig_vals[sort_inds]
    eig_vals_abs = np.abs(eig_vals)
    eig_vects = eig_vects[:, sort_inds]
    
    # average together eigenvectors across lags
    eig_vects_stacked = np.stack(np.split(eig_vects, model.dynamics_lags, axis=0))
    eig_vects_comb_real = np.mean(np.real(eig_vects_stacked), axis=0)
    eig_vects_comb_abs = np.mean(np.abs(eig_vects_stacked), axis=0)

    plt.figure()
    plt.plot(eig_vals_abs)
    plt.xlabel('eigenvalue #')
    plt.ylabel('eigenvalue magnitude')

    # plot eigenvalue magnitude in decreasing magnitude
    fig = plt.figure()
    x_circ = np.cos(np.linspace(0, 2 * np.pi, 100))
    y_circ = np.sin(np.linspace(0, 2 * np.pi, 100))

    # plot the eigenvalues real vs imaginary components
    ax = fig.add_subplot()
    plt.scatter(np.real(eig_vals), np.imag(eig_vals))
    plt.plot(x_circ, y_circ)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_aspect('equal', 'box')
    plt.xlabel('real[eigenvalues]')
    plt.ylabel('imag[eigenvalues]')

    # loop through each eigenvector and plot the top num_neuron_plot entries
    ylim_max = np.max(np.abs(eig_vects_comb_real))
    ylim_plot = (-ylim_max, ylim_max)
    eig_cell_type = np.zeros(eig_vects_comb_abs.shape[1])
    num_synap = np.zeros((eig_vects_comb_abs.shape[1], 2))

    for n in range(eig_vects_comb_abs.shape[1]):
        cell_ids_plot = cell_ids_all.copy()
        this_eig_vect_comb_real = eig_vects_comb_real[:, n]
        this_eig_vect_comb_abs = eig_vects_comb_abs[:, n]
        this_eig_vect_stacked = eig_vects_stacked[:, :, n]

        cell_sort_inds = np.argsort(this_eig_vect_comb_abs)[::-1]

        this_eig_vect_comb_abs = this_eig_vect_comb_abs[cell_sort_inds]
        this_eig_vect_comb_real = this_eig_vect_comb_real[cell_sort_inds]
        this_eig_vect_stacked = this_eig_vect_stacked[:, cell_sort_inds]
        this_eig_cell_types = cell_types[cell_sort_inds, :]

        eig_vect_cdf = np.cumsum(this_eig_vect_comb_abs) / np.sum(this_eig_vect_comb_abs)
        cutoff = eig_vect_cdf > 0.5
        num_neuron_plot = np.where(cutoff)[0][0] + 1
        plot_x = np.arange(num_neuron_plot)

        cell_ids_plot = [cell_ids_plot[i] for i in cell_sort_inds[:num_neuron_plot]]

        this_eig_vect_comb_abs = this_eig_vect_comb_abs[:num_neuron_plot]
        this_eig_vect_comb_real = this_eig_vect_comb_real[:num_neuron_plot]
        this_eig_vect_stacked = this_eig_vect_stacked[:, :num_neuron_plot]
        this_eig_cell_types = this_eig_cell_types[:num_neuron_plot, :]
        eig_cell_type[n] = np.argmax(np.sum(this_eig_cell_types, axis=0))

        # get the subnetwork defined by the eigenvector
        neuron_inds = [cell_ids['all'].index(i) for i in cell_ids_plot]
        network = masks['chem'] | masks['gap']
        network_chem = masks['chem']
        network_gap = masks['gap']
        network[np.eye(network.shape[0], dtype=bool)] = False
        network = network[neuron_inds, :][:, neuron_inds]
        network_chem = network_chem[neuron_inds, :][:, neuron_inds]
        network_gap = network_gap[neuron_inds, :][:, neuron_inds]

        num_synap[n, 0] = np.sum(network_gap)
        num_synap[n, 1] = np.sum(network_chem)

        if n < num_vect_plot:
            # plot the eigenvector
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.axhline(0, color='k', linestyle='--')
            plt.scatter(plot_x, this_eig_vect_comb_abs[:num_neuron_plot])
            plt.ylim(ylim_plot)

            plt.subplot(3, 1, 2)
            plt.axhline(0, color='k', linestyle='--')
            plt.scatter(plot_x, this_eig_vect_comb_real[:num_neuron_plot])
            plt.ylim(ylim_plot)

            plt.subplot(3, 1, 3)
            plt.imshow(np.real(this_eig_vect_stacked[:, :num_neuron_plot]))
            plt.xticks(plot_x, cell_ids_plot, rotation=90)
            plt.ylabel('time lags')

    # define 3 regions
    # 1 large eig, real
    # 2 medium eig, real
    # 3 small eig, real
    # 4 medium eig, complex 45 deg
    # 5 medium eig, complex 90 deg

    # define the eigenvalues in each section
    section_names = ['large real', 'mid real', 'mid complex 45', 'mid complex 90']
    # section_names = ['large real', 'mid real', 'small real', 'mid complex 45', 'mid complex 90']
    def get_angle(eig_vals):
        real_part = np.real(eig_vals)
        abs_imag_part = np.abs(np.imag(eig_vals))
        return (np.arctan(abs_imag_part / real_part) % np.pi) * 180 / np.pi

    real_cutoff = 0.125
    angle_cutoff = 60

    eig_val_angle = get_angle(eig_vals)
    sections = [(np.abs(eig_vals) > 0.95) & (np.abs(np.imag(eig_vals)) < real_cutoff),
                (np.abs(eig_vals) < 0.95) & (np.abs(np.imag(eig_vals)) < real_cutoff),
                #(np.abs(eig_vals) < 0.5) & (np.imag(eig_vals) < real_cutoff),
                (np.abs(eig_vals) > 0.4) & (np.abs(eig_vals) < 0.9) & (eig_val_angle > 15) & (eig_val_angle < angle_cutoff),
                (np.abs(eig_vals) > 0.4) & (np.abs(eig_vals) < 0.9) & (eig_val_angle > angle_cutoff) & (eig_val_angle < 150)]

    # plot the eigenvalues in their sections
    fig = plt.figure()
    ax = fig.add_subplot()
    for ii, i in enumerate(sections):
        selected_vals = eig_vals[i]
        plt.scatter(np.real(selected_vals), np.imag(selected_vals), label=section_names[ii])

    plt.plot(x_circ, y_circ)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    ax.set_aspect('equal', 'box')
    plt.legend()

    # get the counts of cell type in each section
    bin_edges = np.linspace(-0.5, 4.5, 6)
    all_counts = np.histogram(eig_cell_type, bins=bin_edges)[0].astype(float)

    section_counts = []
    synapse_counts = []

    num_synap[:, 0] = num_synap[:, 0] > num_synap[:, 1]
    num_synap[:, 1] = 1 - num_synap[:, 0]
    for i in sections:
        section_counts.append(np.histogram(eig_cell_type[i], bins=bin_edges)[0].astype(float))
        section_counts[-1] /= (all_counts * np.sum(i))
        synapse_counts.append(np.sum(num_synap[i, :], axis=0))
        synapse_counts[-1] /= (np.sum(num_synap, axis=0) * np.sum(i))

    chance = 1 / eig_vals.shape[0]

    # plot cell types by group
    plt.figure()
    plot_x = np.arange(len(section_counts))
    plt.figure()
    plt.bar(plot_x, [np.mean(i) for i in section_counts])
    plt.xticks(plot_x, section_names)

    plt.figure()
    plot_x = np.arange(bin_edges.shape[0] - 1)
    for ii, i in enumerate(section_counts):
        plt.scatter(plot_x, i / chance, label=section_names[ii])

    plt.axhline(1, label='chance', color='k', linestyle='--')
    plt.ylim([0, plt.ylim()[1]])
    plt.xticks(np.arange(len(cell_type_labels)), cell_type_labels)
    plt.legend()

    # plot synapse count by group
    synapse_names = ['gap', 'chemical']
    plot_x = [1, 2]
    plt.figure()
    # plt.subplot(1, 2, 1)
    plt.bar(plot_x, np.mean(num_synap, axis=0))
    plt.xticks(plot_x, synapse_names)

    plt.figure()
    # plt.subplot(1, 2, 2)
    plot_x = np.arange(2)
    for ii, i in enumerate(synapse_counts):
        plt.scatter(plot_x, i / chance, label=section_names[ii])

    plt.axhline(1, label='chance', color='k', linestyle='--')
    plt.ylim([0, plt.ylim()[1]])
    plt.xticks(np.arange(len(synapse_names)), synapse_names)
    plt.legend()

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
        cell_ids_sub = cell_ids['sorted'][:num_neurons]
        # TODO remove this? was added to force SAADL into the list
        cell_ids_sub[-1] = 'SAADL'
        cell_ids_plot = sorted(cell_ids_sub)
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
    plt.colorbar()
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
    plt.colorbar()
    plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
    plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
    corr_max_all = np.nanmax(np.abs(data_corr))
    corr_limits = (-corr_max_all, corr_max_all)
    plt.clim(corr_limits)
    plt.title('data correlation matrix')
    if num_neurons is None:
        plt.savefig(fig_save_path / 'full_data_corr.pdf')
    else:
        plt.savefig(fig_save_path / 'sampled_data_corr.pdf')

    # get the color limits
    irm_max_all = 0
    # corr_max_all = 0
    for m in ['synap', 'unconstrained', 'synap_randA']:
        model_irms = weights['models'][m]['irms'].copy()

        model_irms = model_irms[:, not_all_nan][not_all_nan, :]
        # model_irms = model_irms[np.ix_(new_order, new_order)]

        model_irms[np.eye(model_irms.shape[0], dtype=bool)] = np.nan

        model_irms = model_irms[np.ix_(neuron_inds, neuron_inds)]

        irm_max = np.nanmax(np.abs(model_irms))

        if irm_max > irm_max_all:
            irm_max_all = irm_max

    irm_limits = (-irm_max_all, irm_max_all)

    for m in ['synap', 'unconstrained', 'synap_randA']:
        neurons_to_mask = []
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
        plt.colorbar()
        plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
        plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
        plt.clim(irm_limits)
        plt.title(m + ' IRMs')
        if num_neurons is None:
            plt.savefig(fig_save_path / ('full_data_irms_' + m + '.pdf'))
        else:
            plt.savefig(fig_save_path / ('sampled_data_irms_' + m + '.pdf'))

        plt.figure()
        plt.imshow(model_corr, interpolation='nearest', cmap=colormap)
        plt.colorbar()
        plt.xticks(plot_x, labels=cell_ids_plot, size=font_size, rotation=90)
        plt.yticks(plot_x, labels=cell_ids_plot, size=font_size)
        plt.clim(corr_limits)
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
        plt.title(cell_ids[plot_ind, 1] + ' -> ' + cell_ids[plot_ind, 0])
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


def plot_silencing_results(model, cell_ids, weights, fig_save_path=None, silence_type='hand_picked'):
    neuron_pair = ['RMDDR', 'RMDDL']
    silenced_neurons = ['SAADR']
    # silenced_neurons = ['RMDDL']
    # silenced_neurons = ['RMDDL', 'SAADR']
    # silenced_neurons = ['SAADR', 'RMDDR', 'RMDDL']
    # silenced_neurons = ['SAADR', 'RMDDL']

    neuron_pair_inds = [cell_ids['all'].index(i) for i in neuron_pair]
    silenced_neurons_inds = [cell_ids['all'].index(i) for i in silenced_neurons]

    # silence each neuron and calculate the irf
    import copy
    silenced_model = copy.deepcopy(model)

    if silence_type == 'cell':
        silenced_model = ssmu.get_silenced_model(model, silenced_neurons)

    if silence_type == 'all_other_cell':
        cell_ids_copy = copy.copy(cell_ids['all'])
        for sn in silenced_neurons + neuron_pair:
            cell_ids_copy.pop(cell_ids_copy.index(sn))
        silenced_model = ssmu.get_silenced_model(model, cell_ids_copy)

    elif silence_type == 'synapse':
        for sn in silenced_neurons_inds:
            silenced_model.dynamics_weights[neuron_pair_inds[0], sn] = 0

    elif silence_type == 'all_other_incoming_synapse':
        all_inds_no_diag = np.arange(silenced_model.dynamics_dim)
        all_inds_no_diag = np.setdiff1d(all_inds_no_diag, [neuron_pair_inds[0]] + silenced_neurons_inds)
        silenced_model.dynamics_weights[neuron_pair_inds[0], all_inds_no_diag] = 0

    elif silence_type == 'handpicked':
        rmddl_ind = cell_ids['all'].index('RMDDL')
        saadr_ind = cell_ids['all'].index('SAADR')
        silenced_model.dynamics_weights[saadr_ind, rmddl_ind] = 0

    elif silence_type == 'all_other_synapse':
        silenced_model.dynamics_weights = np.diag(np.diag(silenced_model.dynamics_weights))

        # enable specific connections
        riar_ind = cell_ids['all'].index('RIAR')
        rmddl_ind = cell_ids['all'].index('RMDDL')
        rmddr_ind = cell_ids['all'].index('RMDDR')
        saadr_ind = cell_ids['all'].index('SAADR')

        # silenced_model.dynamics_weights[riar_ind, rmddl_ind] = model.dynamics_weights[riar_ind, rmddl_ind]
        # silenced_model.dynamics_weights[saadr_ind, riar_ind] = model.dynamics_weights[saadr_ind, riar_ind]
        # silenced_model.dynamics_weights[rmddr_ind, saadr_ind] = model.dynamics_weights[rmddr_ind, saadr_ind]

        # silenced_model.dynamics_weights[rmddl_ind, riar_ind] = model.dynamics_weights[rmddl_ind, riar_ind]
        # silenced_model.dynamics_weights[riar_ind, saadr_ind] = model.dynamics_weights[riar_ind, saadr_ind]
        # silenced_model.dynamics_weights[saadr_ind, rmddr_ind] = model.dynamics_weights[saadr_ind, rmddr_ind]

        # silenced_model.dynamics_weights[rmddr_ind, rmddl_ind] = model.dynamics_weights[rmddr_ind, rmddl_ind]
        # silenced_model.dynamics_weights[rmddl_ind, rmddr_ind] = model.dynamics_weights[rmddl_ind, rmddr_ind]

        # ALL three of these connections are necessary for the large response
        # the loop between RMDDR and SAADR amplifies incoming signals
        silenced_model.dynamics_weights[rmddr_ind, saadr_ind] = model.dynamics_weights[rmddr_ind, saadr_ind]
        silenced_model.dynamics_weights[saadr_ind, rmddr_ind] = model.dynamics_weights[saadr_ind, rmddr_ind]
        # silenced_model.dynamics_weights[rmddr_ind, rmddl_ind] = model.dynamics_weights[rmddr_ind, rmddl_ind]

        # does the link from rmddl to saadr matter?
        silenced_model.dynamics_weights[saadr_ind, rmddl_ind] = model.dynamics_weights[saadr_ind, rmddl_ind]

        # reenable the direct connection
        # silenced_model.dynamics_weights[neuron_pair_inds[0], neuron_pair_inds[1]] = model.dynamics_weights[neuron_pair_inds[0], neuron_pair_inds[1]]

        # for each silenced neuron, enable the path from the stim to unsilenced neuron
        # and unsilenced neuron to responding neuron
        # for sn in silenced_neurons_inds:
        #     # silenced_model.dynamics_weights[neuron_pair_inds[0], sn] = model.dynamics_weights[neuron_pair_inds[0], sn]
        #     # silenced_model.dynamics_weights[sn, neuron_pair_inds[1]] = model.dynamics_weights[sn, neuron_pair_inds[1]]
        #     silenced_model.dynamics_weights[:, sn] = model.dynamics_weights[:, sn]
        #     silenced_model.dynamics_weights[sn, :] = model.dynamics_weights[sn, :]

        # silenced_model.dynamics_weights[silenced_neurons_inds[1], neuron_pair_inds[1]] = 0

    silenced_model_irfs = ssmu.calculate_irfs(silenced_model)

    default_irf = weights['irfs'][:, neuron_pair_inds[0], neuron_pair_inds[1]]
    silenced_irf = silenced_model_irfs[:, neuron_pair_inds[0], neuron_pair_inds[1]]
    plt.figure()
    plt.plot(default_irf)
    plt.plot(silenced_irf)
    plt.show()

    a=1


def break_down_irf(model, weights, masks, cell_ids, window, fig_save_path=None):
    # format is [responding neuron, stimulated neuron]
    chosen_pairs = np.array([['AVAL', 'AVEL'],
                             ['AVDL', 'AIML']])
                             #['RMDDR', 'RMDDL']])
    # size multiplier to make it look better in illustrator
    i_mult = 0.5
    fontsize = 12 * i_mult
    n_best_connections = 4
    num_neurons = model.dynamics_dim
    sample_rate = model.sample_rate
    num_t = int(window[1] * sample_rate)

    # here we are going to go through every incoming synapse to the responding neuron and individually
    model_weights = model.dynamics_weights.copy()

    broken_down_irfs = []  # list of the IRFs when each synapse is silenced
    top_cell_ids = []  # list of the synapses that most contributed
    data = []  # list of the actual measured perturbation response
    width_mult = [[50 * i_mult, 0.25 * i_mult],
                  [100 * i_mult, 1 * i_mult]]

    # find the postsynaptic sites that contribute most in the responding neuron when the stimulated neuron is activated
    for pair_ind, (resp, stim) in enumerate(zip(chosen_pairs[:, 0], chosen_pairs[:, 1])):
        resp_ind = cell_ids['all'].index(resp)
        stim_ind = cell_ids['all'].index(stim)
        data.append(weights['data']['test']['irfs'][:, resp_ind, stim_ind])

        # get the magnitude of weights for paths of length 2 between stim and resp cells
        model_weights_2_steps_all = model_weights[:, stim_ind] * model_weights[resp_ind, :]
        connection_inds = model_weights_2_steps_all.nonzero()[0]
        connection_inds = np.setdiff1d(connection_inds, (stim_ind, resp_ind))
        model_weights_2_steps = model_weights_2_steps_all[connection_inds]

        # get the largest responses
        # keep adding in more synapses and calculating the IRF
        num_t_all = int(np.sum(window) * sample_rate)

        # IRF components is each IRF with a subset of the synapses active
        # save 1 extra spot for including the direct path
        # save 1 extra spot for the response including all synapses
        # +2 total

        irf_components = np.zeros((num_t_all, n_best_connections + 2))
        sorted_connections = connection_inds[np.argsort(model_weights_2_steps)[::-1]]  # list of connections sorted by response size

        # get the top best connections and their cell IDs
        # make sure you always include the direct connection
        top_resp_inds = [stim_ind] + list(sorted_connections[:n_best_connections])
        top_cell_ids.append([cell_ids['all'][i] for i in top_resp_inds])

        # disable all incoming synapses onto the responding cell
        # loop through and enable them one by one
        enabled_synapses = [resp_ind]  # start with the responding cells self term enabled
        inputs = np.zeros((num_t, num_neurons))
        inputs[0, stim_ind] = 1
        for tri, tr in enumerate(top_resp_inds):
            new_model = deepcopy(model)

            enabled_synapses.append(tr)  # add in the next synapse in the list
            # get the indicies of every potential incoming connection
            # silence all neurons not in best_connections by deleting them from the array
            all_resp_inds = np.arange(model.dynamics_dim)
            all_resp_inds = np.setdiff1d(all_resp_inds, enabled_synapses)
            new_model.dynamics_weights[resp_ind, all_resp_inds] = 0

            irf_components[-num_t:, tri] = new_model.sample(num_time=num_t, inputs=inputs, add_noise=False)['emissions'][:, resp_ind]

        # add in the normal model response
        irf_components[-num_t:, -1] = model.sample(num_time=num_t, inputs=inputs, add_noise=False)['emissions'][:, resp_ind]
        top_cell_ids[-1].append('all')

        broken_down_irfs.append(irf_components)

        # now we want to plot the graph of the network
        # get the outgoing connections from the stimulating cell
        top_resp_inds = np.array(top_resp_inds[1:])  # get rid of the direct connection

        # get the outgoing/incoming weights from the stimulated cell from the model
        # extra index for the direct weight
        model_weight_network = np.zeros((top_resp_inds.shape[0]+2, 2))
        model_weight_network[0, 0] = model.dynamics_weights[resp_ind, stim_ind]
        model_weight_network[1:-1, 0] = np.abs(model.dynamics_weights[top_resp_inds, stim_ind])
        model_weight_network[1:-1, 1] = np.abs(model.dynamics_weights[resp_ind, top_resp_inds])

        # get the outgoing/incoming weights from the stimulated cell from the connectome
        connectome = weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn']
        conn_weights = np.zeros((top_resp_inds.shape[0] + 2, 2))
        conn_weights[0, 0] = model.dynamics_weights[resp_ind, stim_ind]
        conn_weights[1:-1, 0] = connectome[top_resp_inds, stim_ind]
        conn_weights[1:-1, 1] = connectome[resp_ind, top_resp_inds]

        # get all other connections
        all_other_conn = np.arange(model.dynamics_dim)
        all_other_conn = all_other_conn[model_weights_2_steps_all.nonzero()[0]]
        all_other_conn = np.setdiff1d(all_other_conn, top_resp_inds)
        all_other_conn = np.setdiff1d(all_other_conn, (resp_ind, stim_ind))
        model_weight_network[-1, 0] = np.sum(np.abs(model.dynamics_weights[all_other_conn, stim_ind]))
        model_weight_network[-1, 1] = np.sum(np.abs(model.dynamics_weights[resp_ind, all_other_conn]))

        conn_weights[-1, 0] = np.sum(connectome[all_other_conn, stim_ind])
        conn_weights[-1, 1] = np.sum(connectome[resp_ind, all_other_conn])

        name = ['model', 'connectome']

        # set the predetermined position of each of the nodes in the graph
        node_names = top_cell_ids[-1][:-1] + ['other'] + [resp]
        pos_dict = []
        pos_dict.append((-2 * i_mult, 0))
        pos_dict.append((0, 2 * i_mult))
        pos_dict.append((0, 1 * i_mult))
        pos_dict.append((0, -1 * i_mult))
        pos_dict.append((0, -2 * i_mult))
        pos_dict.append((0, -3 * i_mult))
        pos_dict.append((2 * i_mult, 0))
        # angles = [-np.arctan(2/2), -np.arctan(1/2), np.arctan(1/2), np.arctan(2/2)]
        angles = [np.arctan(2/2), np.arctan(1/2), -np.arctan(1/2), -np.arctan(2/2), -np.pi/2]

        circle_size = 0.35 * i_mult
        line_width = 2 * i_mult
        head_width_mult = 1 / 120

        for ni, n in enumerate([model_weight_network, conn_weights]):
            fig, ax = plt.subplots()

            x_circ = np.cos(np.linspace(0, 2 * np.pi, 100))
            y_circ = np.sin(np.linspace(0, 2 * np.pi, 100))

            # all the outgoing connections from the stimulated neuron
            for si in range(1, n.shape[0]):
                thickness = n[si, 0] * width_mult[pair_ind][ni]
                x = pos_dict[0][0] + circle_size * np.cos(angles[si-1])
                y = pos_dict[0][1] + circle_size * np.sin(angles[si-1])
                dy = pos_dict[si][1] - y
                dx = pos_dict[si][0] - x - (circle_size + thickness * 0.03)

                plt.arrow(x, y, dx, dy, head_width=thickness * head_width_mult, linewidth=thickness, color='k', length_includes_head=True)

            # plot the direct arrow
            thickness = n[0, 0] * width_mult[pair_ind][ni]
            x = pos_dict[0][0] + circle_size
            y = pos_dict[0][1]
            dx = pos_dict[-1][0] - x - (circle_size + thickness * 0.03)
            dy = pos_dict[-1][1] - y
            plt.arrow(x, y, dx, dy, head_width=thickness * head_width_mult, linewidth=thickness, color='k', length_includes_head=True)

            # all the incoming connections for the responding neuron
            for si in range(1, n.shape[0]):
                thickness = n[si, 1] * width_mult[pair_ind][ni]
                x = pos_dict[si][0] + circle_size
                y = pos_dict[si][1]
                dx = pos_dict[-1][0] - x - (circle_size + thickness * 0.03) * np.cos(-angles[si-1])
                dy = pos_dict[-1][1] - y - (circle_size + thickness * 0.03) * np.sin(-angles[si-1])
                plt.arrow(x, y, dx, dy, head_width=thickness * head_width_mult, linewidth=thickness, color='k', length_includes_head=True)
            plt.xlim((-2, 2))
            plt.ylim(-2, 2)

            for pdi, pd in enumerate(pos_dict):
                plt.text(pd[0] - len(node_names[pdi]) / 2 * fontsize * 0.009, pd[1] - 0.04, node_names[pdi], fontsize=fontsize)
                circ = plt.Circle((pd[0], pd[1]), circle_size, color=[0.7, 0.7, 0.7])
                ax.add_patch(circ)
                plt.plot(circle_size * x_circ + pd[0], circle_size * y_circ + pd[1], linewidth=line_width, color='k')

            ax.set_aspect('equal', 'box')
            ax.axis('off')

            plt.savefig(fig_save_path / ('network_graph_' + stim + '_to_' + resp + '_' + name[ni] + '.pdf'))

        plt.show()

    plot_x = np.arange(-window[0] * sample_rate, window[1] * sample_rate)
    diplay_x = np.array([-15, 0, 15, 30])
    for ii, i in enumerate(broken_down_irfs):
        plt.figure()
        plt.plot(plot_x, i[:, :-1], alpha=0.5, color=plot_color['synap'])
        plt.plot(plot_x, i[:, -1], alpha=1, color=plot_color['synap'])
        plt.plot(plot_x, data[ii], color=plot_color['data'])
        plt.xticks(diplay_x * sample_rate, diplay_x)
        plt.axvline(0, color='k', linestyle='--')
        for label_ind in range(i.shape[1]):
            plt.text(plot_x[-1], i[-1, label_ind], '+' + top_cell_ids[ii][label_ind])
        plt.xlabel('time (s)')
        plt.ylabel('neural activity')
        plt.title(chosen_pairs[ii][1] + ' -> ' + chosen_pairs[ii][0])

        plt.savefig(fig_save_path / ('breakdown_' + chosen_pairs[ii][1] + '_to_' + chosen_pairs[ii][0] + '.pdf'))

    plt.show()

    return


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
        stim_string = cell_ids[plot_ind, 1] + '_' + cell_ids[plot_ind, 0]
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

    model_synap_dirms_chem = weights['models']['synap']['weights'][chem_no_gap]
    model_uncon_dirms_chem = weights['models']['unconstrained']['weights'][chem_no_gap]
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
    # watlas = wa.NeuroAtlas()
    # chem_sign_out = watlas.get_chemical_synapse_sign()

    # cmplx = np.logical_and(np.any(chem_sign_out == -1, axis=0),
    #                        np.any(chem_sign_out == 1, axis=0))
    # chem_sign = np.nansum(chem_sign_out, axis=0)
    # chem_sign[cmplx] = 0
    # chem_mask = chem_sign == 0

    # chem_sign[chem_sign > 0] = 1
    # chem_sign[chem_sign < 0] = 0
    # chem_sign[chem_mask] = np.nan

    # atlas_ids = list(watlas.neuron_ids)
    # atlas_ids[atlas_ids.index('AWCON')] = 'AWCL'
    # atlas_ids[atlas_ids.index('AWCOFF')] = 'AWCR'
    # cell_inds = np.array([atlas_ids.index(i) for i in cell_ids['all']])
    # chem_sign = chem_sign[np.ix_(cell_inds, cell_inds)]
    # chem_sign[masks['irm_nans']] = np.nan
    # chem_sign = chem_sign[chem_no_gap]

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
    plt.hist(booted_diff, bins=50, density=True)
    plt.title('model prediction vs measured IRMs, p = ' + str(p)[:5])
    plt.xlabel('paired similarity difference')

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

#TODO THIS IS A DUPLICATE figure out if has anything useful
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


def plot_missing_neuron(models, data, posterior_dict, post_save_path=None, sample_rate=2, fig_save_path=None):
    cell_ids = data['cell_ids']
    posterior_missing = posterior_dict['posterior_missing']
    emissions = data['emissions']
    inputs = data['inputs']
    rng = np.random.default_rng(0)
    connectivity = models['synap'].param_props['mask']['dynamics_weights'][:, :models['synap'].dynamics_dim]

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
    p = au.single_sample_boostrap_p(missing_corr - missing_corr_null, n_boot=1000000)

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
    # best_offset = -5  # AVER
    best_offset = 0  # AVER
    median_offset = -4  # URYVL
    best_neuron = np.unravel_index(sorted_corr_inds[-1 + best_offset], missing_corr.shape)
    median_neuron = np.unravel_index(sorted_corr_inds[int(sorted_corr_inds.shape[0] / 2) + median_offset], missing_corr.shape)

    best_data_ind = best_neuron[0]
    best_neuron_ind = best_neuron[1]
    median_data_ind = median_neuron[0]
    median_neuron_ind = median_neuron[1]

    plot_x = np.arange(emissions[best_data_ind].shape[0]) / sample_rate
    display_x = np.arange(0, emissions[best_data_ind].shape[0], 5*60*sample_rate) / sample_rate
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(plot_x, emissions[best_data_ind][:, best_neuron_ind], label='data', color=plot_color['data'])
    plt.plot(plot_x, posterior_missing[best_data_ind][:, best_neuron_ind], label='posterior', color=plot_color['synap'])

    for i in range(inputs[best_data_ind].shape[0]):
        this_inputs = inputs[best_data_ind][i, :]

        if not np.all(this_inputs == 0):
            plt.axvline(plot_x[i], color='k', linestyle='--')
            plt.text(plot_x[i], 1, cell_ids[np.where(this_inputs)[0][0]], rotation=90)

    plt.xlabel('time (s)')
    plt.xticks(display_x)
    plt.ylabel('neural activity (' + cell_ids[best_neuron_ind] + ')')

    plot_x = np.arange(emissions[median_data_ind].shape[0]) / sample_rate
    display_x = np.arange(0, emissions[median_data_ind].shape[0], 5*60*sample_rate) / sample_rate
    plt.subplot(2, 1, 2)
    plt.plot(plot_x, emissions[median_data_ind][:, median_neuron_ind], label='data', color=plot_color['data'])
    plt.plot(plot_x, posterior_missing[median_data_ind][:, median_neuron_ind], label='posterior', color=plot_color['synap'])
    plt.ylim(plt.ylim()[0], 1.2)
    plt.xlabel('time (s)')
    plt.xticks(display_x)
    plt.ylabel('neural activity (' + cell_ids[median_neuron_ind] + ')')

    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'recon_examples.pdf')

    plt.show()

    # loop through the best R2 neuron and eliminate each other neuron individually
    # and see how that changes the prediction
    # get the data and remove the neuron and its sister pair
    best_neuron = cell_ids[best_neuron_ind]
    best_data = emissions[best_data_ind].copy()
    best_inputs = inputs[best_data_ind]
    best_offset = posterior_dict['emissions_offset'][best_data_ind]
    best_init_mean = posterior_dict['init_mean'][best_data_ind]
    best_init_cov = posterior_dict['init_cov'][best_data_ind]
    num_neurons = best_data.shape[1]
    best_data[:, best_neuron_ind] = np.nan
    true_activity = emissions[best_data_ind][:, best_neuron_ind]

    # if 'recon_score_filter' in posterior_dict['neuron_recon']:
    #     recon_score_filter = posterior_dict['neuron_recon_ava']['recon_score_filter']
    #     recon_score_smoother = posterior_dict['neuron_recon_ava']['recon_score_smoother']
    #     recon_filter = posterior_dict['neuron_recon_ava']['recon_filter']
    #     recon_smoother = posterior_dict['neuron_recon_ava']['recon_smoother']
    # else:
    #     recon_score_filter = np.zeros(num_neurons)
    #     recon_score_smoother = np.zeros(num_neurons)
    #     recon_filter = np.zeros_like(best_data)
    #     recon_smoother = np.zeros_like(best_data)
    #
    # best_score = met.nan_corr(true_activity, posterior_missing[best_data_ind][:, best_neuron_ind])[0]
    #
    # # silence the sister pair so that reconstruction isn't trivial
    # sister_pair = au.get_sister_cell(best_neuron, cell_ids)
    #
    # if sister_pair is not None:
    #     best_data[:, cell_ids.index(sister_pair)] = np.nan
    #
    # start = time.time()
    # # loop through each neuron in the silenced data and silence that neuron as well.
    # # Then see how well you can reconstruct
    # silence_ava = True
    # # for silence_ava in [False, True]:
    # for i in range(num_neurons):
    #     if (recon_score_filter[i] != 0) and (recon_score_smoother[i] != 0):
    #         continue
    #
    #     neuron_name = cell_ids[i]
    #     sister_pair = au.get_sister_cell(cell_ids[i], cell_ids)
    #     neurons_to_silence = [neuron_name]
    #     if sister_pair is not None:
    #         neurons_to_silence.append(sister_pair)
    #
    #     if silence_ava:
    #         neurons_to_silence.append('AVAL')
    #         neurons_to_silence.append('AVAR')
    #
    #     model_silenced = ssmu.get_silenced_model(models['synap'], neurons_to_silence)
    #
    #     _, recon_smoother_out, _, recon_filter_out = model_silenced.lgssm_smoother(best_data, best_inputs, best_offset, best_init_mean, best_init_cov)
    #     recon_score_filter[i] = met.nan_corr(true_activity, recon_filter_out[:, best_neuron_ind])[0]
    #     recon_score_smoother[i] = met.nan_corr(true_activity, recon_smoother_out[:, best_neuron_ind])[0]
    #     recon_filter[:, i] = recon_filter_out[:, best_neuron_ind]
    #     recon_smoother[:, i] = recon_smoother_out[:, best_neuron_ind]
    #
    #     if sister_pair is not None:
    #         recon_score_filter[cell_ids.index(sister_pair)] = recon_score_filter[i]
    #         recon_score_smoother[cell_ids.index(sister_pair)] = recon_score_smoother[i]
    #         recon_filter[:, cell_ids.index(sister_pair)] = recon_filter[:, i]
    #         recon_smoother[:, cell_ids.index(sister_pair)] = recon_smoother[:, i]
    #
    #     if post_save_path is not None:
    #         if silence_ava:
    #             posterior_dict['neuron_recon_ava'] = {'data_ind': best_data_ind,
    #                                                   'chosen_neuron': best_neuron,
    #                                                   'recon_score_filter': recon_score_filter,
    #                                                   'recon_score_smoother': recon_score_smoother,
    #                                                   'recon_filter': recon_filter,
    #                                                   'recon_smoother': recon_smoother,
    #                                                   }
    #         else:
    #             posterior_dict['neuron_recon'] = {'data_ind': best_data_ind,
    #                                               'chosen_neuron': best_neuron,
    #                                               'recon_score_filter': recon_score_filter,
    #                                               'recon_score_smoother': recon_score_smoother,
    #                                               'recon_filter': recon_filter,
    #                                               'recon_smoother': recon_smoother,
    #                                               }
    #         post_file = open(post_save_path, 'wb')
    #         pickle.dump(posterior_dict, post_file)
    #         post_file.close()
    #
    #     end = time.time() - start
    #     print('completed ' + str(i + 1) + '/' + str(num_neurons))
    #     print('expected ' + str(end / (i + 1) * (num_neurons - i - 1)) + ' s remaining')
    #
    # # switch score to R2
    # for i in range(num_neurons):
    #     if np.all(recon_smoother[:, i] == 0):
    #         recon_score_filter[i] = 1
    #         recon_score_smoother[i] = 1
    #     else:
    #         recon_score_filter[i] = met.nan_r2(true_activity, recon_filter[:, i])
    #         recon_score_smoother[i] = met.nan_r2(true_activity, recon_smoother[:, i])
    #
    # sort_inds = np.argsort(recon_score_smoother)
    # recon_score_filter = recon_score_filter[sort_inds]
    # recon_score_smoother = recon_score_smoother[sort_inds]
    # recon_filter = recon_filter[:, sort_inds]
    # recon_smoother = recon_smoother[:, sort_inds]
    # removed_cell_ids = [cell_ids[i] for i in sort_inds]
    #
    # plt.figure()
    # plt.plot(recon_score_filter, label='filter')
    # plt.plot(recon_score_smoother, label='smoother')
    # plt.legend()
    # plt.show()
    #
    # for i in range(20):
    #     if i > 0:
    #         if removed_cell_ids[i][:-1] == removed_cell_ids[i-1][:-1]:
    #             continue
    #
    #     removed_neuron = removed_cell_ids[i]
    #
    #     if (removed_neuron[-1] == 'L') or (removed_neuron[-1] == 'R'):
    #         removed_neuron = removed_neuron[:-1]
    #
    #         orig_ind = sort_inds[i]
    #         sister_pair = au.get_sister_cell(removed_cell_ids[i], cell_ids)
    #
    #         if connectivity[orig_ind, best_neuron_ind] or connectivity[best_neuron_ind, orig_ind]:
    #             removed_neuron = removed_neuron + '*'
    #
    #         if sister_pair is not None:
    #             sister_ind = cell_ids.index(sister_pair)
    #             if connectivity[sister_ind, best_neuron_ind] or connectivity[best_neuron_ind, sister_ind]:
    #                 if removed_neuron[-1] != '*':
    #                     removed_neuron = removed_neuron + '*'
    #
    #     plt.figure()
    #     plt.subplot(2, 1, 1)
    #     plt.plot(posterior_missing[best_data_ind][:, best_neuron_ind], label=removed_neuron + ' held in')
    #     plt.plot(recon_smoother[:, i], label=removed_neuron + ' held out')
    #     plt.title('smoother prediction of ' + best_neuron + ', correlation=' + str(recon_score_smoother[i])[:4])
    #     plt.legend()
    #
    #     plt.subplot(2, 1, 2)
    #     plt.plot(recon_smoother[:, i], label='Kalman smoother')
    #     plt.plot(recon_filter[:, i], label='Kalman filter')
    #     plt.legend()
    #
    #     plt.show()

    return


def plot_sampled_model(data, posterior_dict, cell_ids, sample_rate=2, num_neurons=10,
                       window_size=1000, fig_save_path=None):
    emissions = data['emissions']
    inputs = data['inputs']
    posterior = posterior_dict['posterior']
    model_sampled = posterior_dict['model_sampled']
    model_sampled_noise = posterior_dict['model_sampled_noise']

    cell_ids_chosen = sorted(cell_ids['chosen'])
    neuron_inds_chosen = np.array([cell_ids['all'].index(i) for i in cell_ids_chosen])

    # get all the inputs but with only the chosen neurons
    inputs_truncated = [i[:, neuron_inds_chosen] for i in inputs]
    emissions_truncated = [e[:, neuron_inds_chosen] for e in emissions]
    # data_ind_chosen, time_window = au.get_example_data_set(inputs_truncated, emissions=emissions_truncated, window_size=window_size)
    # data_ind_chosen, time_window = au.get_example_data_set(inputs, emissions=emissions_truncated, window_size=window_size)
    specific_neuron_ind = cell_ids_chosen.index('RMDDL')
    data_ind_chosen, time_window = au.get_example_data_set_simple(inputs_truncated, emissions_truncated, specific_neuron_ind, cell_ids, sample_rate)

    emissions_chosen = emissions[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    inputs_chosen = inputs[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    all_inputs = inputs[data_ind_chosen][time_window[0]:time_window[1], :]
    posterior_chosen = posterior[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    model_sampled_chosen = model_sampled[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]
    model_sampled_noise_chosen = model_sampled_noise[data_ind_chosen][time_window[0]:time_window[1], neuron_inds_chosen]

    stim_events = np.where(np.sum(all_inputs, axis=1) > 0)[0]
    stim_ids = [cell_ids['all'][np.where(all_inputs[i, :])[0][0]] for i in stim_events]

    filt_shape = np.ones(5)
    for i in range(inputs_chosen.shape[1]):
        inputs_chosen[:, i] = np.convolve(inputs_chosen[:, i], filt_shape, mode='same')

    plot_y = np.arange(len(cell_ids_chosen))
    plot_x = np.arange(0, emissions_chosen.shape[0], 60 * sample_rate)

    plt.figure()
    cmax = np.nanpercentile(np.abs((model_sampled_noise_chosen, posterior_chosen)), plot_percent)

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
    plt.imshow(model_sampled_noise_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
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
    model_sampled_noise_chosen = model_sampled_noise_chosen + data_offset[None, :]

    plt.figure()
    for stim_time, stim_name in zip(stim_events, stim_ids):
        plt.axvline(stim_time, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.text(stim_time-5, 1.5, stim_name, rotation=90)
    plt.plot(emissions_chosen)
    plt.ylim([data_offset[-1] - 1, 1])
    plt.yticks(data_offset, cell_ids_chosen)
    plt.xticks(plot_x, (plot_x / sample_rate).astype(int))
    plt.xlabel('time (s)')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'emissions_time_traces.pdf')

    # plt.figure()
    # for stim_time, stim_name in zip(stim_events, stim_ids):
    #     plt.axvline(stim_time, color=[0.6, 0.6, 0.6], linestyle='--')
    #     plt.text(stim_time, 1.5, stim_name, rotation=90)
    # plt.plot(model_sampled_chosen, alpha=0.5)
    # plt.ylim([data_offset[-1] - 1, 1])
    # plt.yticks(data_offset, cell_ids_chosen)
    # plt.xticks(plot_x, (plot_x / sample_rate).astype(int))
    # plt.xlabel('time (s)')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path / 'model_sampled_time_traces.pdf')

    plt.figure()
    for stim_time, stim_name in zip(stim_events, stim_ids):
        plt.axvline(stim_time, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.text(stim_time, 1.5, stim_name, rotation=90)
    plt.plot(model_sampled_noise_chosen)
    plt.ylim([data_offset[-1] - 1, 1])
    plt.yticks(data_offset, cell_ids_chosen)
    plt.xticks(plot_x, (plot_x / sample_rate).astype(int))
    plt.xlabel('time (s)')
    plt.tight_layout()

    plt.show()

    return


def compare_eig_spec(models, rng=np.random.default_rng()):
    un = models['unconstrained'].dynamics_weights
    cc = models['synap'].dynamics_weights
    eye_rand = np.eye(un.shape[0])*0 + 0.3*rng.standard_normal(un.shape)

    un[np.eye(un.shape[0], dtype=bool)] = 0
    cc[np.eye(un.shape[0], dtype=bool)] = 0

    Uu, Su, Vu = np.linalg.svd(un)
    Uc, Sc, Vc = np.linalg.svd(cc)
    Ur, Sr, Vr = np.linalg.svd(eye_rand)

    Sue = np.sort(Su ** 2)[::-1]
    Suc = np.sort(Sc ** 2)[::-1]
    Sur = np.sort(Sr ** 2)[::-1]

    Sue = Sue / np.sum(Sue)
    Suc = Suc / np.sum(Suc)
    Sur = Sur / np.sum(Sur)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(Suc, label='connectome')
    plt.plot(Sue, label='uncon')
    plt.ylim([0, plt.ylim()[1]])
    plt.xlabel('Nth principal component')
    plt.ylabel('variance explained')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(Suc), label='connectome')
    plt.plot(np.cumsum(Sue), label='uncon')
    plt.xlabel('Nth principal component')
    plt.ylabel('cumulative variance explained')
    plt.ylim([0, 1])
    # plt.legend()

    plt.tight_layout()

    plt.figure()
    plt.plot(np.cumsum(Suc), label='connectome')
    plt.plot(np.cumsum(Sur), label='random')
    plt.xlabel('Nth principal component')
    plt.ylabel('cumulative variance explained')
    plt.legend()
    plt.ylim([0, 1])

    plt.show()

