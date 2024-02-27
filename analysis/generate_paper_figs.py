from pathlib import Path
import analysis_utilities as au
import lgssm_utilities as ssmu
import tmac.preprocessing as tp
import scipy.signal as ss
import metrics as met
import numpy as np
import pickle
import loading_utilities as lu
import analysis.paper_figures as pf

run_params = lu.get_run_params(param_name='../analysis_params/paper_figures.yml')

# this analysis requires 4 models
 # synap: a model constrained to have weights only between neurons that have synapses in the connectome
# synap_randC: a model with randomized cell ids for every animal
# synap_randA: a model constrained to have weights between neurons with a randomized version of the connectome
# unconstrained: a model with unconstrained dynamics matrix

saved_run_folder = Path(run_params['saved_run_folder'])
model_folders = run_params['model_folders']
for k in model_folders:
    model_folders[k] = Path(model_folders[k])
fig_save_path = Path(run_params['fig_save_path'])
q_path = Path(run_params['q_path'])
q_alpha = run_params['q_alpha']
required_num_stim = run_params['required_num_stim']
sub_pre_stim = run_params['sub_pre_stim']
window = run_params['window']
num_stim_sweep_params = run_params['num_stim_sweep_params']
num_obs_sweep_params = run_params['num_obs_sweep_params']
rng = np.random.default_rng(run_params['random_seed'])
metric = getattr(met, run_params['metric'])
filter_tau = run_params['filter_tau']
num_chosen = run_params['num_cell_ids_chosen']

# get the models
models = {}
posterior_dicts = {}
for mf in model_folders:
    model_file = open(saved_run_folder / model_folders[mf] / 'models' / 'model_trained.pkl', 'rb')
    models_in = pickle.load(model_file)
    model_file.close()

    models[mf] = au.normalize_model(models_in)[0]

    post_file = open(saved_run_folder / model_folders[mf] / 'posterior_test.pkl', 'rb')
    posterior_dicts[mf] = pickle.load(post_file)
    post_file.close()

# get the data (the same for all runs)
data_folder = list(model_folders.values())[0]
data_train_file = open(saved_run_folder / data_folder / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

if 'data_corr_ci' in data_train:
    data_corr_train = data_train['data_corr']
    data_corr_train_ci = data_train['data_corr_ci']
else:
    data_corr_train, data_corr_train_ci = au.nan_corr_data(data_train['emissions'])

    data_train['data_corr'] = data_corr_train
    data_train['data_corr_ci'] = data_corr_train_ci

    data_train_file = open(saved_run_folder / data_folder / 'data_train.pkl', 'wb')
    pickle.dump(data_train, data_train_file)
    data_train_file.close()

data_test_file = open(saved_run_folder / data_folder / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

if 'data_corr_ci' in data_test:
    data_corr_test = data_test['data_corr']
    data_corr_test_ci = data_test['data_corr_ci']
else:
    data_corr_test, data_corr_test_ci = au.nan_corr_data(data_test['emissions'])

    data_test['data_corr'] = data_corr_test
    data_test['data_corr_ci'] = data_corr_test_ci

    data_test_file = open(saved_run_folder / data_folder / 'data_test.pkl', 'wb')
    pickle.dump(data_test, data_test_file)
    data_test_file.close()

cell_ids = {'all': data_test['cell_ids']}
sample_rate = models['synap'].sample_rate

# get data IRMs before interpolation
# count the number of stimulation events before interpolation
data_irfs_train, data_irfs_sem_train, data_irfs_train_all = \
    ssmu.get_impulse_response_functions(data_train['emissions'], data_train['inputs'],
                                        sample_rate=sample_rate, window=window, sub_pre_stim=sub_pre_stim)
nan_loc = np.all(np.isnan(data_irfs_train), axis=0)
data_irms_train = np.nansum(data_irfs_train[window[0]:], axis=0) * sample_rate
data_irms_train[nan_loc] = np.nan

data_irfs_test, data_irfs_sem_test, data_irfs_test_all = \
    ssmu.get_impulse_response_functions(data_test['emissions'], data_test['inputs'],
                                        sample_rate=sample_rate, window=window, sub_pre_stim=sub_pre_stim)
nan_loc = np.all(np.isnan(data_irfs_test), axis=0)
data_irms_test = np.nansum(data_irfs_test[window[0]:], axis=0) / sample_rate
data_irms_test[nan_loc] = np.nan

num_neurons = data_irfs_test.shape[1]
num_stim_train = np.zeros((num_neurons, num_neurons))
num_stim_test = np.zeros((num_neurons, num_neurons))
for ni in range(num_neurons):
    for nj in range(num_neurons):
        resp_to_stim_train = data_irfs_train_all[ni][:, window[0]:, nj]
        num_obs_when_stim_train = np.sum(np.any(~np.isnan(resp_to_stim_train), axis=1))
        num_stim_train[nj, ni] += num_obs_when_stim_train

        resp_to_stim_test = data_irfs_test_all[ni][:, window[0]:, nj]
        num_obs_when_stim_test = np.sum(np.any(~np.isnan(resp_to_stim_test), axis=1))
        num_stim_test[nj, ni] += num_obs_when_stim_test

# for each data set determihne whether a neuron was measured
obs_train = np.stack([np.mean(np.isnan(i), axis=0) < run_params['obs_threshold'] for i in data_train['emissions']])
obs_test = np.stack([np.mean(np.isnan(i), axis=0) < run_params['obs_threshold'] for i in data_test['emissions']])
# count the number of times two neurons were measured together
num_obs_train = np.zeros((obs_train.shape[1], obs_train.shape[1]))
num_obs_test = np.zeros((obs_test.shape[1], obs_test.shape[1]))
for i in range(obs_train.shape[1]):
    for j in range(obs_train.shape[1]):
        num_obs_train[i, j] = np.sum(obs_train[:, i] & obs_train[:, j])
        num_obs_test[i, j] = np.sum(obs_test[:, i] & obs_test[:, j])

# interpolate nans in data for viewing
for di, d in enumerate(data_train['emissions']):
    data_train['emissions'][di] = tp.interpolate_over_nans(d)[0]

for di, d in enumerate(data_test['emissions']):
    data_test['emissions'][di] = tp.interpolate_over_nans(d)[0]

# make a causal filter to smooth the data with
if filter_tau > 0:
    max_time = filter_tau * 3
    filter_x = np.arange(0, max_time, sample_rate)
    filt_shape = np.exp(-filter_x / filter_tau)
    filt_shape = filt_shape / np.sum(filt_shape)

    for di, d in enumerate(data_train['emissions']):
        data_train['emissions'][di] = ss.convolve2d(d, filt_shape[:, None], mode='full')[:-filt_shape.size+1, :]

    for di, d in enumerate(data_test['emissions']):
        data_test['emissions'][di] = ss.convolve2d(d, filt_shape[:, None], mode='full')[:-filt_shape.size+1, :]

    for pi, p in enumerate(posterior_dicts):
        for di, d in enumerate(posterior_dicts[p]['model_sampled_noise']):
            posterior_dicts[p]['model_sampled_noise'][di] = ss.convolve2d(d, filt_shape[:, None], mode='full')[:-filt_shape.size+1, :]

# get data IRFs after interpolation
data_irfs_train, data_irfs_sem_train = \
    ssmu.get_impulse_response_functions(data_train['emissions'], data_train['inputs'],
                                        sample_rate=sample_rate, window=window, sub_pre_stim=sub_pre_stim)[:2]

data_irfs_test, data_irfs_sem_test = \
    ssmu.get_impulse_response_functions(data_test['emissions'], data_test['inputs'],
                                        sample_rate=sample_rate, window=window, sub_pre_stim=sub_pre_stim)[:2]

# get the q values from francesco's paper
ids_path = Path('anatomical_data/cell_ids.pkl')
if not ids_path.exists():
    ids_path = Path('../') / ids_path
ids_file = open(ids_path, 'rb')
atlas_ids = pickle.load(ids_file)
ids_file.close()
atlas_inds = [atlas_ids.index(i) for i in cell_ids['all']]
q_in = np.load(str(q_path))[np.ix_(atlas_inds, atlas_inds)]

weights = {'data': {}}
weights['data']['train'] = {'irms': data_irms_train,
                            'irfs': data_irfs_train,
                            'irfs_sem': data_irfs_sem_train,
                            'corr': data_corr_train,
                            'corr_binarized': ((data_corr_train_ci[0] > 0) | (data_corr_train_ci[1] < 0)).astype(float),
                            'q': (q_in < q_alpha).astype(float),
                            }

weights['data']['test'] = {'irms': data_irms_test,
                           'irfs': data_irfs_test,
                           'irfs_sem': data_irfs_sem_test,
                           'corr': data_corr_test,
                           'corr_binarized': ((data_corr_test_ci[0] > 0) | (data_corr_test_ci[1] < 0)).astype(float),
                           'q': (q_in < q_alpha).astype(float),
                           }

# get anatomical data
weights['anatomy'] = au.load_anatomical_data(cell_ids=cell_ids['all'])

# get the model weights
weights['models'] = {}
# get the IRMs of the models and data
std_factor = 100
for m in models:
    save_post = False
    sample_rate = models[m].sample_rate
    window_size = (np.sum(np.array(window) * sample_rate)).astype(int)
    if 'irfs' not in posterior_dicts[m] or posterior_dicts[m]['irfs'].shape[0] != window_size:
        posterior_dicts[m]['irfs'] = ssmu.calculate_irfs(models[m], window=window)
        save_post = True

    if 'dirfs' not in posterior_dicts[m] or posterior_dicts[m]['dirfs'].shape[0] != window_size:
        posterior_dicts[m]['dirfs'] = ssmu.calculate_dirfs(models[m], window=window)
        save_post = True

    if 'rdirfs' not in posterior_dicts[m] or posterior_dicts[m]['rdirfs'].shape[0] != window_size:
        posterior_dicts[m]['rdirfs'] = ssmu.calculate_dirfs(models[m], add_recipricol=True, window=window)
        save_post = True

    if 'eirfs' not in posterior_dicts[m] or posterior_dicts[m]['eirfs'].shape[0] != window_size:
        posterior_dicts[m]['eirfs'] = ssmu.calculate_eirfs(models[m], window=window)
        save_post = True

    weights['models'][m] = {'irfs': posterior_dicts[m]['irfs'],
                            'irms': np.sum(posterior_dicts[m]['irfs'], axis=0) / sample_rate,
                            'dirfs': posterior_dicts[m]['dirfs'],
                            'rdirfs': posterior_dicts[m]['rdirfs'],
                            'dirms': np.sum(posterior_dicts[m]['dirfs'], axis=0) / sample_rate,
                            'eirfs': posterior_dicts[m]['eirfs'],
                            'eirms': np.sum(posterior_dicts[m]['eirfs'], axis=0) / sample_rate,
                            }

    abs_eirms = np.abs(weights['models'][m]['dirms'])
    eirms_binarized = abs_eirms > (np.nanstd(abs_eirms) / std_factor)
    weights['models'][m]['eirms_binarized'] = eirms_binarized.astype(float)

    if save_post:
        # save the posterior dicts so the irfs and dirfs are saved
        post_file = open(saved_run_folder / model_folders[m] / 'posterior_test.pkl', 'wb')
        pickle.dump(posterior_dicts[m], post_file)
        post_file.close()

# model correlation
for m in models:
    model_corr = ssmu.predict_model_corr_coef(models[m])
    weights['models'][m]['corr'] = model_corr

# set up the masks
data_nan = np.isnan(weights['data']['test']['irms']) | np.isnan(weights['data']['test']['corr'])
# get masks based on number of stims
n_stim_mask = []
n_stim_sweep = np.arange(num_stim_sweep_params[0], num_stim_sweep_params[1], num_stim_sweep_params[2])
diag_mask = np.eye(num_neurons, dtype=bool)
irm_nans = np.isnan(weights['data']['test']['irms']) | diag_mask
corr_nans = np.isnan(weights['data']['test']['corr']) | diag_mask

for ni, n in enumerate(n_stim_sweep):
    # loop through number of stimulations and include all pairs which were stimulated
    # within num_stim_sweep_params[2] of n
    stim_sweep_mask = num_stim_test != n
    for i in range(1, num_stim_sweep_params[2]):
        stim_sweep_mask &= num_stim_test != (n + i)

    n_stim_mask.append(stim_sweep_mask | irm_nans)

# get mask based on number of observations
n_obs_mask = []
n_obs_sweep = np.arange(num_obs_sweep_params[0], num_obs_sweep_params[1], num_obs_sweep_params[2])
for ni, n in enumerate(n_obs_sweep):
    # loop through number of observations and include all pairs which were observed
    # within num_obs_sweep_params[2] of n
    obs_sweep_mask_test = num_obs_test != n
    for i in range(1, num_obs_sweep_params[2]):
        obs_sweep_mask_test &= num_obs_test != (n + i)

    n_obs_mask.append(obs_sweep_mask_test | corr_nans)

# put all the masks in a dictionary
masks = {'diagonal': np.eye(data_irms_train.shape[0], dtype=bool),
         'synap': (weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn']) > 0,
         'chem': weights['anatomy']['chem_conn'] > 0,
         'gap': weights['anatomy']['gap_conn'] > 0,
         'irm_nans': irm_nans,
         'corr_nans': corr_nans,
         'irm_nans_num_stim': (num_stim_train < required_num_stim) | (num_stim_test < required_num_stim) | irm_nans,
         'corr_nans_num_stim': (num_obs_train < required_num_stim) | (num_obs_test < required_num_stim) | corr_nans,
         'n_stim_mask': n_stim_mask,
         'n_stim_sweep': n_stim_sweep,
         'n_obs_mask': n_obs_mask,
         'n_obs_sweep': n_obs_sweep}
masks['unconnected'] = ~masks['synap']

weights_masked = ssmu.mask_weights_to_nan(weights, masks['irm_nans'], masks['corr_nans'])

# get the highlighted cell IDs
stim_in_both = num_stim_train.diagonal() * num_stim_test.diagonal()
stim_in_both_inds = np.argsort(stim_in_both)[::-1]
cell_ids['sorted'] = [cell_ids['all'][i] for i in stim_in_both_inds]

### Exploration
# Figure 1
# pf.plot_irms(weights, cell_ids, use_chosen_ids=False, fig_save_path=fig_save_path)
# pf.plot_irms(weights, cell_ids, use_chosen_ids=True, fig_save_path=fig_save_path)

# am.plot_irm(model_weights=weights_masked['models']['synap']['dirms'],
#             measured_irm=weights_masked['data']['test']['irms'],
#             model_irm=weights_masked['models']['synap']['irms'],
#             data_corr=weights_masked['data']['train']['corr'],
#             cell_ids=cell_ids['all'], cell_ids_chosen=cell_ids['sorted'],
#             fig_save_path=fig_save_path)

# pf.plot_irfs(weights_masked, masks, cell_ids, window, num_plot=20, fig_save_path=fig_save_path)
# pf.plot_irfs_train_test(weights_masked, masks, cell_ids, window, num_plot=20, fig_save_path=fig_save_path)

# pf.weight_prediction(weights_masked, 'irms', fig_save_path=fig_save_path)
# pf.weight_prediction_sweep(weights_masked, masks, 'irms', fig_save_path=fig_save_path)
# pf.weight_prediction(weights_masked, 'corr', fig_save_path=fig_save_path)
# pf.weight_prediction_sweep(weights_masked, masks, 'corr', fig_save_path=fig_save_path)
#
# pf.weights_vs_connectome(weights, masks, metric=metric, fig_save_path=fig_save_path)

# Figure 2
# pf.plot_dirfs(weights_masked, masks, cell_ids, window, chosen_mask=masks['synap'], num_plot=20, fig_save_path=fig_save_path)
# pf.plot_dirfs(weights_masked, masks, cell_ids, window, chosen_mask=masks['unconnected'], num_plot=20, fig_save_path=fig_save_path)
# pf.plot_dirfs_train_test(weights_masked, masks, cell_ids, window, chosen_mask=masks['synap'], num_plot=10, fig_save_path=fig_save_path)
# pf.plot_dirfs_train_test(weights_masked, masks, cell_ids, window, chosen_mask=masks['unconnected'], num_plot=10, fig_save_path=fig_save_path)
# pf.plot_dirm_diff(weights_masked, masks, cell_ids, window, num_plot=10, fig_save_path=fig_save_path)
# pf.plot_dirfs_train_test_swap(weights_masked, masks, cell_ids, window, chosen_mask=masks['synap'], num_plot=10, fig_save_path=fig_save_path)
# pf.plot_dirfs_gt_irfs(weights_masked, masks, cell_ids, window, chosen_mask=masks['synap'], num_plot=10, fig_save_path=fig_save_path)
# pf.irm_vs_dirm(weights_masked, masks, cell_ids)

# Figure 3
# pf.predict_chem_synapse_sign(weights_masked, masks, cell_ids, metric=metric, rng=rng, fig_save_path=fig_save_path)
# pf.predict_gap_synapse_sign(weights_masked, masks, metric=metric, rng=rng, fig_save_path=fig_save_path)
# pf.unconstrained_vs_constrained_model(weights_masked, fig_save_path=fig_save_path)
# pf.unconstrained_model_vs_connectome(weights_masked, masks, fig_save_path=fig_save_path)

# Figure 4
# pf.corr_zimmer_paper(weights_masked, models, cell_ids)
# pf.plot_eigenvalues(models, cell_ids)

### Final verson of the figures
pairs = np.array([['AVDR', 'AVJR'],
                  ['SAADR', 'RMDDL'],

                  ['AVAL', 'AVEL'],
                  ['RMDDR', 'RMDDL'],

                  ['AWBR', 'AWAL'],
                  ['CEPVL', 'CEPDR'],
                  ])

# Figure 1
# pf.plot_irms(weights, cell_ids, num_neurons=None, fig_save_path=fig_save_path / 'fig_1')
# pf.plot_irms(weights, cell_ids, num_neurons=20, fig_save_path=fig_save_path / 'fig_1')

# pf.plot_specific_dirfs(weights_masked, masks, cell_ids, pairs, window, fig_save_path=fig_save_path / 'fig_1')

# pf.weights_vs_connectome(weights, masks, metric=metric, fig_save_path=fig_save_path / 'fig_1')

pf.weight_prediction(weights, masks, 'irms', fig_save_path=fig_save_path / 'fig_1')
# pf.weight_prediction_sweep(weights_masked, masks, 'irms', fig_save_path=fig_save_path / 'fig_1')
# pf.weight_prediction(weights_masked, masks, 'corr', fig_save_path=fig_save_path / 'fig_1')
# pf.weight_prediction_sweep(weights_masked, masks, 'corr', fig_save_path=fig_save_path / 'fig_1')

# pf.plot_sampled_model(data_test, posterior_dicts['synap'], sample_rate=sample_rate, cell_ids=cell_ids,
#                       num_neurons=10, fig_save_path=fig_save_path / 'fig_1')

# Figure 2

# Figure 3
# pf.plot_missing_neuron(data_test, posterior_dicts['synap'], sample_rate=sample_rate, fig_save_path=fig_save_path / 'fig_3')


