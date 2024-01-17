from pathlib import Path
import analysis_utilities as au
import numpy as np
import pickle
import loading_utilities as lu
import analysis.paper_figs as pf
import copy

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
cell_ids_chosen = run_params['cell_ids_chosen']

# get the models
models = {}
posterior_dicts = {}
for mf in model_folders:
    model_file = open(saved_run_folder / model_folders[mf] / 'models' / 'model_trained.pkl', 'rb')
    models[mf] = pickle.load(model_file)
    model_file.close()

    post_file = open(saved_run_folder / model_folders[mf] / 'posterior_test.pkl', 'rb')
    posterior_dicts[mf] = pickle.load(post_file)
    post_file.close()

# get the data (the same for all runs)
data_folder = list(model_folders.values())[0]
data_train_file = open(saved_run_folder / data_folder / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

if 'data_corr' in data_train:
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

if 'data_corr' in data_test:
    data_corr_test = data_test['data_corr']
    data_corr_test_ci = data_test['data_corr_ci']
else:
    data_corr_test, data_corr_test_ci = au.nan_corr_data(data_test['emissions'])

    data_test['data_corr'] = data_corr_test
    data_test['data_corr_ci'] = data_corr_test_ci

    data_test_file = open(saved_run_folder / data_folder / 'data_test.pkl', 'wb')
    pickle.dump(data_test, data_test_file)
    data_test_file.close()

cell_ids = {'all': data_test['cell_ids'], 'chosen': cell_ids_chosen}
models = [au.normalize_model(m)[0] for m in models]

# get data IRMs
data_irms_train, data_irfs_train, data_irfs_sem_train, _ = \
    au.simple_get_irms(data_train['emissions'], data_train['inputs'], required_num_stim=required_num_stim,
                       window=window, sub_pre_stim=sub_pre_stim)

data_irms_test, data_irfs_test, data_irfs_sem_test, num_stim = \
    au.simple_get_irms(data_test['emissions'], data_test['inputs'], required_num_stim=required_num_stim,
                       window=window, sub_pre_stim=sub_pre_stim)

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
    if 'irfs' not in posterior_dicts[m] or posterior_dicts[m]['irfs'].shape[0] != window[1] / models[m].sample_rate:
        posterior_dicts[m]['irfs'] = au.calculate_irfs(models[m], duration=window[1])

    if 'dirfs' not in posterior_dicts[m] or posterior_dicts[m]['dirfs'].shape[0] != window[1] / models[m].sample_rate:
        posterior_dicts[m]['dirfs'] = au.calculate_dirfs(models[m], duration=window[1])

    weights['models'][m] = {'irfs': posterior_dicts[m]['irfs'],
                            'irms': np.sum(posterior_dicts[m]['irfs'], axis=0),
                            'dirfs': posterior_dicts[m]['dirfs'],
                            'dirms': np.sum(posterior_dicts[m]['dirfs'], axis=0),
                            }

    abs_dirms = np.abs(weights['models'][m]['dirms'])
    dirms_binarized = abs_dirms > (np.nanstd(abs_dirms) / std_factor)
    weights['models'][m]['dirms_binarized'] = dirms_binarized.astype(float)

# save the posterior dicts so the irfs and dirfs are saved
for mf in model_folders:
    post_file = open(saved_run_folder / model_folders[mf] / 'posterior_test.pkl', 'wb')
    pickle.dump(posterior_dicts[mf], post_file)
    post_file.close()

# model correlation
for m in models:
    model_corr = models[m].dynamics_weights @ models[m].dynamics_weights.T + models[m].dynamics_cov
    for i in range(100):
        model_corr = models[m].dynamics_weights @ model_corr @ models[m].dynamics_weights.T + models[m].dynamics_cov
    model_corr = model_corr[:models[m].dynamics_dim, :models[m].dynamics_dim]

    weights['models'][m]['corr'] = model_corr

# set up the masks
# get masks based on number of stims
num_stim_sweep = np.arange(1, 15)
n_stim_mask = []
for ni, n in enumerate(num_stim_sweep):
    n_stim_mask.append((num_stim < n) | np.isnan(weights['data']['test']['corr']))

masks = {'diagonal': np.eye(data_irms_train.shape[0], dtype=bool),
         'synap': (weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn']) > 0,
         'chem': weights['anatomy']['chem_conn'] > 0,
         'nan': np.isnan(weights['data']['test']['irms']) | np.isnan(weights['data']['test']['corr']),
         'n_stim_mask': n_stim_mask,
         'n_stim_sweep': num_stim_sweep}

# set all the weights to nan with the nan mask
weights_masked = copy.deepcopy(weights)
for i in weights_masked:
    for j in weights_masked[i]:
        if isinstance(weights_masked[i][j], dict):
            for k in weights_masked[i][j]:
                if weights_masked[i][j][k].ndim == 2:
                    weights_masked[i][j][k][masks['nan']] = np.nan
                elif weights_masked[i][j][k].ndim == 3:
                    weights_masked[i][j][k][:, masks['nan']] = np.nan
                else:
                    raise Exception('Weights shape not recognized')

        else:
            if weights_masked[i][j].ndim == 2:
                weights_masked[i][j][masks['nan']] = np.nan
            elif weights_masked[i][j].ndim == 3:
                weights_masked[i][j][:, masks['nan']] = np.nan
            else:
                raise Exception('Weights shape not recognized')

pf.corr_irm_recon(weights, weights_masked, masks, fig_save_path=fig_save_path)
# pf.figure_1(weights, weights_masked, masks, cell_ids, fig_save_path=fig_save_path, window=window)
# pf.figure_2(weights_masked, masks, cell_ids, fig_save_path=fig_save_path, window=window)
# pf.figure_3(weights_masked, masks, cell_ids, fig_save_path=fig_save_path, window=window)

