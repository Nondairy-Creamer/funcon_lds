from pathlib import Path
import analysis_utilities as au
import numpy as np
import pickle
import loading_utilities as lu
import analysis.paper_figs as pf

run_params = lu.get_run_params(param_name='../analysis_params/paper_figures.yml')

model_folders = run_params['model_folders']
for k in model_folders.keys():
    model_folders[k] = Path(model_folders[k])
fig_save_path = Path(run_params['fig_save_path'])
required_num_stim = run_params['required_num_stim']
sub_pre_stim = run_params['sub_pre_stim']
window = run_params['window']
cell_ids_chosen = run_params['cell_ids_chosen']

# get the models
models = {}
posterior_dicts = {}
for mf in model_folders.keys():
    model_file = open(model_folders[mf] / 'models' / 'model_trained.pkl', 'rb')
    models[mf] = pickle.load(model_file)
    model_file.close()

    post_file = open(model_folders[mf] / 'posterior_test.pkl', 'rb')
    posterior_dicts[mf] = pickle.load(post_file)
    post_file.close()

# get the data (the same for all runs)
data_folder = list(model_folders.values())[0]
data_train_file = open(data_folder / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

if 'data_corr' in data_train.keys():
    data_corr = data_train['data_corr']
else:
    data_corr = au.nan_corr_data(data_train['emissions'])

    data_train['data_corr'] = data_corr

    data_train_file = open(data_folder / 'data_train.pkl', 'wb')
    pickle.dump(data_train, data_train_file)
    data_train_file.close()

data_test_file = open(data_folder / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

# test data
emissions_test = data_test['emissions']
inputs_test = data_test['inputs']
cell_ids = {'all': data_test['cell_ids'], 'chosen': cell_ids_chosen}
data_irms, data_irfs, data_irfs_sem = \
    au.simple_get_irms(emissions_test, inputs_test, required_num_stim=required_num_stim,
                       window=window, sub_pre_stim=sub_pre_stim)

weights = {}
weights['data'] = {'irms': data_irms,
                   'irfs': data_irfs,
                   'irfs_sem': data_irfs_sem,
                   'corr': data_corr}

# get anatomical data
weights['anatomy'] = au.load_anatomical_data(cell_ids=cell_ids['all'])

# get the model weights
weights['models'] = {}
for m in models.keys():
    weights['models'][m] = {'weights': models[m].stack_dynamics_weights()}

# get the IRMs of the models and data
for m in models.keys():
    if 'irfs' not in posterior_dicts[m].keys() or posterior_dicts[m]['irfs'].shape[0] != window[1] / models[m].sample_rate:
        posterior_dicts[m]['irfs'] = au.calculate_irfs(models[m], duration=window[1])

    if 'dirfs' not in posterior_dicts[m].keys() or posterior_dicts[m]['dirfs'].shape[0] != window[1] / models[m].sample_rate:
        posterior_dicts[m]['dirfs'] = au.calculate_dirfs(models[m], duration=window[1])

    weights['models'][m]['irfs'] = posterior_dicts[m]['irfs']
    weights['models'][m]['irms'] = np.sum(weights['models'][m]['irfs'], axis=0)

    weights['models'][m]['dirfs'] = posterior_dicts[m]['dirfs']
    weights['models'][m]['dirms'] = np.sum(weights['models'][m]['dirfs'], axis=0)

# save the posterior dicts so the irfs and dirfs are saved
for mf in model_folders.keys():
    post_file = open(model_folders[mf] / 'posterior_test.pkl', 'wb')
    pickle.dump(posterior_dicts[mf], post_file)
    post_file.close()

# model correlation
for m in models.keys():
    model_corr = models[m].dynamics_weights @ models[m].dynamics_weights.T + models[m].dynamics_cov
    for i in range(100):
        model_corr = models[m].dynamics_weights @ model_corr @ models[m].dynamics_weights.T + models[m].dynamics_cov
    model_corr = model_corr[:models[m].dynamics_dim, :models[m].dynamics_dim]

    weights['models'][m]['corr'] = model_corr

# set up the masks
masks = {'diagonal': np.eye(data_irms.shape[0], dtype=bool),
         'synap': (weights['anatomy']['chem_conn'] + weights['anatomy']['gap_conn']) > 0,
         'nan': np.isnan(weights['data']['irms']) | np.isnan(weights['data']['corr'])}

pf.figure_1(weights, masks, cell_ids, fig_save_path=fig_save_path, window=window)
pf.figure_2(weights, masks, cell_ids, fig_save_path=fig_save_path, window=window)
pf.figure_3(weights, masks, cell_ids, fig_save_path=fig_save_path, window=window)

