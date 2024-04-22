import numpy as np
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import lgssm_utilities as lgssmu
import metrics as met
import analysis_utilities as au

window = (15, 30)
folder_path = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_nf10_Re+1/20240410_164824')
# pruned_model_path = folder_path / 'pruning_es010_pf015'
# pruned_model_path = folder_path / 'pruning_es020_pf010'
pruned_model_path = folder_path / 'pruning_es040_pf005'

# folder_path = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/syn_test/20240330_222646')
# pruned_model_path = folder_path / 'pruning_es020_pf005'

# load in the data
data_test_file = open(folder_path / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

# calculate data IRMS
data_irfs = lgssmu.get_impulse_response_functions(
    data_test['emissions'], data_test['inputs'], sample_rate=data_test['sample_rate'],
    window=window, sub_pre_stim=True)[0]
data_irms = np.sum(data_irfs, axis=0)
dynamics_dim = data_irms.shape[0]
data_irms[np.eye(data_irms.shape[0], dtype=bool)] = np.nan
nan_loc = np.isnan(data_irms)

# load in the true mask
anatomy = au.load_anatomical_data(cell_ids=data_test['cell_ids'])
true_mask = ((anatomy['gap_conn'] + anatomy['chem_conn']) > 0).astype(float)
# get rid of the diagonal
true_mask[np.eye(dynamics_dim, dtype=bool)] = np.nan

# find all pruned models and load them in
model_pruned = []
model_score = []
model_mask = []

for m in sorted(pruned_model_path.rglob('model_trained.pkl')):
    if not (m.parent.parent / 'posterior_test.pkl').exists():
        continue

    model_file = open(m, 'rb')
    model_pruned.append(pickle.load(model_file))
    model_file.close()

    post_file = open(m.parent.parent / 'posterior_test.pkl', 'rb')
    posterior_dict = pickle.load(post_file)
    post_file.close()

    if 'irfs' not in posterior_dict:
        model_irfs = lgssmu.calculate_irfs(model_pruned[-1], window=window, verbose=False)

        posterior_dict['irfs'] = model_irfs
        post_file = open(m.parent.parent / 'posterior_test.pkl', 'wb')
        pickle.dump(posterior_dict, post_file)
        post_file.close()
    else:
        model_irfs = posterior_dict['irfs']

    if 'eirfs' not in posterior_dict:
        model_eirfs = lgssmu.calculate_eirfs(model_pruned[-1], window=window, verbose=False)

        posterior_dict['eirfs'] = model_eirfs
        post_file = open(m.parent.parent / 'posterior_test.pkl', 'wb')
        pickle.dump(posterior_dict, post_file)
        post_file.close()
    else:
        model_eirfs = posterior_dict['eirfs']

    model_irms = np.sum(model_irfs[window[0]:, :, :], axis=0) / model_pruned[-1].sample_rate

    # get rid of diagonal
    model_irms[np.eye(dynamics_dim, dtype=bool)] = np.nan

    model_score.append(met.nan_corr(data_irms, model_irms)[0])

    model_mask.append(model_pruned[-1].param_props['mask']['dynamics_weights'][:, :model_pruned[-1].dynamics_dim].astype(float))
    # get rid of the diagonal
    model_mask[-1][np.eye(dynamics_dim, dtype=bool)] = np.nan

num_models = len(model_pruned)

# precision recall accuracy
prfa = np.zeros((num_models, 4))
sparsity = np.zeros(num_models)
rng = np.random.default_rng(0)

for mmi, mm in enumerate(model_mask):
    prfa[mmi, 0] = met.precision(true_mask, mm)
    prfa[mmi, 1] = met.recall(true_mask, mm)
    prfa[mmi, 2] = met.f_measure(true_mask, mm)
    prfa[mmi, 3] = met.accuracy(true_mask, mm)
    sparsity[mmi] = np.mean(mm)

# data_irf_threshold = 0.9**np.arange(21) * 100
data_irf_threshold = (0.95 - np.arange(len(model_mask)) * 0.05) * 100
data_guess = []
prfa_data = np.zeros((len(data_irf_threshold), 4))
nan_loc = np.isnan(data_irms)

for dti, dt in enumerate(data_irf_threshold):
    cutoff = np.nanpercentile(data_irms, dt)

    data_guess = (data_irms < cutoff).astype(float)
    data_guess[nan_loc] = np.nan

    prfa_data[dti, 0] = met.precision(true_mask, data_guess)
    prfa_data[dti, 1] = met.recall(true_mask, data_guess)
    prfa_data[dti, 2] = met.f_measure(true_mask, data_guess)
    prfa_data[dti, 3] = met.accuracy(true_mask, data_guess)

    if dti > 0:
        mask_diff = (model_mask[dti - 1] - model_mask[dti]) == 1
        print(np.mean(true_mask[mask_diff]))
        print(np.sum(true_mask[mask_diff]))
        print(np.mean(true_mask[mask_diff]) / np.nanmean(true_mask))
        print('')

plt.figure()
plt.title('model')
plt.plot(prfa[:, 0], label='precision')
plt.plot(prfa[:, 1], label='recall')
plt.plot(prfa[:, 2], label='f measure')
plt.plot(prfa[:, 3], label='accuracy')
plt.plot(model_score, label='model score')
plt.xticks(np.arange(prfa_data.shape[0]), (np.arange(prfa_data.shape[0]) + 1)*5)
plt.xlabel('sparsity')
plt.ylim((0, 1))
plt.legend()

plt.figure()
plt.title('data')
plt.plot(prfa_data[:, 0], label='precision')
plt.plot(prfa_data[:, 1], label='recall')
plt.plot(prfa_data[:, 2], label='f measure')
plt.plot(prfa_data[:, 3], label='accuracy')
plt.xticks(np.arange(prfa_data.shape[0]), (np.arange(prfa_data.shape[0]) + 1)*5)
plt.xlabel('sparsity')
plt.ylim((0, 1))
plt.legend()

plt.show()
a=1

