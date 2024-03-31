import numpy as np
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import lgssm_utilities as lgssmu
import metrics as met
import analysis_utilities as au

window = (15, 30)
folder_path = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL4_IL45_N80_R0_nf10/20240312_204358')
# pruned_model_path = folder_path / 'pruning_es010_pf015'
pruned_model_path = folder_path / 'pruning_es020_pf010'

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
# get rid of the diagonal
data_irms = data_irms[~np.eye(dynamics_dim, dtype=bool)].reshape((dynamics_dim, dynamics_dim - 1))
nan_loc = np.isnan(data_irms)

# load in the true mask
anatomy = au.load_anatomical_data(cell_ids=data_test['cell_ids'])
true_mask = (anatomy['gap_conn'] + anatomy['chem_conn']) > 0
# get rid of the diagonal
true_mask = true_mask[~np.eye(dynamics_dim, dtype=bool)].reshape((dynamics_dim, dynamics_dim - 1))

# find all pruned models and load them in
model_pruned = []
model_score = []
model_mask = []

for m in sorted(pruned_model_path.rglob('model_trained.pkl')):
    if not (m.parent.parent / 'posterior_test.pkl').exists():
        continue

    model_file = open(m, 'rb')
    a = pickle.load(model_file)
    model_file.close()

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

    model_irms = np.sum(model_irfs, axis=0) / model_pruned[-1].sample_rate

    # get rid of diagonal
    model_irms = model_irms[~np.eye(dynamics_dim, dtype=bool)].reshape((dynamics_dim, dynamics_dim - 1))

    model_score.append(met.nan_corr(data_irms, model_irms)[0])

    model_mask.append(model_pruned[-1].param_props['mask']['dynamics_weights'][:, :model_pruned[-1].dynamics_dim])
    # get rid of the diagonal
    model_mask[-1] = model_mask[-1][~np.eye(dynamics_dim, dtype=bool)].reshape((dynamics_dim, dynamics_dim-1))


num_models = len(model_pruned)

# precision recall accuracy
prfa = np.zeros((num_models, 4))
sparsity = np.zeros(num_models)

for mmi, mm in enumerate(model_mask):
    prfa[mmi, 0] = np.mean(true_mask[mm])
    prfa[mmi, 1] = np.mean(mm[true_mask])
    prfa[mmi, 2] = 2 * prfa[mmi, 0] * prfa[mmi, 1] / (prfa[mmi, 0] + prfa[mmi, 1])
    prfa[mmi, 3] = np.mean(true_mask == mm)
    sparsity[mmi] = np.mean(mm)

data_irf_threshold = 0.9**np.arange(21) * 100
data_guess = []
prfa_data = np.zeros((len(data_irf_threshold), 4))

for dti, dt in enumerate(data_irf_threshold):
    cutoff = np.nanpercentile(data_irms, dt)

    data_guess = data_irms <= cutoff

    prfa_data[dti, 0] = np.mean(true_mask[data_guess & ~nan_loc])
    prfa_data[dti, 1] = np.mean(data_guess[true_mask & ~nan_loc])
    prfa_data[dti, 2] = 2 * prfa_data[dti, 0] * prfa_data[dti, 1] / (prfa_data[dti, 0] + prfa_data[dti, 1])
    prfa_data[dti, 3] = np.mean(true_mask == data_guess)

plt.figure()
plt.title('model')
plt.plot(prfa[:, 0], label='precision')
plt.plot(prfa[:, 1], label='recall')
plt.plot(prfa[:, 2], label='f measure')
plt.plot(prfa[:, 3], label='accuracy')
plt.plot(model_score, label='model score')
plt.ylim((0, 1))
plt.legend()

plt.figure()
plt.title('data')
plt.plot(prfa_data[:, 0], label='precision')
plt.plot(prfa_data[:, 1], label='recall')
plt.plot(prfa_data[:, 2], label='f measure')
plt.plot(prfa_data[:, 3], label='accuracy')
plt.ylim((0, 1))
plt.legend()

plt.figure()
plt.plot(sparsity)

plt.show()
a=1

