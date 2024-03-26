import numpy as np
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import lgssm_utilities as lgssmu
import metrics as met

window = [2, 2]

folder_path = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/syn_test/20240326_163641')
pruned_model_path = folder_path / 'pruning'

# load in the data
data_test_file = open(folder_path / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

# calculate data IRMS
data_irfs = lgssmu.get_impulse_response_functions(
    data_test['emissions'], data_test['inputs'], sample_rate=data_test['sample_rate'],
    window=window, sub_pre_stim=True)[0]
data_irms = np.sum(data_irfs, axis=0)
data_irms[np.eye(data_irms.shape[0], dtype=bool)] = np.nan

# load in the true model
model_true_file = open(folder_path / 'models' / 'model_true.pkl', 'rb')
model_true = pickle.load(model_true_file)
model_true_file.close()

dynamics_dim = model_true.dynamics_dim
true_mask = model_true.param_props['mask']['dynamics_weights'][:, :dynamics_dim]

# find all pruned models and load them in
model_pruned = []
model_score = []
model_mask = []

for m in sorted(folder_path.rglob('model_trained.pkl')):
    model_file = open(m, 'rb')
    a = pickle.load(model_file)
    model_file.close()

    model_file = open(m, 'rb')
    model_pruned.append(pickle.load(model_file))
    model_file.close()

    model_irms = lgssmu.calculate_irms(model_pruned[-1], window=window, verbose=False)
    model_score.append(met.nan_corr(data_irms, model_irms)[0])

    model_mask.append(model_pruned[-1].param_props['mask']['dynamics_weights'])

    # plt.figure()
    # plt.plot(model_pruned[-1].log_likelihood)

num_models = len(model_pruned)

# precision recall accuracy
prfa = np.zeros((num_models, 4))
sparsity = np.zeros(num_models)

for mmi, mm in enumerate(model_mask):
    mm = mm[:, :dynamics_dim]
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

    prfa_data[dti, 0] = np.mean(true_mask[data_guess])
    prfa_data[dti, 1] = np.mean(data_guess[true_mask])
    prfa_data[dti, 2] = 2 * prfa_data[dti, 0] * prfa_data[dti, 1] / (prfa_data[dti, 0] + prfa_data[dti, 1])
    prfa_data[dti, 3] = np.mean(true_mask == data_guess)

plt.figure()
plt.title('model')
plt.plot(prfa[:, 0], label='precision')
plt.plot(prfa[:, 1], label='recall')
plt.plot(prfa[:, 2], label='f measure')
plt.plot(prfa[:, 3], label='accuracy')
plt.plot(model_score, label='model score')
plt.legend()

plt.figure()
plt.title('data')
plt.plot(prfa_data[:, 0], label='precision')
plt.plot(prfa_data[:, 1], label='recall')
plt.plot(prfa_data[:, 2], label='f measure')
plt.plot(prfa_data[:, 3], label='accuracy')
plt.legend()

plt.figure()
plt.plot(sparsity)

plt.show()
a=1

