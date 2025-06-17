import numpy as np
import loading_utilities as lu
from pathlib import Path
import pickle
import analysis_utilities as au
import lgssm_utilities as ssmu
import metrics as met
from matplotlib import pyplot as plt

# the goal of this function is to test whether we get the same performance when evaluating the model on different
# subsets of the data

plot_color = {'data': np.array([217, 95, 2]) / 255,
              'synap': np.array([27, 158, 119]) / 255,
              'unconstrained': np.array([117, 112, 179]) / 255,
              'synap_randA': np.array([231, 41, 138]) / 255,
              'synap_randC': np.array([102, 166, 30]) / 255,
              'anatomy': np.array([64, 64, 64]) / 255,
              }

likelihood_divisor = 1
run_params = lu.get_run_params(param_name='../analysis_params/hold_out_cuts.yml')
model_paths = run_params['models']
saved_run_folder = Path(run_params['saved_run_folder'])
window = run_params['window']
fig_save_path = Path(run_params['fig_save_path'])

# load the data for each split
data_test = []
data_train = []
for p in model_paths['synap']:
    data_file = open(saved_run_folder / p / 'data_test.pkl', 'rb')
    data_test.append(pickle.load(data_file))
    data_file.close()

    # train data
    data_file = open(saved_run_folder / p / 'data_train.pkl', 'rb')
    data_train.append(pickle.load(data_file))
    data_file.close()

sample_rate = data_test[0]['sample_rate']
num_neurons = data_test[0]['emissions'][0].shape[1]
data_irms_test = []
data_irms_train = []
data_corr_train = []
data_corr_train_ci = []
data_corr_test = []
data_corr_test_ci = []
train_test_corr_irms = []
train_test_corr_corr = []

for i in range(len(data_test)):
    data_irfs_train, data_irfs_sem_train, data_irfs_train_all = \
        ssmu.get_impulse_response_functions(data_train[i]['emissions'], data_train[i]['inputs'],
                                            sample_rate=sample_rate, window=window, sub_pre_stim=True)

    num_neurons = data_test[i]['emissions'][0].shape[1]
    nan_loc = np.all(np.isnan(data_irfs_train), axis=0) | np.eye(num_neurons, dtype=bool)
    data_irms_train.append(np.nansum(data_irfs_train[int(window[0] * sample_rate):], axis=0) / sample_rate)
    data_irms_train[-1][nan_loc] = np.nan

    data_irfs_test, data_irfs_sem_test, data_irfs_test_all = \
        ssmu.get_impulse_response_functions(data_test[i]['emissions'], data_test[i]['inputs'],
                                            sample_rate=sample_rate, window=window, sub_pre_stim=True)

    nan_loc = np.all(np.isnan(data_irfs_test), axis=0) | np.eye(num_neurons, dtype=bool)
    data_irms_test.append(np.nansum(data_irfs_test[int(window[0] * sample_rate):], axis=0) / sample_rate)
    data_irms_test[-1][nan_loc] = np.nan

    if 'data_corr_ci' not in data_train[i]:
        this_data_corr_train, this_data_corr_train_ci = au.nan_corr_data(data_train[i]['emissions'])

        data_train[i]['data_corr'] = this_data_corr_train
        data_train[i]['data_corr_ci'] = this_data_corr_train_ci

        data_train_file = open(saved_run_folder / model_paths['synap'][i] / 'data_train.pkl', 'wb')
        pickle.dump(data_train[i], data_train_file)
        data_train_file.close()

    if 'data_corr_ci' not in data_test[i]:
        this_data_corr_test, this_data_corr_test_ci = au.nan_corr_data(data_test[i]['emissions'])

        data_test[i]['data_corr'] = this_data_corr_test
        data_test[i]['data_corr_ci'] = this_data_corr_test_ci

        data_test_file = open(saved_run_folder / model_paths['synap'][i] / 'data_test.pkl', 'wb')
        pickle.dump(data_test[i], data_test_file)
        data_test_file.close()

    data_corr_train.append(data_train[i]['data_corr'])
    data_corr_train_ci.append(data_train[i]['data_corr_ci'])
    data_corr_train[-1][np.eye(data_corr_train[-1].shape[0], dtype=bool)] = np.nan

    data_corr_test.append(data_test[i]['data_corr'])
    data_corr_test_ci.append(data_test[i]['data_corr_ci'])
    data_corr_test[-1][np.eye(data_corr_test[-1].shape[0], dtype=bool)] = np.nan

    train_test_corr_irms.append(met.nan_corr(data_irms_train[-1], data_irms_test[-1])[0])
    train_test_corr_corr.append(met.nan_corr(data_corr_train[-1], data_corr_test[-1])[0])

models = {}
posterior_dicts = {}
model_irms = {}
model_corr = {}
model_score = {}
model_score_ci = {}
model_corr_score = {}
model_corr_score_ci = {}
model_ll = {}
model_eigs = {}

for model_name in model_paths:
    models[model_name] = []
    posterior_dicts[model_name] = []
    model_irms[model_name] = []
    model_corr[model_name] = []
    model_score[model_name] = []
    model_score_ci[model_name] = []
    model_corr_score[model_name] = []
    model_corr_score_ci[model_name] = []
    model_ll[model_name] = []
    model_eigs[model_name] = []

    for mfi, mf in enumerate(model_paths[model_name]):
        model_file = open(saved_run_folder / mf / 'models' / 'model_trained.pkl', 'rb')
        models_in = pickle.load(model_file)
        model_file.close()

        models[model_name].append(au.normalize_model(models_in)[0])

        post_file = open(saved_run_folder / mf / 'posterior_test.pkl', 'rb')
        posterior_in = pickle.load(post_file)
        post_file.close()

        posterior_dicts[model_name].append(posterior_in)

        window_size = (np.sum(np.array(window) * sample_rate)).astype(int)
        if 'irfs' not in posterior_dicts[model_name][-1] or posterior_dicts[model_name][-1]['irfs'].shape[0] != window_size:
            posterior_dicts[model_name][-1]['irfs'] = ssmu.calculate_irfs(models[model_name][-1], window=window, verbose=True)

            post_file = open(saved_run_folder / mf / 'posterior_test.pkl', 'wb')
            pickle.dump(posterior_dicts[model_name][-1], post_file)
            post_file.close()

        model_irms_this = posterior_dicts[model_name][-1]['irfs'][int(window[0]*sample_rate):].sum(0) / sample_rate
        model_irms[model_name].append(model_irms_this)

        model_corr_this = ssmu.predict_model_corr_coef(models[model_name][-1])
        model_corr[model_name].append(model_corr_this)

        model_score_this, model_score_this_ci = met.nan_corr(data_irms_test[mfi], model_irms[model_name][-1])
        model_score[model_name].append(model_score_this)
        model_score_ci[model_name].append(model_score_this_ci)

        model_corr_score_this, model_corr_score_this_ci = met.nan_corr(data_corr_test[mfi], model_corr[model_name][-1])
        model_corr_score[model_name].append(model_corr_score_this)
        model_corr_score_ci[model_name].append(model_corr_score_this_ci)

        model_ll[model_name].append(posterior_dicts[model_name][-1]['ll'] / likelihood_divisor)
        eigs_this = np.linalg.eigvals(models[model_name][-1].dynamics_weights)
        model_eigs[model_name].append(eigs_this)

# plot the model score and the test log likelihood
model_list = ['synap', 'unconstrained', 'synap_randA']

all_scores = []
for ml in model_list:
    all_scores.append(model_score[ml])

all_scores = np.array(all_scores) / np.array(train_test_corr_irms)[None, :]

# plot the models correlation to the measured IRMs across multi-fold cross validation
y_lim = (0, 1.1)
plt.figure()
plt.plot(all_scores, color=(0.9, 0.9, 0.9), zorder=1)

for mi, m in enumerate(model_list):
    plot_x = np.ones(len(model_score[m])) * mi
    plt.scatter(plot_x, model_score[m] / np.array(train_test_corr_irms), color=plot_color[m], zorder=2)

plt.xlim((-0.5, 2.5))
plt.ylim(y_lim)
plt.ylabel('relative correlation')
plt.xticks(np.arange(len(model_list)), model_list, rotation=45)
plt.savefig(fig_save_path / 'fig_s3' / 'multifold_cv_irms.pdf')

# plot the models correlation to the measured correlations across multi-fold cross validation
all_corr_scores = []
for ml in model_list:
    all_corr_scores.append(model_corr_score[ml])

all_corr_scores = np.array(all_corr_scores) / np.array(train_test_corr_corr)[None, :]

plt.figure()
plt.plot(all_corr_scores, color=(0.9, 0.9, 0.9), zorder=1)

for mi, m in enumerate(model_list):
    plot_x = np.ones(len(model_corr_score[m])) * mi
    plt.scatter(plot_x, model_corr_score[m] / np.array(train_test_corr_corr), color=plot_color[m], zorder=2)

plt.xlim((-0.5, 2.5))
plt.ylim(y_lim)
plt.ylabel('relative correlation')
plt.xticks(np.arange(len(model_list)), model_list, rotation=45)
plt.tight_layout()
plt.savefig(fig_save_path / 'fig_s3' / 'multifold_cv_corr.pdf')


# comparing similarity between model weights
model_name = 'synap'
num_model = len(models[model_name])

# get the cell ids across all neurons
cell_ids_all = []
for mn in models:
    for m in models[mn]:
        cell_ids_all = list(np.unique(cell_ids_all + m.cell_ids))
num_neurons_all = len(cell_ids_all)

# get a mask for the anatomical connectiosn for all these neurons
anat = au.load_anatomical_data(cell_ids_all)
chosen_mask = (anat['chem_conn'] + anat['gap_conn']) > 0
chosen_mask[np.eye(chosen_mask.shape[0], dtype=bool)] = False

# expand the matrix of STAMs to a unified basis across all cross validation cuts
expanded_weights = {}
for mn in models:  # loop through the different model types
    expanded_weights[mn] = []

    for m in models[mn]:  # loop through the multifold cross validation
        expanded_weights[mn].append(np.zeros((num_neurons_all, num_neurons_all)))

        for c1i, c1 in enumerate(m.cell_ids):
            for c2i, c2 in enumerate(m.cell_ids):
                e1 = cell_ids_all.index(c1)
                e2 = cell_ids_all.index(c2)
                expanded_weights[mn][-1][e1, e2] = m.dynamics_weights[c1i, c2i]

# do the same for the data IRMs
expanded_irms_train = []

for di, d in enumerate(data_irms_train):
    expanded_irms_train.append(np.zeros((num_neurons_all, num_neurons_all)))

    for c1i, c1 in enumerate(data_train[di]['cell_ids']):
        for c2i, c2 in enumerate(data_train[di]['cell_ids']):
            e1 = cell_ids_all.index(c1)
            e2 = cell_ids_all.index(c2)
            expanded_irms_train[-1][e1, e2] = d[c1i, c2i]

corr_out = np.zeros((num_model, num_model))
corr_out_data = np.zeros((num_model, num_model))

for i in range(num_model):
    for j in range(num_model):
        a = expanded_weights[model_name][i][chosen_mask]
        b = expanded_weights[model_name][j][chosen_mask]
        corr_out[i, j] = met.nan_corr(a, b)[0]

        a = expanded_irms_train[i][chosen_mask]
        b = expanded_irms_train[j][chosen_mask]
        corr_out_data[i, j] = met.nan_corr(a, b)[0]

plt.figure()
plt.imshow(corr_out)
plt.clim(-1, 1)
plt.colorbar()
plt.xlabel('model repeats')
plt.ylabel('model repeats')
plt.title('correlation between model weights across repeats')

plt.figure()
plt.imshow(corr_out_data)
plt.clim(-1, 1)
plt.colorbar()
plt.xlabel('model repeats')
plt.ylabel('model repeats')
plt.title('correlation between data STAMs across repeats')

plt.figure()
upper_vals = corr_out[np.triu_indices(num_model, k=1)]
plt.hist(upper_vals)
plt.xlabel('correlation')
plt.ylabel('count')
plt.title('hist of correlation between model weights across repeats')
plt.show()

plt.figure()
upper_vals = corr_out[np.triu_indices(num_model, k=1)] / corr_out_data[np.triu_indices(num_model, k=1)]
plt.hist(upper_vals)
plt.xlabel('correlation')
plt.ylabel('count')
plt.title('hist of correlation between model weights across repeats')
plt.show()

a=1

