import numpy as np
import loading_utilities as lu
from pathlib import Path
import pickle
import analysis_utilities as au
import lgssm_utilities as ssmu
import metrics as met
from matplotlib import pyplot as plt

plot_color = {'data': np.array([217, 95, 2]) / 255,
              'synap': np.array([27, 158, 119]) / 255,
              'unconstrained': np.array([117, 112, 179]) / 255,
              'synap_randA': np.array([231, 41, 138]) / 255,
              # 'synap_randC': np.array([102, 166, 30]) / 255,
              'synap_randC': np.array([128, 128, 128]) / 255,
              'anatomy': np.array([64, 64, 64]) / 255,
              }

likelihood_divisor = 1
run_params = lu.get_run_params(param_name='../analysis_params/paper_figures.yml')
model_repeat_paths = run_params['model_repeats']
saved_run_folder = Path(run_params['saved_run_folder'])
window = run_params['window']
fig_save_path = Path(run_params['fig_save_path'])

# all the data should be the same, load it once
# test data
data_file = open(saved_run_folder / model_repeat_paths['synap'][0] / 'data_test.pkl', 'rb')
data_test = pickle.load(data_file)
data_file.close()

sample_rate = data_test['sample_rate']
num_neurons = data_test['emissions'][0].shape[1]

data_irfs_test, data_irfs_sem_test, data_irfs_test_all = \
    ssmu.get_impulse_response_functions(data_test['emissions'], data_test['inputs'],
                                        sample_rate=sample_rate, window=window, sub_pre_stim=True)
nan_loc = np.all(np.isnan(data_irfs_test), axis=0) | np.eye(num_neurons, dtype=bool)
data_irms_test = np.nansum(data_irfs_test[int(window[0]*sample_rate):], axis=0) / sample_rate
data_irms_test[nan_loc] = np.nan

# train data
data_file = open(saved_run_folder / model_repeat_paths['synap'][0] / 'data_train.pkl', 'rb')
data_train = pickle.load(data_file)
data_file.close()

if 'data_corr_ci' in data_test:
    data_corr_test = data_test['data_corr']
    data_corr_test_ci = data_test['data_corr_ci']
else:
    data_corr_test, data_corr_test_ci = au.nan_corr_data(data_test['emissions'])

    data_test['data_corr'] = data_corr_test
    data_test['data_corr_ci'] = data_corr_test_ci

    data_test_file = open(saved_run_folder / model_repeat_paths['synap'][0] / 'data_test.pkl', 'wb')
    pickle.dump(data_test, data_test_file)
    data_test_file.close()

if 'data_corr_ci' in data_train:
    data_corr_train = data_train['data_corr']
    data_corr_train_ci = data_train['data_corr_ci']
else:
    data_corr_train, data_corr_train_ci = au.nan_corr_data(data_train['emissions'])

    data_train['data_corr'] = data_corr_train
    data_train['data_corr_ci'] = data_corr_train_ci

    data_train_file = open(saved_run_folder / model_repeat_paths['synap'][0] / 'data_train.pkl', 'wb')
    pickle.dump(data_train, data_train_file)
    data_train_file.close()

data_corr_test[np.eye(data_corr_test.shape[0], dtype=bool)] = np.nan
data_corr_train[np.eye(data_corr_train.shape[0], dtype=bool)] = np.nan

data_irfs_train, data_irfs_sem_train, data_irfs_train_all = \
    ssmu.get_impulse_response_functions(data_train['emissions'], data_train['inputs'],
                                        sample_rate=sample_rate, window=window, sub_pre_stim=True)
nan_loc = np.all(np.isnan(data_irfs_train), axis=0) | np.eye(num_neurons, dtype=bool)
data_irms_train = np.nansum(data_irfs_train[int(window[0]*sample_rate):], axis=0) / sample_rate
data_irms_train[nan_loc] = np.nan

train_test_corr_irms = met.nan_corr(data_irms_train, data_irms_test)[0]
train_test_corr_corr = met.nan_corr(data_corr_train, data_corr_test)[0]

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

for model_name in model_repeat_paths:
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

    for model_folder in model_repeat_paths[model_name]:
        model_file = open(saved_run_folder / model_folder / 'models' / 'model_trained.pkl', 'rb')
        models_in = pickle.load(model_file)
        model_file.close()

        models[model_name].append(au.normalize_model(models_in)[0])

        post_file = open(saved_run_folder / model_folder / 'posterior_test.pkl', 'rb')
        posterior_in = pickle.load(post_file)
        post_file.close()

        posterior_dicts[model_name].append(posterior_in)

        window_size = (np.sum(np.array(window) * sample_rate)).astype(int)
        if 'irfs' not in posterior_dicts[model_name][-1] or posterior_dicts[model_name][-1]['irfs'].shape[0] != window_size:
            posterior_dicts[model_name][-1]['irfs'] = ssmu.calculate_irfs(models[model_name][-1], window=window, verbose=True)

            post_file = open(saved_run_folder / model_folder / 'posterior_test.pkl', 'wb')
            pickle.dump(posterior_dicts[model_name][-1], post_file)
            post_file.close()

        model_irms_this = posterior_dicts[model_name][-1]['irfs'][int(window[0]*sample_rate):].sum(0) / sample_rate
        model_irms[model_name].append(model_irms_this)

        model_corr_this = ssmu.predict_model_corr_coef(models[model_name][-1])
        model_corr[model_name].append(model_corr_this)

        model_score_this, model_score_this_ci = met.nan_corr(data_irms_test, model_irms[model_name][-1])
        model_score[model_name].append(model_score_this)
        model_score_ci[model_name].append(model_score_this_ci)

        model_corr_score_this, model_corr_score_this_ci = met.nan_corr(data_corr_test, model_corr[model_name][-1])
        model_corr_score[model_name].append(model_corr_score_this)
        model_corr_score_ci[model_name].append(model_corr_score_this_ci)

        # num_lags = models[model_name][-1].dynamics_lags
        # mask = models[model_name][-1].param_props['mask']['dynamics_weights']
        # mask[np.tile(np.eye(num_neurons, dtype=bool), (1, num_lags))] = 0
        # rand_set = np.random.randn(np.sum(mask))
        # # models[model_name][-1].dynamics_weights[:num_neurons, :][mask] = rand_set / 1000000
        # models[model_name][-1].dynamics_weights[:num_neurons, :][mask] = 0

        model_ll[model_name].append(posterior_dicts[model_name][-1]['ll'] / likelihood_divisor)
        eigs_this = np.linalg.eigvals(models[model_name][-1].dynamics_weights)
        model_eigs[model_name].append(eigs_this)

# plot the model score and the test log likelihood
model_list = ['synap', 'unconstrained', 'synap_randA']
plt.figure()
plt.subplot(1, 2, 1)
for mi, m in enumerate(model_list):
    plot_x = np.ones(len(model_score[m])) * mi
    plt.scatter(plot_x, model_score[m] / train_test_corr_irms, color=plot_color[m])
plt.xlim((-0.5, 2.5))
plt.ylim((0, 1))
plt.ylabel('relative correlation')
plt.xticks(np.arange(len(model_list)), model_list, rotation=45)

ax = plt.subplot(1, 2, 2)
for mi, m in enumerate(model_list):
    plot_x = np.ones(len(model_ll[m])) * mi
    plt.scatter(plot_x, model_ll[m], color=plot_color[m] )
plt.xlim((-0.5, 2.5))
plt.ylabel('test log-likelihood')
plt.xticks(np.arange(len(model_list)), model_list, rotation=45)
plt.tight_layout()
plt.savefig(fig_save_path / 'fig_s2' / 'stam_ll_rand_init.pdf')

plt.figure()
for mi, m in enumerate(model_list):
    plot_x = np.ones(len(model_corr_score[m])) * mi
    plt.scatter(plot_x, model_corr_score[m] / train_test_corr_corr, color=plot_color[m])
plt.xlim((-0.5, 2.5))
plt.ylim((0, 1))
plt.ylabel('relative correlation')
plt.xticks(np.arange(len(model_list)), model_list, rotation=45)
plt.tight_layout()
plt.savefig(fig_save_path / 'fig_s2' / 'corr_rand_init.pdf')

plt.show()
a=1
# plot log likelihood and score across different lags
# plt.figure()
# plt.subplot(1, 2, 1)
# plot_x = np.arange(len(model_score['synap_sweep']))
# plot_x = np.array([1, 5, 10, 15, 30, 45, 60])
# plt.scatter(plot_x, np.array(model_score['synap_sweep']) / train_test_corr)
# plt.ylim((0, 1))
# plt.ylabel('relative correlation')
# plt.xlabel('dynamics input lags')
# plt.title('unconstrained')
#
# plt.subplot(1, 2, 2)
# plot_x = np.arange(len(model_ll['synap_sweep']))
# plt.scatter(plot_x, model_ll['synap_sweep'])
# plt.ylabel('test log-likelihood')
# plt.xlabel('dynamics input lags')
#
# plt.tight_layout()

# for i in model_eigs['synap_sweep']:
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.plot(np.sort(np.abs(i))[::-1])
#     plt.subplot(1, 2, 2)
#     plt.scatter(np.real(i), np.imag(i))

plt.show()
a=1


