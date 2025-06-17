from pathlib import Path
import analysis_utilities as au
import pickle
import loading_utilities as lu
import lgssm_utilities as ssmu
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as ss

rng = np.random.default_rng()
run_params = lu.get_run_params(param_name='../analysis_params/paper_figures.yml')

window = run_params['window']
sub_pre_stim = run_params['sub_pre_stim']

saved_run_folder = Path(run_params['saved_run_folder'])
model_folders = run_params['model_folders']
for k in model_folders:
    model_folders[k] = Path(model_folders[k])

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

data_folder = list(model_folders.values())[0]
data_train_file = open(saved_run_folder / data_folder / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

data_test_file = open(saved_run_folder / data_folder / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

num_neurons = data_train['emissions'][0].shape[1]
sample_rate = models['synap'].sample_rate

# we're going to store all the STAMs in a neuron x neuron matrix which accounts for each pair of neurons
# each cell in the matrix is going to be an array [[worm 1 stams], [worm 2 stams], ... ]
stams_all_grouped = [[[] for _ in range(num_neurons)] for _ in range(num_neurons)]
stams_all_grouped_resampled = [[[] for _ in range(num_neurons)] for _ in range(num_neurons)]
stams_all = [[[] for _ in range(num_neurons)] for _ in range(num_neurons)]

# loop through all the data
for e, i in zip(data_train['emissions'], data_train['inputs']):
    # for this data set, get the measured STAs
    data_irfs_train, data_irfs_sem_train, data_irfs_train_all = \
        ssmu.get_impulse_response_functions([e], [i], sample_rate=sample_rate, window=window, sub_pre_stim=sub_pre_stim)

    # the data_irfs_train_all data comes out as an array across all neurons. in it is the response every every other
    # neuron each time the neuron was stimulated
    # here we'll calculate the time average of all the responses
    for n1i, n1 in enumerate(data_irfs_train_all):
        if n1.shape[0] > 0:
            # find if all measurements of this STA are actually mostly nan
            nan_loc = np.mean(np.isnan(n1), axis=1) > 0.8
            this_worm_stams = np.nansum(n1, axis=1)
            this_worm_stams[nan_loc] = np.nan

            for n2i in range(n1.shape[2]):
                this_pair_stams = this_worm_stams[:, n2i]
                this_pair_stams_no_nan = this_pair_stams[~np.isnan(this_pair_stams)]
                if this_pair_stams_no_nan.shape[0] > 0:
                    stams_all_grouped[n2i][n1i].append(this_pair_stams_no_nan)
                    stams_all[n2i][n1i] += list(this_pair_stams_no_nan)
                    stams_all_grouped[n2i][n1i].append(this_pair_stams_no_nan)

within_worm_var = np.zeros((num_neurons, num_neurons))
within_worm_var_resampled = np.zeros((num_neurons, num_neurons))
all_var = np.zeros((num_neurons, num_neurons))
across_worm_var = np.zeros((num_neurons, num_neurons))
for n1 in range(num_neurons):
    for n2 in range(num_neurons):
        # get a resampled version of the stams
        for w in stams_all_grouped[n1][n2]:
            resampled_vals = rng.choice(stams_all[n1][n2], size=len(w), replace=False)
            stams_all_grouped_resampled[n1][n2].append(resampled_vals)

        num_worms = len(stams_all_grouped[n1][n2])
        # calculate the within worm variance and average across worms
        this_worm_var = np.zeros(num_worms)
        this_worm_var_resampled = np.zeros(num_worms)
        for w in range(num_worms):
            this_worm_var[w] = np.var(stams_all_grouped[n1][n2][w], ddof=1)
            this_worm_var_resampled[w] = np.var(stams_all_grouped_resampled[n1][n2][w], ddof=1)

        if ~np.all(np.isnan(this_worm_var)):
            within_worm_var[n1, n2] += np.nanmean(this_worm_var)
            within_worm_var_resampled[n1, n2] += np.nanmean(this_worm_var_resampled)

        # calculate the across worm variance
        # get all the stams in one array.
        all_var[n1, n2] = np.nanvar(stams_all[n1][n2], ddof=1)

within_worm_var[within_worm_var == 0] = np.nan
within_worm_var_resampled[within_worm_var_resampled == 0] = np.nan
# across_worm_var = all_var - within_worm_var
across_worm_var = all_var
within_worm_var[np.eye(num_neurons, dtype=bool)] = np.nan
within_worm_var_resampled[np.eye(num_neurons, dtype=bool)] = np.nan
across_worm_var[np.eye(num_neurons, dtype=bool)] = np.nan

within_worm_var_no_nan = within_worm_var[~np.isnan(within_worm_var)]
within_worm_var_resampled_no_nan = within_worm_var_resampled[~np.isnan(within_worm_var_resampled)]
across_worm_var_no_nan = across_worm_var[~np.isnan(across_worm_var)]

percent_cutoff = 10
num_bins = 100
save_path = '/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/nature_review_appeal/'
plt.figure()
plot_range = (0, np.percentile(across_worm_var_no_nan, 100 - percent_cutoff))
plt.hist(across_worm_var_no_nan, label='across', density=True, range=plot_range, bins=num_bins, alpha=0.5)
plt.hist(within_worm_var_no_nan, label='within', density=True, range=plot_range, bins=num_bins, alpha=0.5)
plt.legend()
plt.xlabel('variance between STAMs')
plt.ylabel('probability density')
plt.savefig(save_path + 'var_across_and_within.pdf')

var_diff = across_worm_var - within_worm_var
var_diff_no_nan = var_diff[~np.isnan(var_diff)]
p = ss.ttest_1samp(var_diff_no_nan, 0).pvalue
plt.figure()
plot_range = (np.percentile(var_diff_no_nan, percent_cutoff / 2), np.percentile(var_diff_no_nan, 100 - percent_cutoff / 2))
plt.hist(var_diff_no_nan, range=plot_range, density=True, bins=num_bins, alpha=0.5)
plt.title('p = ' + str(p))
plt.xlabel('across-within')
plt.ylabel('probability density')
plt.savefig(save_path + 'std_across_minus_within.pdf')

plt.figure()
plot_range = (0, np.percentile(within_worm_var_resampled_no_nan, 100 - percent_cutoff))
plt.hist(within_worm_var_resampled_no_nan, range=plot_range, label='resampled', density=True, bins=num_bins, alpha=0.5)
plt.hist(within_worm_var_no_nan, range=plot_range, label='within', density=True, bins=num_bins, alpha=0.5)
plt.legend()
plt.xlabel('variance between STAMs')
plt.ylabel('probability density')

var_diff = within_worm_var_resampled - within_worm_var
var_diff_no_nan = var_diff[~np.isnan(var_diff)]
p = ss.ttest_1samp(var_diff_no_nan, 0).pvalue
plt.figure()
plot_range = (np.percentile(var_diff_no_nan, percent_cutoff / 2), np.percentile(var_diff_no_nan, 100 - percent_cutoff / 2))
plt.hist(var_diff_no_nan, range=plot_range, density=True, bins=num_bins, alpha=0.5)
plt.title('p = ' + str(p))
plt.xlabel('resampled-within')
plt.ylabel('probability density')

plt.show()

debug=1

