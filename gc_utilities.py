import os.path
import pickle
import numpy as np
import loading_utilities as lu
from matplotlib import pyplot as plt
from ssm_classes import Lgssm
import inference_utilities as iu
import analysis_utilities as au


def gc_preprocessing(run_params, load_dir='/Users/lsmith/Documents/python/', rerun=False):
    if os.path.exists(load_dir + 'preprocessed_data.pkl') and os.path.isfile(load_dir + 'preprocessed_data.pkl') \
            and not rerun:
        with open(load_dir + 'preprocessed_data.pkl', 'rb') as f:
            A, num_neurons, num_data_sets, emissions, inputs, cell_ids = pickle.load(f)
    else:
        # load in the data for the model and do any preprocessing here
        data_train, data_test = \
            lu.load_and_preprocess_data(run_params['data_path'], num_data_sets=run_params['num_data_sets'],
                                        start_index=run_params['start_index'],
                                        force_preprocess=run_params['force_preprocess'],
                                        correct_photobleach=run_params['correct_photobleach'],
                                        interpolate_nans=run_params['interpolate_nans'],
                                        held_out_data=run_params['held_out_data'],
                                        neuron_freq=run_params['neuron_freq'])
        emissions, inputs, cell_ids = data_train['emissions'], data_train['inputs'], data_train['cell_ids']
        num_neurons = emissions[0].shape[1]
        num_data_sets = len(emissions)

        # get the input dimension after removing the neurons that were never stimulated
        input_dim = inputs[0].shape[1]

        # initialize the model and set model weights
        model_true = Lgssm(num_neurons, num_neurons, input_dim,
                           dynamics_lags=run_params['dynamics_lags'],
                           dynamics_input_lags=run_params['dynamics_input_lags'],
                           verbose=run_params['verbose'],
                           param_props=run_params['param_props'])

        model_true.emissions_weights = np.eye(model_true.emissions_dim, model_true.dynamics_dim_full)
        model_true.emissions_input_weights = np.zeros((model_true.emissions_dim, model_true.input_dim_full))

        # randomize the parameters (defaults are nonrandom)
        model_true.randomize_weights()

        A = model_true.dynamics_weights

        with open(load_dir + 'preprocessed_data.pkl', 'wb') as f:
            pickle.dump((A, num_neurons, num_data_sets, emissions, inputs, cell_ids), f)

    return A, num_neurons, num_data_sets, emissions, inputs, cell_ids


# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i) for tau=1,...,L is signif larger
# than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)
def run_gc(num_data_sets, emissions_num_lags, inputs_num_lags, num_neurons, inputs, emissions, f_name='gc_data',
           load_dir='/Users/lsmith/Documents/python/', rerun=False):
    f_name = f_name + str(emissions_num_lags) + str(inputs_num_lags) + '.pkl'

    if os.path.exists(load_dir + f_name) and os.path.isfile(load_dir + f_name) and not rerun:
        with open(load_dir + f_name, 'rb') as f:
            all_a_hat, all_a_hat_0, all_b_hat, mse = pickle.load(f)

    else:
        all_a_hat = np.empty((num_neurons, num_neurons * emissions_num_lags, num_data_sets))
        all_a_hat_0 = np.empty((num_neurons, num_neurons * emissions_num_lags, num_data_sets))
        all_b_hat = np.empty((num_neurons, num_neurons * inputs_num_lags, num_data_sets))
        mse = np.zeros(num_data_sets)

        for d in range(num_data_sets):
            # num_time, neurons varies depending on the dataset
            num_time, num_neurons = emissions[d].shape
            curr_inputs = inputs[d]

            # get rid of any inputs that never receive stimulation
            has_stims = np.any(curr_inputs, axis=0)

            nans = np.any(np.isnan(emissions[d]), axis=0)

            curr_inputs = curr_inputs[:, has_stims]
            weights_mask = np.eye(inputs[d].shape[1])
            # take out the neurons that were never stimulated from the columns
            weights_mask = weights_mask[:, has_stims]
            # take out the neurons that are nan for the emissions for the rows for fitting later
            weights_mask = weights_mask[~nans, :]

            # need to delete columns with NaN neurons, but make a list of these indices to add them in as 0s in the end
            nans_mask = nans[:, None] | nans[:, None].T

            # nan_list.append(nans)
            non_nan_emissions = emissions[d][:, ~nans]

            # the times of both time series must align, so chop off the extra time for emissions or inputs, whichever has
            # less lags
            # num_lags = emissions_num_lags
            # if emissions_num_lags < inputs_num_lags:
            #     num_lags = inputs_num_lags - emissions_num_lags
            #     non_nan_emissions = non_nan_emissions[:-num_lags, :]
            # elif inputs_num_lags < emissions_num_lags:
            #     num_lags = emissions_num_lags
            #     curr_inputs = curr_inputs[:-num_lags, :]
            # else:
            #     num_lags = inputs_num_lags

            # y_target is the time series we are trying to predict from A_hat @ y_history
            # y_target should start at t=0+num_lags
            # y_target = np.zeros((num_time - num_lags, num_neurons))
            # y_target is the lagged time series, should start at t=0+num_lags-1
            # we will concatenate each of the columns of the y_history matrix where each column corresponds to a lagged time
            # series
            # y_history = np.zeros((num_time - num_lags, 0))
            # input_history = np.zeros((num_time - num_lags, 0))

            # note this goes from time num_lags to T

            y_history = get_lagged_data(non_nan_emissions, emissions_num_lags, add_pad=False)
            input_history = get_lagged_data(curr_inputs, inputs_num_lags, add_pad=False)

            len_diff = np.abs(emissions_num_lags - inputs_num_lags)
            len_max = np.max((emissions_num_lags, inputs_num_lags))

            if y_history.shape[0] > input_history.shape[0]:
                y_history = y_history[len_diff:-1, :]
                input_history = input_history[:-1, :]
            else:
                input_history = input_history[len_diff:-1, :]
                y_history = y_history[:-1, :]

            y_target = non_nan_emissions[len_max:, :]

            # # build lagged y_history from emissions (x_(t-1))
            # for p in reversed(range(emissions_num_lags)):
            #     y_history = np.concatenate((y_history, non_nan_emissions[p:p - emissions_num_lags, :]), axis=1)
            #
            # # add to y_history the inputs to get input weights (u_t)
            # for p in reversed(range(inputs_num_lags)):
            #     if (p - inputs_num_lags + 1) != 0:
            #         input_history = np.concatenate((input_history, curr_inputs[(p + 1):(p - inputs_num_lags + 1), :]), axis=1)
            #     else:
            #         input_history = np.concatenate((input_history, curr_inputs[(p + 1):, :]), axis=1)

            emission_input_history = np.concatenate((y_history, input_history), axis=1)

            # create a mask for the dynamics_input_weights. This allows us to fit dynamics weights that are diagonal
            input_mask = np.tile(weights_mask.T, (inputs_num_lags, 1))
            input_mask = np.concatenate(
                (np.ones((non_nan_emissions.shape[1] * emissions_num_lags, non_nan_emissions.shape[1])),
                 input_mask), axis=0)

            # a_hat is a col vector of each A_hat_p matrix for each lag p -> need to transpose each A_hat_p
            num_emission_neurons = len(non_nan_emissions[0, :])
            num_input_neurons = len(curr_inputs[0, :])

            # ab_hat = np.linalg.lstsq(emission_input_history, y_target, rcond=None)[0]
            # instead do masking from utils to get rid of the 0 entries and get proper fitting
            ab_hat = iu.solve_masked(emission_input_history, y_target, input_mask)

            a_hat = ab_hat[:emissions_num_lags * num_emission_neurons, :].T
            b_hat = ab_hat[emissions_num_lags * num_emission_neurons:, :].T

            # # make impulse response function plots for a specific subset of neurons
            # cell_ids_chosen = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']
            # neuron_inds_chosen = np.array([cell_ids.index(i) for i in cell_ids_chosen])
            #
            # # get vector of input neurons' fitted weights for each lag
            # b_split = np.split(b_hat, num_lags, axis=1)
            # b = [i[weights_mask.astype(bool)] for i in b_split]
            # b = [i[:, None] for i in b]
            # b = np.concatenate(b, axis=1)
            # # b = b[neuron_inds_chosen, :]
            #
            # plt.figure()
            # plt.imshow(b, aspect='auto', interpolation='nearest', cmap=colormap)
            # plt.colorbar()
            # plt.title('fitted input neuron weights vs. lags in time')
            # cell_ids_array = np.array(cell_ids)[has_stims]
            # plot_y = np.arange(cell_ids_array.size)
            # plt.yticks(plot_y, cell_ids_array)
            # string = fig_path + 'bhat_lags%i.png' % d
            # plt.savefig(string)
            # # plt.show()

            y_hat = emission_input_history @ ab_hat
            mse[d] = np.mean((y_target - y_hat) ** 2)

            # add NaNs back in for plotting and to compare across datasets

            # # plot just a_hat without nans added in
            # plt.figure()
            # a_hat_pos = plt.imshow(a_hat, aspect='auto', interpolation='nearest', cmap=colormap)
            # # plt.clim((-color_limits_a_hat, color_limits_a_hat))
            # plt.colorbar(a_hat_pos)
            # plt.show()

            # fill in nans across first dimension
            a_split = np.split(a_hat, emissions_num_lags, axis=1)
            a_hat_full = np.zeros((num_neurons, num_neurons * emissions_num_lags))
            a_hat_full_0 = np.zeros((num_neurons, num_neurons * emissions_num_lags))
            a_hat_full[:] = np.nan
            a_hat_full_0[:] = np.nan
            temp_nan = np.zeros((num_emission_neurons, num_neurons))
            for i in range(emissions_num_lags):
                temp_nan[:] = np.nan
                temp_nan[:, ~nans] = a_split[i][:, :num_emission_neurons]
                # fill in nans across second dimension
                a_hat_full_split = np.zeros((num_neurons, num_neurons))
                a_hat_full_split[:] = np.nan
                a_hat_full_split[~nans, :] = temp_nan
                a_hat_full[:, num_neurons * i:num_neurons * (i + 1)] = a_hat_full_split[:]
                np.fill_diagonal(a_hat_full_split, 0.0)
                a_hat_full_0[:, num_neurons * i:num_neurons * (i + 1)] = a_hat_full_split[:]

            # set diagonal elements = 0 for plotting
            # np.fill_diagonal(a_hat_full, 0.0)

            b_split = np.split(b_hat, inputs_num_lags, axis=1)
            b_hat_full = np.zeros((num_neurons, num_neurons * inputs_num_lags))
            b_hat_full[:] = np.nan
            temp_nan = np.zeros((num_emission_neurons, num_neurons))
            for i in range(inputs_num_lags):
                temp_nan[:] = np.nan
                temp_nan[:, has_stims] = b_split[i][:, :num_input_neurons]
                # fill in nans across second dimension
                b_hat_full_split = np.zeros((num_neurons, num_neurons))
                b_hat_full_split[:] = np.nan
                b_hat_full_split[~nans, :] = temp_nan
                b_hat_full[:, num_neurons * i:num_neurons * (i + 1)] = b_hat_full_split[:]

            all_a_hat[:, :, d] = a_hat_full
            all_a_hat_0[:, :, d] = a_hat_full_0
            all_b_hat[:, :, d] = b_hat_full

        with open(load_dir + f_name, 'wb') as f:
            pickle.dump((all_a_hat, all_a_hat_0, all_b_hat, mse), f)

    return all_a_hat, all_a_hat_0, all_b_hat, mse


# from matt
def get_lagged_data(data, lags, add_pad=True):
    num_time, num_neurons = data.shape

    if add_pad:
        final_time = num_time
        pad = np.zeros((lags - 1, num_neurons))
        data = np.concatenate((pad, data), axis=0)
    else:
        final_time = num_time - lags + 1

    lagged_data = np.zeros((final_time, 0))

    for tau in reversed(range(lags)):
        if tau == lags - 1:
            lagged_data = np.concatenate((lagged_data, data[tau:, :]), axis=1)
        else:
            lagged_data = np.concatenate((lagged_data, data[tau:-lags + tau + 1, :]), axis=1)

    return lagged_data


def get_lagged_weights(weights, lags_out, fill='eye'):
    lagged_weights = np.concatenate(np.split(weights, weights.shape[0], 0), 2)[0, :, :]

    if fill == 'eye':
        fill_mat = np.eye(lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1])
    elif fill == 'zeros':
        fill_mat = np.zeros((lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1]))
    else:
        raise Exception('fill value not recognized')

    lagged_weights = np.concatenate((lagged_weights, fill_mat), 0)

    return lagged_weights


# if save=True, will save plots to fig_path instead of showing
def plot_weights_all_data_sets(weight_mtx, num_data_sets, colormap, color_lims, num_lags, save=False, fig_path='',
                               data_name=''):
    for d in range(num_data_sets):
        plt.figure()
        title_str = 'dataset %(dataset)i GC for %(lags)i lags: ' + data_name % {"dataset": d, "lags": num_lags}
        plt.title(title_str)
        pos = plt.imshow(weight_mtx[:, :, d], aspect='auto', interpolation='none', cmap=colormap)
        plt.clim((-color_lims, color_lims))
        plt.colorbar(pos)
        if save:
            string = fig_path + data_name + '%i.png' % d
            plt.savefig(string)
        else:
            plt.show()
    plt.close()

def plot_weights_mean_median(weight_mtx, colormap, save=False, fig_path='', data_title='', data_name=''):
    color_lims = np.nanquantile(np.abs(weight_mtx).flatten(), 0.99)
    if 'avg_b_hat' == data_name or 'median_b_hat' == data_name:
        color_lims = np.nanquantile(np.abs(weight_mtx).flatten(), 0.995)
    # should i be making separate color limits/colorbar for each avg/med mtx or use the original colorbar from all
    # datasets?

    plt.figure()
    plt.title(data_title)
    pos = plt.imshow(weight_mtx, aspect='auto', interpolation='none', cmap=colormap)
    plt.clim((-color_lims, color_lims))
    plt.colorbar(pos)
    if save:
        string = fig_path + data_name + '.png'
        plt.savefig(string)
    else:
        plt.show()


def plot_weights_mean_median_split(weight_mtx, colormap, num_lags, save=False, fig_path='', data_title='',
                                   data_name=''):
    color_lims = np.nanquantile(np.abs(weight_mtx).flatten(), 0.99)
    # should i be making separate color limits/colorbar for each avg/med mtx or use the original colorbar from all
    # datasets?

    num_cols = num_lags // 2 + num_lags % 2
    fig, axs = plt.subplots(2, num_cols)
    plt.suptitle(data_title)
    temp = np.split(weight_mtx, num_lags, axis=1)
    for i in range(2):
        for j in range(num_cols):
            ix = i * num_cols + j
            if len(temp) == 2:
                pos = axs[i].imshow(temp[i], aspect='auto', interpolation='none', cmap=colormap,
                                    norm=plt.Normalize(vmin=-color_lims, vmax=color_lims))
            elif ix < len(temp):
                pos = axs[i, j].imshow(temp[ix], aspect='auto', interpolation='none', cmap=colormap,
                                       norm=plt.Normalize(vmin=-color_lims, vmax=color_lims))
    plt.tight_layout()
    fig.colorbar(pos, ax=axs.ravel().tolist(), pad=0.04, aspect=30)
    if save:
        string = fig_path + data_name + '.png'
        plt.savefig(string)
    else:
        plt.show()

# sample from model for a specific stimulated neuron for num_sim
def impulse_response_func(num_sim, cell_ids, cell_ids_chosen, num_neurons, num_data_sets, emissions, inputs, all_a_hat,
                          all_b_hat, emissions_num_lags, inputs_num_lags, f_name='impulse_response_data',
                          load_dir='/Users/lsmith/Documents/python/', rerun=False):
    f_name = f_name + str(emissions_num_lags) + str(inputs_num_lags) + '.pkl'
    if os.path.exists(load_dir + f_name) and os.path.isfile(load_dir + f_name) and not rerun:
        with open(load_dir + f_name, 'rb') as f:
            avg_pred_x_all_data = pickle.load(f)

    else:
        avg_pred_x = np.zeros((num_sim, num_neurons, num_data_sets))
        # fixed problem: was overwriting avg_pred_x_all_data for each stim neuron, so it only saved the last neuron's
        # responses
        # now it is a vector of time points x all responding neurons x subset of stim neurons
        avg_pred_x_all_data = np.zeros((num_sim, num_neurons, len(cell_ids_chosen)))
        a_hat_avg = np.nanmean(all_a_hat, axis=2)
        a_hat_avg[np.isnan(a_hat_avg)] = 0
        for n in range(len(cell_ids_chosen)):
            neuron_to_stim = cell_ids_chosen[n]
            avg_pred_x[:] = 0
            for d in range(num_data_sets):
                curr_emissions = emissions[d].copy()
                curr_emissions[np.isnan(curr_emissions)] = 0
                temp_a_hat = all_a_hat[:, :, d].copy()
                temp_a_hat[np.isnan(temp_a_hat)] = 0
                temp_b_hat = all_b_hat[:, :, d].copy()
                temp_b_hat[np.isnan(temp_b_hat)] = 0

                # build the a_bar mtx: block mtx
                a_hat_split = np.array(np.split(temp_a_hat, emissions_num_lags, axis=1))
                a_bar = get_lagged_weights(a_hat_split, emissions_num_lags)
                # use averaged ahat instead
                # a_hat_split = np.array(np.split(a_hat_avg, emissions_num_lags, axis=1))
                # a_bar = get_lagged_weights(a_hat_split, emissions_num_lags)

                # b_bar mtx
                b_hat_split = np.array(np.split(temp_b_hat, inputs_num_lags, axis=1))
                b_bar = get_lagged_weights(b_hat_split, emissions_num_lags, fill='zeros')

                # use np.where to find locations where inputs are stimulated, get back the rows of where a certain
                # neuron was stimmed, then use these row locations for the simulation

                stim_times = np.where(inputs[d][:, cell_ids.index(neuron_to_stim)])[0]

                # inputs
                u_t_bar = get_lagged_data(inputs[d], inputs_num_lags)

                # inputs can be calculated outside the simulation
                system_inputs = u_t_bar @ b_bar.T

                pred_x_0_col = np.zeros((emissions_num_lags * num_neurons, stim_times.size))
                # instead of initializing with previous emissions before stim, set to zeros
                # for i in range(stim_times.size):
                #     # get initial values of emissions before stimulation
                #     pred_x_0 = curr_emissions[stim_times[i] - emissions_num_lags:stim_times[i], :]
                #     pred_x_0_col[:, i] = np.reshape(pred_x_0, (emissions_num_lags * num_neurons))

                pred_x_bar = np.zeros((num_sim, num_neurons * emissions_num_lags, stim_times.size))
                pred_x_bar[0, :, :] = pred_x_0_col
                pred_x = np.zeros((num_sim, num_neurons, stim_times.size))

                for i in range(stim_times.size):
                    for t in range(1, num_sim):
                        if t+stim_times[i] > system_inputs.shape[0]-1:
                            pred_x_bar[t, :, i] = a_bar @ pred_x_bar[t - 1, :, i]
                        else:
                            pred_x_bar[t, :, i] = a_bar @ pred_x_bar[t - 1, :, i] + system_inputs[t+stim_times[i], :]

                    pred_x[:, :, i] = pred_x_bar[:, :num_neurons, i]

                # average the predicted model response after the stimulation over all stimulation events
                avg_pred_x[:, :, d] = np.nanmean(pred_x, axis=2)

            # average (across all datasets) the predicted responses in all neurons after stim
            avg_pred_x_all_data[:, :, n] = np.nanmean(avg_pred_x, axis=2)

    with open(load_dir + f_name, 'wb') as f:
        pickle.dump(avg_pred_x_all_data, f)

    return avg_pred_x_all_data

def plot_l2_norms(emissions, inputs, cell_ids, cell_ids_chosen, avg_pred_x_all_data, colormap, a_hat_avg, save=False,
                  fig_path=''):
    # plotting from matt code
    window = (0, 120)
    # list of neuron indices
    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]

    plot_x = np.arange(len(chosen_neuron_inds))
    measured_stim_responses = au.get_impulse_response_function(emissions, inputs, sub_pre_stim=True, window=window)[0]
    measured_response_norm = au.p_norm(measured_stim_responses, axis=0)
    measured_response_norm = measured_response_norm[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    measured_response_norm[np.eye(measured_response_norm.shape[0], dtype=bool)] = np.nan

    pred_response_norm = au.p_norm(avg_pred_x_all_data, axis=0)
    pred_response_norm_plot = pred_response_norm[chosen_neuron_inds, :]

    # set diag = nan
    pred_response_norm_plot[np.eye(pred_response_norm_plot.shape[0], dtype=bool)] = np.nan

    a_hat_avg = a_hat_avg[np.ix_(chosen_neuron_inds, chosen_neuron_inds)]
    a_hat_avg[np.eye(a_hat_avg.shape[0], dtype=bool)] = np.nan

    # correlations:
    corr_data, corr_data_subset = get_correlation(emissions, inputs, cell_ids, cell_ids_chosen, avg_pred_x_all_data,
                                                  colormap, save=True, fig_path=fig_path)
    corr_data_subset[np.eye(corr_data_subset.shape[0], dtype=bool)] = np.nan

    # always normalize the corr and ahat weights!
    corr_data_subset = corr_data_subset / np.nanmax(corr_data_subset)
    a_hat_avg = a_hat_avg / np.nanmax(a_hat_avg)

    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw=dict(hspace=0.25, wspace=0.25))
    obj = ax[0, 0].imshow(measured_response_norm, interpolation='nearest', cmap=colormap)
    ax[0, 0].set_title('measured response L2 norm')
    ax[0, 0].set_xticks(plot_x, cell_ids_chosen)
    ax[0, 0].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[0, 0].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    obj = ax[0, 1].imshow(pred_response_norm_plot, interpolation='nearest', cmap=colormap)
    ax[0, 1].set_title('model response L2 norm')
    ax[0, 1].set_xticks(plot_x, cell_ids_chosen)
    ax[0, 1].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[0, 1].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    obj = ax[1, 0].imshow(corr_data_subset, interpolation='nearest', cmap=colormap)
    ax[1, 0].set_title('correlation subset')
    ax[1, 0].set_xticks(plot_x, cell_ids_chosen)
    ax[1, 0].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[1, 0].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    obj = ax[1, 1].imshow(a_hat_avg, interpolation='nearest',
                          cmap=colormap)
    ax[1, 1].set_title('avg model weights')
    ax[1, 1].set_xticks(plot_x, cell_ids_chosen)
    ax[1, 1].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[1, 1].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    if save:
        string = fig_path + 'l2norm.png'
        plt.savefig(string)
    else:
        plt.show()

    plt.figure()
    obj = plt.imshow(corr_data, interpolation='nearest', cmap=colormap)
    plt.title('correlation')
    plt.clim((-1, 1))
    plt.colorbar(obj)

    if save:
        string = fig_path + 'corr.png'
        plt.savefig(string)
    else:
        plt.show()

    # normalize:
    measured_response_norm = measured_response_norm / np.nanmax(measured_response_norm)
    pred_response_norm_plot = pred_response_norm_plot / np.nanmax(pred_response_norm_plot)

    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw=dict(hspace=0.25, wspace=0.25))
    plt.suptitle("normalized")
    obj = ax[0, 0].imshow(measured_response_norm, interpolation='nearest', cmap=colormap)
    ax[0, 0].set_title('measured response L2 norm')
    ax[0, 0].set_xticks(plot_x, cell_ids_chosen)
    ax[0, 0].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[0, 0].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    obj = ax[0, 1].imshow(pred_response_norm_plot, interpolation='nearest', cmap=colormap)
    ax[0, 1].set_title('model response L2 norm')
    ax[0, 1].set_xticks(plot_x, cell_ids_chosen)
    ax[0, 1].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[0, 1].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    obj = ax[1, 0].imshow(corr_data_subset, interpolation='nearest', cmap=colormap)
    ax[1, 0].set_title('correlation subset')
    ax[1, 0].set_xticks(plot_x, cell_ids_chosen)
    ax[1, 0].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[1, 0].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    obj = ax[1, 1].imshow(a_hat_avg, interpolation='nearest', cmap=colormap)
    ax[1, 1].set_title('avg model weights')
    ax[1, 1].set_xticks(plot_x, cell_ids_chosen)
    ax[1, 1].set_yticks(plot_x, cell_ids_chosen)
    for label in ax[1, 1].get_xticklabels():
        label.set_rotation(90)
    obj.set_clim((-1, 1))

    if save:
        string = fig_path + 'l2norm_normalized.png'
        plt.savefig(string)
    else:
        plt.show()


def plot_imp_resp(emissions, inputs, neuron_inds_chosen, num_neurons, num_data_sets, cell_ids, cell_ids_chosen,
                  neuron_to_stim, avg_pred_x_all_data, save=False, fig_path=''):
    sample_rate = 0.5
    window = (-60, 120)
    # list of neuron indices
    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]
    measured_stim_responses = au.get_impulse_response_function(emissions, inputs, sub_pre_stim=True, window=window)[0]
    # measured_response_norm = au.p_norm(measured_stim_responses, axis=0)
    # measured_response_norm = measured_response_norm[chosen_neuron_inds, :][:, chosen_neuron_inds]
    # measured_response_norm[np.eye(measured_response_norm.shape[0], dtype=bool)] = 0
    # measured_response_norm = measured_response_norm / np.nanmax(measured_response_norm)

    # # add on the avg emissions initial conditions before the stimulus
    # model_pred_resp = np.zeros((np.abs(window[0]), num_neurons, num_data_sets))
    # for d in range(num_data_sets):
    #     stim_times = np.where(inputs[d][:, cell_ids.index(neuron_to_stim)])[0]
    #     temp = np.zeros((np.abs(window[0]), num_neurons, stim_times.size))
    #     for i in range(stim_times.size):
    #         # get initial values of emissions before stimulation
    #         if (stim_times[i] - np.abs(window[0])) < 0:
    #             temp[(np.abs(window[0]) - stim_times[i]):, :, i] = emissions[d][:stim_times[i], :]
    #         else:
    #             temp[:, :, i] = emissions[d][(stim_times[i] - np.abs(window[0])):stim_times[i], :]
    #     model_pred_resp[:, :, d] = np.nanmean(temp, axis=2)
    # model_pred_resp_avg = np.nanmean(model_pred_resp, axis=2)

    # get the index of the neuron that was stimulated out of all the cell indices
    neuron_to_stim_ind = cell_ids.index(neuron_to_stim)
    # but we already removed all the other indices except for the subset we're analyzing, so find the corr. index of the
    # stim neuron out of the subset
    neuron_to_stim_ind = chosen_neuron_inds.index(neuron_to_stim_ind)

    # time x all responding neurons for the stim neuron
    # plot_model_resp = np.concatenate((model_pred_resp_avg, avg_pred_x_all_data[:, :, neuron_to_stim_ind]), axis=0)
    plot_model_resp = np.concatenate((measured_stim_responses[:60, :, cell_ids.index(neuron_to_stim)],
                                      avg_pred_x_all_data[:, :, neuron_to_stim_ind]), axis=0)

    # pull out subset of meas stim responses to compare against model
    measured_stim_responses = measured_stim_responses[:, chosen_neuron_inds, :]

    plot_x = np.arange(window[0], window[1]) * sample_rate
    temp = np.concatenate((measured_stim_responses[:, :, cell_ids.index(neuron_to_stim)], plot_model_resp[:, neuron_inds_chosen]),
                          axis=1)
    ylim = (np.nanmin(temp), np.nanmax(temp))

    # then plot the measured results from the experimental data
    for i in range(len(neuron_inds_chosen)):
        plt.figure()
        plt.title('stimulated input neuron: ' + neuron_to_stim)
        plt.plot(plot_x, measured_stim_responses[:, i, cell_ids.index(neuron_to_stim)])
        plt.plot(plot_x, plot_model_resp[:, neuron_inds_chosen[i]])
        plt.xlabel('time')
        plt.ylabel('avg response in ' + cell_ids_chosen[i])
        plt.axvline(0, color='k', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.ylim(ylim)
        plt.legend(['measured', 'model'])

        if save:
            string = fig_path + 'response in ' + cell_ids_chosen[i] + '.png'
            plt.savefig(string)
        else:
            plt.show()


def plot_input_weights_neurons(b_hat, num_lags, cell_ids, colormap, data_name, subset, subset_inds, save=False,
                               fig_path=''):
    color_lims = np.nanquantile(np.abs(b_hat).flatten(), 0.99)
    # get vector of input neurons' fitted weights for each lag
    b_split = np.split(b_hat, num_lags, axis=1)
    b = [np.diag(i) for i in b_split]
    b = [i[:, None] for i in b]
    b = np.concatenate(b, axis=1)
    if subset:
        b = b[subset_inds, :]
        temp = cell_ids
        cell_ids = []
        for i in range(subset_inds.size):
            cell_ids.append(temp[subset_inds[i]])

    plt.figure()
    plt.imshow(b, aspect='auto', interpolation='none', cmap=colormap,
               norm=plt.Normalize(vmin=-color_lims, vmax=color_lims))
    plt.colorbar()
    plt.title('fitted input neuron weights vs. lags: ' + data_name)
    cell_ids_array = np.array(cell_ids)
    plot_y = np.arange(cell_ids_array.size)
    if not subset:
        plt.yticks(plot_y, cell_ids_array, fontsize=2)
    else:
        plt.yticks(plot_y, cell_ids_array, fontsize=10)
    if save:
        string = fig_path + 'bhat_lags_' + data_name + '.png'
        plt.savefig(string)
    else:
        plt.show()

def nancorrcoef(data):
    num_data = len(data)
    corr_coef = np.zeros((num_data, num_data))

    for ii, i in enumerate(data):
        for ji, j in enumerate(data):
            i = (i - np.nanmean(i, axis=0)) / np.nanstd(i, ddof=1)
            j = (j - np.nanmean(j, axis=0)) / np.nanstd(j, ddof=1)

            corr_coef[ii, ji] = np.nanmean(i * j)

    return corr_coef
def get_correlation(emissions, inputs, cell_ids, cell_ids_chosen, avg_pred_x_all_data, colormap, save=False,
                     fig_path=''):
    window = (0, 120)
    # list of neuron indices
    chosen_neuron_inds = [cell_ids.index(i) for i in cell_ids_chosen]

    # measured_stim_responses = au.get_impulse_response_function(emissions, inputs, sub_pre_stim=True, window=(0, emissions[0].shape[0]))
    measured_stim_responses = au.get_impulse_response_function(emissions, inputs, sub_pre_stim=True, window=window)
    measured_response_norm = au.p_norm(measured_stim_responses[0], axis=0)
    measured_response_norm = measured_response_norm[chosen_neuron_inds, :][:, chosen_neuron_inds]
    measured_response_norm[np.eye(measured_response_norm.shape[0], dtype=bool)] = 0
    # measured_response_norm = measured_response_norm / np.nanmax(measured_response_norm)

    pred_response_norm = au.p_norm(avg_pred_x_all_data, axis=0)
    pred_response_norm_plot = pred_response_norm[chosen_neuron_inds, :]

    # set diag = 0 and normalize
    pred_response_norm_plot[np.eye(pred_response_norm_plot.shape[0], dtype=bool)] = 0
    # pred_response_norm_plot = pred_response_norm_plot / np.nanmax(pred_response_norm_plot)

    corr_meas_to_pred = nancorrcoef([measured_response_norm.flatten(), pred_response_norm_plot.flatten()])[0, 1]
    corr_data = np.abs(au.nan_corr_data(emissions))

    # set diag = 0 and normalize
    corr_data[np.eye(corr_data.shape[0], dtype=bool)] = 0
    # corr_data = corr_data / np.nanmax(corr_data)

    corr_data_subset = corr_data[:, chosen_neuron_inds][chosen_neuron_inds, :]
    corr_meas_to_corr = nancorrcoef([measured_response_norm.flatten(), corr_data_subset.flatten()])[0, 1]
    corr_pred_to_corr = nancorrcoef([pred_response_norm_plot.flatten(), corr_data_subset.flatten()])[0, 1]

    plt.figure()
    values = [corr_meas_to_pred, corr_meas_to_corr, corr_pred_to_corr]
    labels = ['corr_meas_to_pred', 'corr_meas_to_corr', 'corr_pred_to_corr']
    plt.bar(labels, values)
    plt.title('correlations')

    if save:
        string = fig_path + 'corr_bar.png'
        plt.savefig(string)
    else:
        plt.show()

    return corr_data, corr_data_subset

def plot_dynamics_subset(weight_mtx, cell_ids_chosen, chosen_neuron_inds, colormap, num_lags, save=False, fig_path='',
                         data_name=''):
    num_cols = num_lags // 2 + num_lags % 2
    fig, ax = plt.subplots(2, num_cols, figsize=(14, 9), subplot_kw=dict(box_aspect=1),
                           gridspec_kw=dict(hspace=0.25, wspace=0.25))
    plot_x = np.arange(len(chosen_neuron_inds))
    plt.suptitle("a_hat subset " + data_name)

    temp = np.split(weight_mtx, num_lags, axis=1)
    temp = [temp[i][chosen_neuron_inds, :][:, chosen_neuron_inds] for i in range(len(temp))]
    color_lims = np.nanquantile(np.abs(np.array(temp)).flatten(), 0.99)

    for i in range(2):
        for j in range(num_cols):
            ix = i * num_cols + j
            if len(temp) == 2:
                pos = ax[i].imshow(temp[i], aspect='auto', interpolation='none', cmap=colormap,
                                   norm=plt.Normalize(vmin=-color_lims, vmax=color_lims))
                ax[i].set_xticks(plot_x, cell_ids_chosen)
                ax[i].set_yticks(plot_x, cell_ids_chosen)
                ax[i].set_title(f"lag {i+1}")
                for label in ax[i].get_xticklabels():
                    label.set_rotation(90)
            elif ix < len(temp):
                pos = ax[i, j].imshow(temp[ix], aspect='auto', interpolation='none', cmap=colormap,
                                      norm=plt.Normalize(vmin=-color_lims, vmax=color_lims))
                ax[i, j].set_title(f"lag {ix+1}")
                ax[i, j].set_xticks(plot_x, cell_ids_chosen)
                ax[i, j].set_yticks(plot_x, cell_ids_chosen)
                for label in ax[i, j].get_xticklabels():
                    label.set_rotation(90)
    # fig.colorbar(pos, ax=ax.ravel().tolist(), pad=0.04, aspect=30)
    fig.colorbar(pos, ax=ax)
    if save:
        string = fig_path + 'ahat_subset_' + data_name + '.png'
        plt.savefig(string)
    else:
        plt.show()