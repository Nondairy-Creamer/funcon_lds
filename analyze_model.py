import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path


def rms(data, axis=-1):
    return np.sqrt(np.mean(data**2, axis=axis))


def stack_weights(weights, num_split, axis=-1):
    return np.stack(np.split(weights, num_split, axis=axis))


def plot_model_params(model_folder):
    colormap = mpl.colormaps['coolwarm']

    model_path = model_folder / 'model_trained.pkl'

    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    model_params = model.get_run_params()
    A_full = model_params['trained']['dynamics_weights'][:model.dynamics_dim, :]
    # A_full = A_full - np.eye(A_full.shape[0], A_full.shape[1])
    A = np.split(A_full, model.dynamics_lags, axis=1)
    cmax = np.max(np.abs(A_full))

    for ai, aa in enumerate(A):
        plt.figure()
        plt.imshow(aa, cmap=colormap)
        plt.clim((-cmax, cmax))
        plt.colorbar()
        plt.title('A' + str(ai))

    B_full = model_params['trained']['dynamics_input_weights'][:model.dynamics_dim, :]
    B = np.split(B_full, model.dynamics_input_lags, axis=1)
    cmax = np.max(np.abs(B_full))

    for bi, bb in enumerate(B):
        plt.figure()
        plt.imshow(bb, cmap=colormap)
        plt.clim((-cmax, cmax))
        plt.colorbar()
        plt.title('B' + str(bi))

    Q = model_params['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim]
    cmax = np.max(np.abs(Q))
    plt.figure()
    plt.imshow(model_params['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim], cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('Q')
    plt.colorbar()

    R = model_params['trained']['emissions_cov']
    cmax = np.max(np.abs(R))
    plt.figure()
    plt.imshow(model_params['trained']['emissions_cov'], cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('R')
    plt.colorbar()

    plt.show()


def get_stim_response(emissions, inputs, window=(0, 10)):
    num_neurons = emissions[0].shape[1]

    # get structure which is responding neuron x stimulated neuron
    stim_responses = np.zeros((num_neurons, num_neurons, window[1] - window[0]))
    num_stims = np.zeros((num_neurons, num_neurons))

    for e, i in zip(emissions, inputs):
        num_time = e.shape[0]
        stim_events = np.where(i)

        for time, target in zip(stim_events[0], stim_events[1]):
            if window[0] + time >= 0 and window[1] + time < num_time:
                responses = e[window[0]+time:window[1]+time, :]
                nan_responses = np.any(np.isnan(responses), axis=0)
                responses[:, nan_responses] = 0
                num_stims[target, ~nan_responses] += 1
                for r in range(num_neurons):
                    stim_responses[r, target, :] = stim_responses[r, target, :] + responses[:, r]

    num_stims[num_stims == 0] = 1
    stim_responses = stim_responses / num_stims[:, :, None]

    return stim_responses


def find_stim_events(inputs, measured, window_size=1000):
    max_data_set = 0
    max_ind = 0
    max_val = 0
    max_window = 0

    for ii, i in enumerate(inputs):
        this_window_size = np.min((window_size, i.shape[0]))
        t_filt = np.ones(this_window_size)
        inputs_c = np.zeros((i.shape[0] - this_window_size + 1, i.shape[1]))

        for n in range(i.shape[1]):
            inputs_c[:, n] = np.convolve(i[:, n], t_filt, mode='valid')

        total_stim = inputs_c.sum(1)
        this_max_val = np.max(total_stim) + measured[ii]
        this_max_ind = np.argmax(total_stim)

        if this_max_val > max_val:
            max_val = this_max_val
            max_ind = this_max_ind
            max_data_set = ii
            max_window = this_window_size

    return max_data_set, max_ind, max_window


def predict_from_model(model_folder, chosen_cell_ids):
    num_time = 1000
    colormap = mpl.colormaps['coolwarm']

    # load in the model and the data
    model_path = model_folder / 'model_trained.pkl'
    data_path = model_folder / 'data.pkl'
    smoothed_means_path = model_folder / 'smoothed_means.pkl'

    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    data_file = open(data_path, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    smoothed_means_file = open(smoothed_means_path, 'rb')
    posterior = pickle.load(smoothed_means_file)
    posterior = [(i @ model.emissions_weights.T).detach().cpu().numpy() for i in posterior]
    smoothed_means_file.close()

    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']

    # choose which neurons to show
    chosen_neuron_inds = np.array([list(cell_ids).index(i) for i in chosen_cell_ids])
    plot_neuron = np.arange(len(chosen_cell_ids))
    neuron_to_remove = 'AVJR'
    neuron_to_remove_ind = list(cell_ids).index(neuron_to_remove)
    neuron_to_remove_trunc_ind = chosen_cell_ids.index(neuron_to_remove)

    # get the data set / time with the most stimulation events
    inputs_trunc = [i[:, chosen_neuron_inds] for i in inputs]
    measured_emissions = [np.sum(np.any(np.isnan(i[:, chosen_neuron_inds]), axis=0)) for i in emissions]
    chosen_data_ind, time_ind, num_time = find_stim_events(inputs_trunc, measured_emissions, window_size=num_time)
    time_range = (time_ind, time_ind + num_time)

    # convert data set to torch
    emissions_torch, inputs_torch = model.standardize_inputs(emissions, inputs)
    init_mean_torch = model.estimate_init_mean(emissions_torch)
    init_cov_torch = model.estimate_init_cov(emissions_torch)

    # draw from the prior but with the stimulation events from the data
    prior = model.sample([i.shape[0] for i in emissions], num_data_sets=len(inputs), inputs_list=inputs, add_noise=False)['emissions']

    # pull out specific data sets to show
    emissions_chosen = emissions[chosen_data_ind][time_range[0]:time_range[1], chosen_neuron_inds]
    posterior_chosen = posterior[chosen_data_ind][time_range[0]:time_range[1], chosen_neuron_inds]
    prior_chosen = prior[chosen_data_ind][time_range[0]:time_range[1], chosen_neuron_inds]
    inputs_chosen = inputs[chosen_data_ind][time_range[0]:time_range[1], chosen_neuron_inds]

    # plotting
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(inputs_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-1, 1))
    plt.colorbar()

    cmax = np.nanmax(np.abs(emissions_chosen))

    plt.subplot(3, 1, 2)
    plt.imshow(emissions_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('measured data')
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    cmax = np.nanmax(np.abs(prior_chosen))

    plt.subplot(3, 1, 3)
    plt.imshow(prior_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('model prior')
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.tight_layout()

    # plot the posterior with a neuron missing
    # get the posterior with a neuron missing
    emissions_missing = emissions_torch[chosen_data_ind]
    emissions_missing[:, neuron_to_remove_ind] = np.nan
    posterior_missing = model.lgssm_smoother(emissions_missing,
                                             inputs_torch[chosen_data_ind],
                                             init_mean_torch[chosen_data_ind],
                                             init_cov_torch[chosen_data_ind])[3]
    posterior_missing = (posterior_missing @ model.emissions_weights.T).detach().cpu().numpy()[time_range[0]:time_range[1], chosen_neuron_inds]

    cmax = np.nanmax(np.abs((emissions_chosen, posterior_chosen, posterior_missing)))

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(emissions_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('measured data')
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(posterior_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('model posterior, all data')
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(posterior_missing.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('model posterior, with ' + neuron_to_remove + ' missing')
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.tight_layout()

    plt.figure()
    plt.plot(emissions_chosen[:, neuron_to_remove_trunc_ind])
    plt.plot(posterior_chosen[:, neuron_to_remove_trunc_ind])
    plt.plot(posterior_missing[:, neuron_to_remove_trunc_ind])
    plt.legend(['measured', 'full posterior', 'posterior missing'])
    plt.title([neuron_to_remove + ' missing'])

    # go through emissions and get the average input response for each neuron
    measured_stim_responses = get_stim_response(emissions, inputs)
    prior_stim_responses = get_stim_response(prior, inputs)
    posterior_stim_responses = get_stim_response(posterior, inputs)
    model_weights = model.dynamics_weights[:model.dynamics_dim, :]

    measured_response_power = rms(measured_stim_responses, axis=2)
    prior_response_power = rms(prior_stim_responses, axis=2)
    posterior_response_power = rms(posterior_stim_responses, axis=2)
    model_weights_power = rms(stack_weights(model_weights, model.dynamics_lags, axis=1), axis=0)

    measured_response_power = measured_response_power[chosen_neuron_inds, :]
    measured_response_power = measured_response_power[:, chosen_neuron_inds]
    prior_response_power = prior_response_power[chosen_neuron_inds, :]
    prior_response_power = prior_response_power[:, chosen_neuron_inds]
    posterior_response_power = posterior_response_power[chosen_neuron_inds, :]
    posterior_response_power = posterior_response_power[:, chosen_neuron_inds]
    model_weights_power = model_weights_power[chosen_neuron_inds, :]
    model_weights_power = model_weights_power[:, chosen_neuron_inds]
    model_weights_power[np.eye(model_weights_power.shape[0], dtype=bool)] = 0

    measured_response_power = measured_response_power / np.max(measured_response_power)
    prior_response_power = prior_response_power / np.max(prior_response_power)
    posterior_response_power = posterior_response_power / np.max(posterior_response_power)
    model_weights_power = model_weights_power / np.max(model_weights_power)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(prior_response_power, interpolation='nearest', cmap=colormap)
    plt.title('prior response power')
    plt.xticks(plot_neuron, chosen_cell_ids)
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-1, 1))

    plt.subplot(2, 2, 2)
    plt.imshow(posterior_response_power, interpolation='nearest', cmap=colormap)
    plt.title('posterior response power')
    plt.xticks(plot_neuron, chosen_cell_ids)
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-1, 1))

    plt.subplot(2, 2, 3)
    plt.imshow(measured_response_power, interpolation='nearest', cmap=colormap)
    plt.title('measured response power')
    plt.xticks(plot_neuron, chosen_cell_ids)
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-1, 1))

    plt.subplot(2, 2, 4)
    plt.imshow(model_weights_power, interpolation='nearest', cmap=colormap)
    plt.title('model weights power')
    plt.xticks(plot_neuron, chosen_cell_ids)
    plt.yticks(plot_neuron, chosen_cell_ids)
    plt.clim((-1, 1))

    plt.tight_layout()

    plt.show()


model_folder = Path('/home/mcreamer/Documents/data_sets/fun_con_models')
model_name = Path('48067823')
chosen_cell_ids = ['AVAL', 'AVAR', 'AVEL', 'AVER', 'AFDL', 'AFDR', 'AVJL', 'AVJR', 'AVDL', 'AVDR']

predict_from_model(model_folder / model_name, chosen_cell_ids)
# plot_model_params(model_folder / model_name)
