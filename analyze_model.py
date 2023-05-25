import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path


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


def get_stim_response(emissions, inputs, window=(-10, 10)):
    num_neurons = emissions[0].shape[1]

    # get structure which is responding neuron x stimulated neuron
    stim_responses = np.zeros((num_neurons, num_neurons, window[1] - window[0]))
    num_stims = np.zeros(num_neurons)

    for e, i in zip(emissions, inputs):
        num_time = e.shape[0]
        stim_events = np.where(i)

        for time, target in zip(stim_events[0], stim_events[1]):
            if window[0] + time >= 0 and window[1] + time < num_time:
                responses = e[window[0]+time:window[1]+time, :]
                num_stims[target] += 1
                for r in range(num_neurons):
                    stim_responses[r, target, :] = stim_responses[r, target, :] + responses[:, r]

    stim_responses = stim_responses / num_stims[None, :, None]

    return stim_responses


def predict_from_model(model_folder):
    data_set_ind = 0
    num_time = 500
    colormap = mpl.colormaps['coolwarm']

    # load in the model and the data
    model_path = model_folder / 'model_trained.pkl'
    data_path = model_folder / 'data.pkl'

    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    data_file = open(data_path, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']

    # grab the data set you want to analyze
    emission_chosen = emissions[data_set_ind][:num_time, :]
    input_chosen = inputs[data_set_ind][:num_time, :]
    measured_neurons = ~np.any(np.isnan(emission_chosen), axis=0)
    emission_chosen = emission_chosen[:, measured_neurons]
    input_chosen = input_chosen[:, measured_neurons]

    # look at the model posterior and see if it's a good representation of the data
    emissions_torch, inputs_torch = model.standardize_inputs(emissions, inputs)
    init_mean_torch = model.estimate_init_mean(emissions_torch)
    init_cov_torch = model.estimate_init_cov(emissions_torch)
    emissions_chosen_torch = emissions_torch[data_set_ind]
    inputs_chosen_torch = inputs_torch[data_set_ind]
    init_mean_chosen_torch = init_mean_torch[data_set_ind]
    init_cov_chosen_torch = init_cov_torch[data_set_ind]

    # get the posterior
    posterior = [model.lgssm_smoother(emissions_torch[i], inputs_torch[i], init_mean_torch[i], init_cov_torch[i])[3] for i in range(len(emissions))]
    posterior = [(i @ model.emissions_weights.T).detach().cpu().numpy()[:num_time, :] for i in posterior]
    posterior_chosen = posterior[data_set_ind]

    # get the posterior with a neuron missing
    emissions_chosen_missing_torch = emissions_chosen_torch
    emissions_chosen_missing_torch[:, 2] = np.nan
    posterior_missing = model.lgssm_smoother(emissions_chosen_missing_torch, inputs_chosen_torch, init_mean_chosen_torch, init_cov_chosen_torch)[3]
    posterior_missing = (posterior_missing @ model.emissions_weights.T).detach().cpu().numpy()[:num_time, :]

    # draw from the prior but with the stimulation events from the data
    sampled_prior = model.sample(num_time, inputs_list=inputs, add_noise=False)['emissions']
    sampled_prior_measured = sampled_prior[data_set_ind][:, measured_neurons]

    cmax = np.max(emission_chosen)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(emission_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('measured data')
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    cmax = np.max(sampled_prior_measured)

    plt.subplot(3, 1, 2)
    plt.imshow(sampled_prior_measured.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('model prior')
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(input_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('stimulation events')
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-1, 1))
    plt.colorbar()

    plt.tight_layout()

    # calculate the posterior with a neuron missing
    cmax = np.max((emission_chosen, posterior_chosen, posterior_missing))

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(emission_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('measured data')
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(posterior_chosen.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('model posterior, all data')
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(posterior_missing.T, interpolation='nearest', aspect='auto', cmap=colormap)
    plt.title('model posterior, third neuron missing')
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))
    plt.colorbar()

    plt.tight_layout()

    # go through emissions and get the average input response for each neuron
    measured_stim_responses = get_stim_response(emissions, inputs)
    prior_stim_responses = get_stim_response(sampled_prior, inputs)
    posterior_stim_responses = get_stim_response(posterior, inputs)
    model_weights = model.dynamics_weights[:model.dynamics_dim, :]

    measured_response_power = np.sum(measured_stim_responses**2, axis=2)**(1/2)
    measured_response_power = measured_response_power / np.max(measured_response_power)
    prior_response_power = np.sum(prior_stim_responses**2, axis=2)**(1/2)
    prior_response_power = prior_response_power / np.max(prior_response_power)
    posterior_response_power = np.sum(posterior_stim_responses**2, axis=2)**(1/2)
    posterior_response_power = posterior_response_power / np.max(posterior_response_power)
    model_weights_power = np.sum(np.stack(np.split(model_weights**2, model.dynamics_lags, axis=1)), axis=0)**(1/2)
    model_weights_power = model_weights_power / np.max(model_weights_power)

    cmax = np.max((measured_response_power, prior_response_power, posterior_response_power, model_weights_power))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(prior_response_power, interpolation='nearest', cmap=colormap)
    plt.title('prior response power')
    plt.xticks(np.arange(model.dynamics_dim), cell_ids)
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))

    plt.subplot(2, 2, 2)
    plt.imshow(posterior_response_power, interpolation='nearest', cmap=colormap)
    plt.title('posterior response power')
    plt.xticks(np.arange(model.dynamics_dim), cell_ids)
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))

    plt.subplot(2, 2, 3)
    plt.imshow(measured_response_power, interpolation='nearest', cmap=colormap)
    plt.title('measured response power')
    plt.xticks(np.arange(model.dynamics_dim), cell_ids)
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))

    plt.subplot(2, 2, 4)
    plt.imshow(model_weights_power, interpolation='nearest', cmap=colormap)
    plt.title('model weights power')
    plt.xticks(np.arange(model.dynamics_dim), cell_ids)
    plt.yticks(np.arange(model.dynamics_dim), cell_ids)
    plt.clim((-cmax, cmax))

    plt.tight_layout()

    plt.show()


model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models')
model_name = Path('local')

predict_from_model(model_folder / model_name)
# plot_model_params(model_folder / model_name)
