import numpy as np
import analysis_utilities as au
from ssm_classes import Lgssm
import warnings


def remove_nan_irfs(weights, cell_ids, data_type='test', chosen_mask=None):
    if chosen_mask is None:
        chosen_mask = np.zeros_like(weights['data'][data_type]['irms']) == 0

    data_irfs = weights['data'][data_type]['irfs'][:, chosen_mask]
    data_irfs_sem = weights['data'][data_type]['irfs_sem'][:, chosen_mask]
    model_irfs = weights['models']['synap']['irfs'][:, chosen_mask]
    model_dirfs = weights['models']['synap']['dirfs'][:, chosen_mask]
    model_rdirfs = weights['models']['synap']['rdirfs'][:, chosen_mask]
    model_eirfs = weights['models']['synap']['eirfs'][:, chosen_mask]

    data_irms = weights['data'][data_type]['irms'][chosen_mask]
    model_irms = weights['models']['synap']['irms'][chosen_mask]
    model_dirms = weights['models']['synap']['dirms'][chosen_mask]

    num_neurons = len(cell_ids['all'])
    post_synaptic = np.empty((num_neurons, num_neurons), dtype=object)
    pre_synaptic = np.empty((num_neurons, num_neurons), dtype=object)
    for ci in range(num_neurons):
        for cj in range(num_neurons):
            post_synaptic[ci, cj] = cell_ids['all'][ci]
            pre_synaptic[ci, cj] = cell_ids['all'][cj]

    cell_stim_names = np.stack((post_synaptic[chosen_mask], pre_synaptic[chosen_mask]))

    # get rid of nans
    # these should all be the same, but for safety and clarity check for nans in all
    nan_loc = np.isnan(data_irms) | np.isnan(model_irms) | np.isnan(model_dirms)

    data_irfs = data_irfs[:, ~nan_loc]
    data_irfs_sem = data_irfs_sem[:, ~nan_loc]
    model_irfs = model_irfs[:, ~nan_loc]
    model_dirfs = model_dirfs[:, ~nan_loc]
    model_rdirfs = model_rdirfs[:, ~nan_loc]
    model_eirfs = model_eirfs[:, ~nan_loc]

    model_irms = model_irms[~nan_loc]
    model_dirms = model_dirms[~nan_loc]
    cell_ids_no_nan = np.stack((cell_stim_names[0, ~nan_loc], cell_stim_names[1, ~nan_loc]))

    selected_irfs = {'data_irfs': data_irfs,
                     'data_irfs_sem': data_irfs_sem,
                     'model_irfs': model_irfs,
                     'model_dirfs': model_dirfs,
                     'model_rdirfs': model_rdirfs,
                     'model_eirfs': model_eirfs,
                     'model_irms': model_irms,
                     'model_dirms': model_dirms,
                     'cell_ids': cell_ids_no_nan,
                     }

    return selected_irfs


def get_impulse_response_functions(data, inputs, sample_rate=0.5, window=(15, 30), sub_pre_stim=True):
    # get IRFs from data
    if window[0] < 0 or window[1] < 0 or np.sum(window) <= 0:
        raise Exception('window must be positive and sum to > 0')

    window = (np.array(window) / sample_rate).astype(int)

    num_neurons = data[0].shape[1]

    responses = []
    for n in range(num_neurons):
        responses.append([])

    # loop through data and inputs to find when the inputs are 1
    for e, i in zip(data, inputs):
        num_time = e.shape[0]
        stim_events = np.where(i == 1)

        for time, target in zip(stim_events[0], stim_events[1]):
            if time - window[0] >= 0 and window[1] + time < num_time:
                this_clip = e[time-window[0]:time+window[1], :]

                if sub_pre_stim:
                    baseline = np.nanmean(this_clip[:window[0], :], axis=0)
                    this_clip = this_clip - baseline

                responses[target].append(this_clip)

    for ri, r in enumerate(responses):
        if len(r) > 0:
            responses[ri] = np.stack(r)
        else:
            responses[ri] = np.zeros((0, np.sum(window), num_neurons))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ave_responses = [np.nanmean(j, axis=0) for j in responses]
    ave_responses = np.stack(ave_responses)
    ave_responses = np.transpose(ave_responses, axes=(1, 2, 0))

    ave_responses_sem = [np.nanstd(j, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(j), axis=0)) for j in responses]
    ave_responses_sem = np.stack(ave_responses_sem)
    ave_responses_sem = np.transpose(ave_responses_sem, axes=(1, 2, 0))

    return ave_responses, ave_responses_sem, responses


def calculate_irfs(model, rng=np.random.default_rng(), window=(15, 30)):
    # get irfs from Lgssm model
    num_t = int(window[1] / model.sample_rate)
    num_n = model.dynamics_dim
    irfs = np.zeros((num_t, num_n, num_n))

    for s in range(model.dynamics_dim):
        inputs = np.zeros((num_t, num_n))
        inputs[0, s] = 1
        irfs[:, :, s] = model.sample(num_time=num_t, inputs_list=[inputs], rng=rng, add_noise=False)['emissions'][0]
        print(s + 1, '/', num_n)

    zero_pad = np.zeros((int(window[0] / model.sample_rate), num_n, num_n))
    irfs = np.concatenate((zero_pad, irfs), axis=0)

    return irfs


def calculate_dirfs(model, rng=np.random.default_rng(), window=(15, 30), add_recipricol=False):
    # get dirfs from Lgssm model
    num_t = int(window[1] / model.sample_rate)
    num_n = model.dynamics_dim
    dirfs = np.empty((num_t, num_n, num_n))
    dirfs[:] = np.nan
    num_in_circuit = 2
    inputs = np.zeros((num_t, num_in_circuit))
    inputs[0, 0] = 1

    for s in range(model.dynamics_dim):
        for r in range(model.dynamics_dim):
            if s == r:
                continue

            sub_model = get_sub_model(model, s, r, add_recipricol=add_recipricol)
            dirfs[:, r, s] = sub_model.sample(num_time=num_t, inputs_list=[inputs], rng=rng, add_noise=False)['emissions'][0][:, 1]

        print(s + 1, '/', num_n)

    zero_pad = np.zeros((int(window[0] / model.sample_rate), num_n, num_n))
    dirfs = np.concatenate((zero_pad, dirfs), axis=0)

    return dirfs


def calculate_eirfs(model, rng=np.random.default_rng(), window=(30, 60)):
    # get eirfs from Lgssm model
    num_t = int(window[1] / model.sample_rate)
    num_n = model.dynamics_dim
    eirfs = np.empty((num_t, num_n, num_n))
    eirfs[:] = np.nan
    num_in_circuit = 2
    init_mean = np.zeros(num_in_circuit * model.dynamics_lags)
    init_mean[0] = 1
    inputs = np.zeros((num_t, num_in_circuit))

    for s in range(model.dynamics_dim):
        for r in range(model.dynamics_dim):
            if s == r:
                continue

            sub_model = get_sub_model(model, s, r)
            eirfs[:, r, s] = sub_model.sample(num_time=num_t, init_mean=[init_mean], inputs_list=[inputs],
                                              rng=rng, add_noise=False)['latents'][0][:, 1]

        print(s + 1, '/', num_n)

    zero_pad = np.zeros((int(window[0] / model.sample_rate), num_n, num_n))
    eirfs = np.concatenate((zero_pad, eirfs), axis=0)

    return eirfs


def get_sub_model(model, s, r, add_recipricol=False):
    # get a subset of model that includes only the stimulated neuron and the responding neuron
    num_in_circuit = 2
    # set up a new model that just has the dynamics of the neurons involved
    sub_model = Lgssm(num_in_circuit, num_in_circuit, num_in_circuit,
                      dynamics_lags=model.dynamics_lags, dynamics_input_lags=model.dynamics_input_lags,
                      emissions_input_lags=model.emissions_input_lags)

    dynamics_inds_s = np.arange(s, model.dynamics_dim_full, model.dynamics_dim)
    dynamics_inds_r = np.arange(r, model.dynamics_dim_full, model.dynamics_dim)
    dynamics_inputs_inds_s = np.arange(s, model.dynamics_input_dim_full, model.input_dim)
    dynamics_inputs_inds_r = np.arange(r, model.dynamics_input_dim_full, model.input_dim)
    emissions_inputs_inds_s = np.arange(s, model.emissions_input_dim_full, model.input_dim)
    emissions_inputs_inds_r = np.arange(r, model.emissions_input_dim_full, model.input_dim)

    dynamics_weights_inds = np.ix_((s, r), au.interleave(dynamics_inds_s, dynamics_inds_r))
    dynamics_input_weights_inds = np.ix_((s, r), au.interleave(dynamics_inputs_inds_s, dynamics_inputs_inds_r))
    cov_inds = np.ix_((s, r), (s, r))
    emissions_weights_inds = np.ix_((s, r), au.interleave(dynamics_inds_s, dynamics_inds_r))
    emissions_input_weights_inds = np.ix_((s, r), au.interleave(emissions_inputs_inds_s, emissions_inputs_inds_r))

    # get the chosen neurons. Then stack them so they can be padded for the delay embedding
    sub_model.dynamics_weights_init = model.dynamics_weights[dynamics_weights_inds]
    sub_model.dynamics_input_weights_init = model.dynamics_input_weights[dynamics_input_weights_inds]
    sub_model.dynamics_cov_init = model.dynamics_cov[cov_inds]

    sub_model.emissions_weights_init = model.emissions_weights[emissions_weights_inds]
    sub_model.emissions_input_weights_init = model.emissions_input_weights[emissions_input_weights_inds]
    sub_model.emissions_cov_init = model.emissions_cov[cov_inds]

    sub_model.dynamics_weights_init = au.stack_weights(sub_model.dynamics_weights_init, model.dynamics_lags, axis=1)
    sub_model.dynamics_input_weights_init = au.stack_weights(sub_model.dynamics_input_weights_init,
                                                             model.dynamics_input_lags, axis=1)
    sub_model.emissions_input_weights_init = au.stack_weights(sub_model.emissions_input_weights_init,
                                                              model.emissions_input_lags, axis=1)

    # set the backward weight from postsynaptic neuron to presynaptic to 0
    if not add_recipricol:
        sub_model.dynamics_weights_init[:, 0, 1] = 0

    sub_model.pad_init_for_lags()
    sub_model.set_to_init()

    return sub_model


def predict_model_cov(model, num_iter=100):
    # model correlation
    model_corr = model.dynamics_weights @ model.dynamics_weights.T + model.dynamics_cov
    for i in range(num_iter):
        model_corr = model.dynamics_weights @ model_corr @ model.dynamics_weights.T + model.dynamics_cov
    model_corr = model_corr[:model.dynamics_dim, :model.dynamics_dim]

    neuron_std = np.sqrt(model_corr.diagonal())
    neuron_std_out = neuron_std[:, None] * neuron_std[None, :]
    model_corr /= neuron_std_out

    return model_corr

