import numpy as np
import analysis_utilities as au
import warnings
import copy
import time
import csv
import torch
from torch.autograd.functional import hessian
from torch.autograd.functional import hvp


def mask_weights_to_nan(weights, irm_mask, corr_mask, combine_masks=False):
    weights = copy.deepcopy(weights)
    if combine_masks:
        irm_mask = irm_mask | corr_mask
        corr_mask = irm_mask

    # set all the weights to nan with the nan mask
    for i in weights:
        for j in weights[i]:
            if isinstance(weights[i][j], dict):
                for k in weights[i][j]:
                    if 'corr' in k:
                        mask = corr_mask
                    else:
                        mask = irm_mask

                    if weights[i][j][k].ndim == 2:
                        weights[i][j][k][mask] = np.nan
                    elif weights[i][j][k].ndim == 3:
                        weights[i][j][k][:, mask] = np.nan
                    else:
                        raise Exception('Weights shape not recognized')

            else:
                if 'corr' in j:
                    mask = corr_mask
                else:
                    mask = irm_mask

                if weights[i][j].ndim == 2:
                    weights[i][j][mask] = np.nan
                elif weights[i][j].ndim == 3:
                    weights[i][j][:, mask] = np.nan
                else:
                    raise Exception('Weights shape not recognized')

    return weights


def remove_nan_irfs(weights, cell_ids, data_type='test', chosen_mask=None):
    if chosen_mask is None:
        chosen_mask = np.zeros_like(weights['data'][data_type]['irms']) == 0

    data_irfs = weights['data'][data_type]['irfs'][:, chosen_mask]
    data_irfs_sem = weights['data'][data_type]['irfs_sem'][:, chosen_mask]
    model_irfs = weights['models']['synap']['irfs'][:, chosen_mask]
    model_dirfs = weights['models']['synap']['dirfs'][:, chosen_mask]
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
    model_eirfs = model_eirfs[:, ~nan_loc]

    model_irms = model_irms[~nan_loc]
    model_dirms = model_dirms[~nan_loc]
    cell_ids_no_nan = np.stack((cell_stim_names[0, ~nan_loc], cell_stim_names[1, ~nan_loc])).T

    selected_irfs = {'data_irfs': data_irfs,
                     'data_irfs_sem': data_irfs_sem,
                     'model_irfs': model_irfs,
                     'model_dirfs': model_dirfs,
                     'model_eirfs': model_eirfs,
                     'model_irms': model_irms,
                     'model_dirms': model_dirms,
                     'cell_ids': cell_ids_no_nan,
                     }

    return selected_irfs


def get_silenced_model(model_original, neurons_to_silence):
    if type(neurons_to_silence) is not list:
        neurons_to_silence = [neurons_to_silence]
    model_silenced = copy.deepcopy(model_original)

    # silence the neurons
    for ns in neurons_to_silence:
        ns_ind = model_silenced.cell_ids.index(ns)
        silence_inds = np.arange(ns_ind, model_silenced.dynamics_dim_full, model_silenced.dynamics_dim)

        y_vals = np.arange(model_silenced.dynamics_dim)
        y_vals = np.delete(y_vals, ns_ind)
        model_silenced.dynamics_weights[np.ix_(y_vals, silence_inds)] = 0

    return model_silenced


def get_impulse_response_functions(data, inputs, sample_rate=2, window=(15, 30), sub_pre_stim=True):
    # get IRFs from data
    if window[0] < 0 or window[1] < 0 or np.sum(window) <= 0:
        raise Exception('window must be positive and sum to > 0')

    window = (np.array(window) * sample_rate).astype(int)

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


def calculate_irfs(model, rng=np.random.default_rng(), window=(15, 30), verbose=False):
    # get irfs from Lgssm model
    num_t = int(window[1] * model.sample_rate)
    num_n = model.dynamics_dim
    irfs = np.zeros((num_t, num_n, num_n))

    for s in range(model.dynamics_dim):
        inputs = np.zeros((num_t, num_n))
        inputs[0, s] = 1
        irfs[:, :, s] = model.sample(num_time=num_t, inputs=inputs, rng=rng, add_noise=False)['emissions']

        if verbose:
            print(s + 1, '/', num_n)

    zero_pad = np.zeros((int(window[0] * model.sample_rate), num_n, num_n))
    irfs = np.concatenate((zero_pad, irfs), axis=0)

    return irfs


def calculate_irms(model, rng=np.random.default_rng(), window=(15, 30), verbose=False):
    irfs = calculate_irfs(model, rng=rng, window=window, verbose=verbose)
    irms = np.sum(irfs[window[0]:, :, :], axis=0) / model.sample_rate

    return irms


def calculate_iirfs(model, rng=np.random.default_rng(), window=(15, 30)):
    # get iirfs from Lgssm model
    num_t = int(window[1] * model.sample_rate)
    num_n = model.dynamics_dim
    iirfs = np.empty((num_t, num_n, num_n))
    iirfs[:] = np.nan

    for s in range(model.dynamics_dim):
        for r in range(model.dynamics_dim):
            if s == r:
                continue
            inputs = np.zeros((num_t, num_n))
            inputs[0, s] = 1

            sub_model = get_indirect_model(model, s, r)
            iirfs[:, r, s] = sub_model.sample(num_time=num_t, inputs=inputs, rng=rng, add_noise=False)['emissions'][:, r]

        print(s + 1, '/', num_n)

    zero_pad = np.zeros((int(window[0] * model.sample_rate), num_n, num_n))
    iirfs = np.concatenate((zero_pad, iirfs), axis=0)

    return iirfs


def calculate_dirfs(model, rng=np.random.default_rng(), window=(15, 30), add_recipricol=False):
    # get dirfs from Lgssm model
    num_t = int(window[1] * model.sample_rate)
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
            dirfs[:, r, s] = sub_model.sample(num_time=num_t, inputs=inputs, rng=rng, add_noise=False)['emissions'][:, 1]

        print(s + 1, '/', num_n)

    zero_pad = np.zeros((int(window[0] * model.sample_rate), num_n, num_n))
    dirfs = np.concatenate((zero_pad, dirfs), axis=0)

    return dirfs


def calculate_eirfs(model, rng=np.random.default_rng(), window=(15, 30), verbose=False):
    # get eirfs from Lgssm model
    num_t = int(window[1] * model.sample_rate)
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
            eirfs[:, r, s] = sub_model.sample(num_time=num_t, init_mean=init_mean, inputs=inputs,
                                              rng=rng, add_noise=False)['emissions'][:, 1]

        if verbose:
            print(s + 1, '/', num_n)

    zero_pad = np.zeros((int(window[0] * model.sample_rate), num_n, num_n))
    eirfs = np.concatenate((zero_pad, eirfs), axis=0)

    return eirfs


def calculate_eirms(model, rng=np.random.default_rng(), window=(15, 30), verbose=False):
    eirfs = calculate_eirfs(model, rng=rng, window=window, verbose=verbose)
    eirms = np.sum(eirfs[window[0]:, :, :], axis=0) / model.sample_rate

    return eirms


def get_sub_model(model_original, s, r, add_recipricol=False):
    # get a subset of model that includes only the stimulated neuron and the responding neuron
    model_new = copy.deepcopy(model_original)
    model_new.dynamics_dim = 2
    model_new.dynamics_input_dim = model_new.dynamics_dim
    model_new.dynamics_input_dim_full = model_new.dynamics_input_dim * model_new.dynamics_input_lags
    model_new.dynamics_dim_full = model_new.dynamics_lags * model_new.dynamics_dim
    model_new.emissions_dim = model_new.dynamics_dim
    model_new.emissions_input_dim = model_new.emissions_dim
    model_new.emissions_input_dim = model_new.emissions_input_dim * model_new.emissions_input_lags

    dynamics_inds_s = np.arange(s, model_original.dynamics_dim_full, model_original.dynamics_dim)
    dynamics_inds_r = np.arange(r, model_original.dynamics_dim_full, model_original.dynamics_dim)
    dynamics_inputs_inds_s = np.arange(s, model_original.dynamics_input_dim_full, model_original.input_dim)
    dynamics_inputs_inds_r = np.arange(r, model_original.dynamics_input_dim_full, model_original.input_dim)
    emissions_inputs_inds_s = np.arange(s, model_original.emissions_input_dim_full, model_original.input_dim)
    emissions_inputs_inds_r = np.arange(r, model_original.emissions_input_dim_full, model_original.input_dim)

    dynamics_weights_inds = np.ix_((s, r), au.interleave(dynamics_inds_s, dynamics_inds_r))
    dynamics_input_weights_inds = np.ix_((s, r), au.interleave(dynamics_inputs_inds_s, dynamics_inputs_inds_r))
    cov_inds = np.ix_((s, r), (s, r))
    emissions_weights_inds = np.ix_((s, r), au.interleave(dynamics_inds_s, dynamics_inds_r))
    emissions_input_weights_inds = np.ix_((s, r), au.interleave(emissions_inputs_inds_s, emissions_inputs_inds_r))

    # get the chosen neurons. Then stack them so they can be padded for the delay embedding
    model_new.dynamics_weights_init = model_new.dynamics_weights[dynamics_weights_inds]
    model_new.dynamics_input_weights_init = model_new.dynamics_input_weights[dynamics_input_weights_inds]
    model_new.dynamics_cov_init = model_new.dynamics_cov[cov_inds]

    model_new.emissions_weights_init = model_new.emissions_weights[emissions_weights_inds]
    model_new.emissions_input_weights_init = model_new.emissions_input_weights[emissions_input_weights_inds]
    model_new.emissions_cov_init = model_new.emissions_cov[cov_inds]

    model_new.dynamics_weights_init = au.stack_weights(model_new.dynamics_weights_init, model_new.dynamics_lags, axis=1)
    model_new.dynamics_input_weights_init = au.stack_weights(model_new.dynamics_input_weights_init,
                                                             model_new.dynamics_input_lags, axis=1)
    model_new.emissions_input_weights_init = au.stack_weights(model_new.emissions_input_weights_init,
                                                              model_new.emissions_input_lags, axis=1)

    # set the backward weight from postsynaptic neuron to presynaptic to 0
    if not add_recipricol:
        model_new.dynamics_weights_init[:, 0, 1] = 0

    model_new.pad_init_for_lags()
    model_new.set_to_init()

    return model_new


def get_indirect_model(model_original, s, r):
    # get a subset of model that includes only the stimulated neuron and the responding neuron
    model_new = copy.deepcopy(model_original)
    dynamics_inds_s = np.arange(s, model_original.dynamics_dim_full, model_original.dynamics_dim)
    model_new.dynamics_weights[r, dynamics_inds_s] = 0

    return model_new


def predict_model_corr_coef(model, num_iter=100):
    # model correlation
    model_corr = model.dynamics_weights @ model.dynamics_weights.T + model.dynamics_cov
    for i in range(num_iter):
        model_corr = model.dynamics_weights @ model_corr @ model.dynamics_weights.T + model.dynamics_cov
    model_corr = model_corr[:model.dynamics_dim, :model.dynamics_dim]

    neuron_std = np.sqrt(model_corr.diagonal())
    neuron_std_out = neuron_std[:, None] * neuron_std[None, :]
    model_corr /= neuron_std_out

    return model_corr


def approximate_hessian_diagonal(model, data):
    # this function will use finite differences to calculate the approximate 2nd order derivative
    # of the loss wrt each parameter in the dynamics matrix A

    cell_ids = model.cell_ids.copy()
    num_neurons = len(cell_ids)
    weights = model.dynamics_weights
    weights_mask = model.param_props['mask']['dynamics_weights']

    smallest_weight = np.min(np.abs(weights[weights_mask]))
    epsilon = 0.01 * smallest_weight
    num_weights = np.sum(weights_mask) - num_neurons  # number of off-diagonal weights
    num_data = len(data['emissions'])
    num_data = 1

    loss = 0
    for d in range(num_data):
        loss += model.lgssm_filter(data['emissions'][d], data['inputs'][d], data['emissions_offset'][d], data['init_mean'][d], data['init_cov'][d])[0]

    csv_output = [['presynaptic cell', 'postsynaptic cell', 'weight', 'standard_deviation']]

    counter = 0
    start = time.time()
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i == j:
                continue

            if weights_mask[j, i]:
                model_plus = copy.deepcopy(model)
                model_minus = copy.deepcopy(model)
                model_plus.dynamics_weights[j, i] += epsilon
                model_minus.dynamics_weights[j, i] -= epsilon

                loss_plus = 0
                loss_minus = 0
                for d in range(num_data):
                    loss_plus += model_plus.lgssm_filter(data['emissions'][d], data['inputs'][d], data['emissions_offset'][d], data['init_mean'][d], data['init_cov'][d])[0]
                    loss_minus += model_plus.lgssm_filter(data['emissions'][d], data['inputs'][d], data['emissions_offset'][d], data['init_mean'][d], data['init_cov'][d])[0]
                hessian = 1 / epsilon**2 * (loss_plus - 2 * loss + loss_minus)

                # save the output
                standard_deviation = np.sqrt(- 1 / hessian)
                csv_output.append([cell_ids[i], cell_ids[j], f"{weights[j, i]:.8f}", f"{standard_deviation:.8f}"])

                counter += 1
                time_elapsed = time.time() - start
                print(counter, '/', num_weights, 'finished')
                print(time.time() - start, 'elapsed')
                time_remaining = time_elapsed / counter * (num_weights - counter)
                print('Estimated', time_remaining, 's remaining')

    with open('model_weights.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)


def cg_solve(Hv_fn, b, tol=1e-5, maxiter=None):
    """Conjugate‑gradient to solve Hx = b for symmetric pos‑def H via Hv_fn."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = r.dot(r)
    maxiter = maxiter or (2*b.numel())
    for _ in range(maxiter):
        Hp = Hv_fn(x, p)  # note: Hv_fn(w, v) -> H(w) @ v
        alpha = rsold / (p.dot(Hp) + 1e-8)
        x = x + alpha*p
        r = r - alpha*Hp
        rsnew = r.dot(r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x


def diag_inv_hessian_estimate(loss_fn, flat_w, mask, K=20):
    """
    Estimates diag(H^{-1}) for H = ∇² loss_fn(flat_w)
    mask: boolean mask into the original full-weight tensor
    flat_w: the 1‑D vector of free params
    """
    P = flat_w.numel()
    est = torch.zeros(P, device=flat_w.device)
    for _ in range(K):
        # 1) sample v ∈ {+1, -1}^P
        v = torch.randint(0, 2, (P,), device=flat_w.device, dtype=torch.float32)*2 - 1

        # 2) define Hv_fn(w, v): compute H(w) @ v
        def Hv_fn(_, vec):
            _, hvps = hvp(loss_fn, (flat_w,), (vec,))
            return hvps[0]

        # 3) solve H x = v  →  x ≈ H^{-1}v
        x = cg_solve(Hv_fn, v, tol=1e-5)

        # 4) accumulate v * x
        est += v * x

    return est.div_(K)


def calc_hessian_torch(model, data):
    # this function will use finite differences to calculate the approximate 2nd order derivative
    # of the loss wrt each parameter in the dynamics matrix A
    cell_ids = model.cell_ids.copy()
    num_neurons = len(cell_ids)
    # model.dynamics_weights = torch.tensor(model.dynamics_weights, requires_grad=True)
    W = model.dynamics_weights.clone().detach().requires_grad_(True)
    weights_mask = torch.tensor(model.param_props['mask']['dynamics_weights'])
    flat_w = W[weights_mask]
    num_data = len(data['emissions'])

    def loss_fn(w):
        full_weights = W.clone()
        full_weights[weights_mask] = w
        model.dynamics_weights = full_weights

        loss = 0.0
        for d in range(num_data):
            loss += model.lgssm_filter_torch(data['emissions'][d], data['inputs'][d], data['emissions_offset'][d], data['init_mean'][d], data['init_cov'][d])[0]

        return loss

    # calculate the hessian here
    # diag_est = diag_inv_hessian_estimate(loss_fn, flat_w, weights_mask, K=10)

    start = time.time()
    H = hessian(loss_fn, flat_w)
    H_inv = torch.linalg.inv(-H)
    vars = torch.diag(H_inv)
    stds = torch.sqrt(vars)
    print(len(flat_w), 'parameters took', time.time() - start, 's')


    csv_output = [['presynaptic cell', 'postsynaptic cell', 'weight', 'standard_deviation']]
    count = 0
    for i in range(num_neurons):
        for j in range(num_neurons):
            if weights_mask[i, j]:
                csv_output.append([cell_ids[j], cell_ids[i], f"{W[i, j]:.8f}", f"{stds[count]:.8f}"])
                count += 1

    with open('model_weights.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)

