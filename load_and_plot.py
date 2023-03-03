import sys
import plotting
import pickle
import torch

model_trained_path = sys.argv[1]

model_trained_file = open(model_trained_path, 'rb')
model_trained = pickle.load(model_trained_file)
model_trained_file.close()

if len(sys.argv) == 3:
    dtype = model_trained.dtype
    device = model_trained.device

    synth_data_dict_path = sys.argv[2]

    synth_data_dict_file = open(synth_data_dict_path, 'rb')
    synth_data_dict = pickle.load(synth_data_dict_file)
    synth_data_dict_file.close()

    model_synth_true = synth_data_dict['model']
    model_synth_true.set_device(model_trained.device)

    init_mean_true = synth_data_dict['init_mean']
    init_cov_true = synth_data_dict['init_cov']
    emissions = synth_data_dict['emissions']
    inputs = synth_data_dict['inputs']

    # get the negative log-likelihood of the data given the true parameters
    init_mean_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :] for i in init_mean_true]
    init_cov_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :, :] for i in init_cov_true]
    emissions_torch = [torch.tensor(i, dtype=dtype, device=device) for i in emissions]
    inputs_torch = [torch.tensor(i, dtype=dtype, device=device) for i in inputs]

    init_mean_true_torch = torch.cat(init_mean_true_torch, dim=0)
    init_cov_true_torch = torch.cat(init_cov_true_torch, dim=0)
    emissions_torch, inputs_torch = model_synth_true.stack_data(emissions_torch, inputs_torch)

    ll_true_params = model_synth_true.loss_fn(emissions_torch, inputs_torch, init_mean_true_torch,
                                              init_cov_true_torch).detach().cpu().numpy()

    plotting.trained_on_synthetic(model_trained, model_synth_true, ll_true_params)

else:
    plotting.trained_on_real(model_trained)

