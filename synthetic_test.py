import torch
from ssm_classes import Lgssm
import preprocessing as pp
import plotting
import pickle
import numpy as np
import inference as infer

params = pp.get_params(param_name='params_synth')

if params['model_type'] == 'full':
    model_class = Lgssm
else:
    raise ValueError('model_type not recognized')

device = params['device']
dtype = getattr(torch, params['dtype'])
rng = np.random.default_rng(params['random_seed'])

# initialize an linear gaussian ssm model
model_synth_true = model_class(params['dynamics_dim'], params['emissions_dim'], params['input_dim'],
                               dtype=dtype, device=device)
# randomize the parameters (defaults are nonrandom)
model_synth_true.randomize_weights(rng=rng)
# sample from the randomized model
synth_data_dict = \
    model_synth_true.sample(num_time=params['num_time'],
                            num_data_sets=params['num_data_sets'],
                            nan_freq=params['nan_freq'])

emissions = synth_data_dict['emissions']
inputs = synth_data_dict['inputs']
latents_true = synth_data_dict['latents']
init_mean_true = synth_data_dict['init_mean']
init_cov_true = synth_data_dict['init_cov']

# make a new model to fit to the random model
model_synth_trained = model_class(params['dynamics_dim'], params['emissions_dim'], params['input_dim'],
                                  dtype=dtype, device=device, verbose=params['verbose'])

if params['fit_type'] == 'gd':
    model_synth_trained.fit_gd(emissions, inputs, learning_rate=params['learning_rate'],
                               num_steps=params['num_grad_steps'])

elif params['fit_type'] == 'em':
    model_synth_trained.fit_em(emissions, inputs, num_steps=params['num_grad_steps'])

elif params['fit_type'] == 'none':
    # do nothing
    a=1

else:
    raise ValueError('Fit type not recognized')

# save the model and data
model_synth_true.save(path=params['model_save_folder'] + '/model_synth_true.pkl')
model_synth_trained.save(path=params['model_save_folder'] + '/model_synth_trained.pkl')

# convert the emissions, inputs, init_mean, and init_cov into tensors
emissions_torch, inputs_torch = model_synth_true.standardize_inputs(emissions, inputs)

init_mean_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :] for i in init_mean_true]
init_cov_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :, :] for i in init_cov_true]
init_mean_true_torch = torch.cat(init_mean_true_torch, dim=0)
init_cov_true_torch = torch.cat(init_cov_true_torch, dim=0)

# get the negative log-likelihood of the data given the true parameters
ll_true_params = model_synth_true.lgssm_filter(emissions_torch, inputs_torch, init_mean_true_torch,
                                               init_cov_true_torch)[0].detach().cpu().numpy()

save_file = open(params['synth_data_save_folder'] + '/synth_data.pkl', 'wb')
pickle.dump(synth_data_dict, save_file)
save_file.close()

# plotting
if params['plot_figures']:
    if params['model_type'] == 'simple':
        plotting.trained_on_synthetic_diag(model_synth_trained, model_synth_true, ll_true_params)
    elif params['model_type'] == 'full':
        plotting.trained_on_synthetic(model_synth_trained, model_synth_true, ll_true_params)
    else:
        raise ValueError('model_type not recognized')

