import torch
from ssm_classes import Lgssm
import preprocessing as pp
import plotting
import pickle
import numpy as np

run_params = pp.get_params(param_name='params_synth')

if run_params['model_type'] == 'full':
    model_class = Lgssm
else:
    raise ValueError('model_type not recognized')

device = run_params['device']
dtype = getattr(torch, run_params['dtype'])
rng = np.random.default_rng(run_params['random_seed'])

# initialize an linear gaussian ssm model
model_synth_true = model_class(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                               dtype=dtype, device=device)
# randomize the parameters (defaults are nonrandom)
model_synth_true.randomize_weights(rng=rng)
# sample from the randomized model
synth_data_dict = \
    model_synth_true.sample(num_time=run_params['num_time'],
                            num_data_sets=run_params['num_data_sets'],
                            nan_freq=run_params['nan_freq'],
                            rng=rng)

emissions = synth_data_dict['emissions']
inputs = synth_data_dict['inputs']
latents_true = synth_data_dict['latents']
init_mean_true = synth_data_dict['init_mean']
init_cov_true = synth_data_dict['init_cov']

# make a new model to fit to the random model
model_synth_trained = model_class(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                                  dtype=dtype, device=device, verbose=run_params['verbose'])

# save the data
synth_data_dict['params_init'] = model_synth_trained.get_params()
synth_data_dict['params_init']['init_mean'] = init_mean_true
synth_data_dict['params_init']['init_cov'] = init_mean_true
synth_data_dict['params_true'] = model_synth_true.get_params()
synth_data_dict['params_true']['init_mean'] = init_mean_true
synth_data_dict['params_true']['init_cov'] = init_mean_true

save_file = open(run_params['synth_data_save_folder'] + '/synth_data.pkl', 'wb')
pickle.dump(synth_data_dict, save_file)
save_file.close()

if run_params['fit_type'] == 'gd':
    model_synth_trained.fit_gd(emissions, inputs, learning_rate=run_params['learning_rate'],
                               num_steps=run_params['num_grad_steps'])

elif run_params['fit_type'] == 'em':
    model_synth_trained.fit_em(emissions, inputs, init_cov=init_cov_true, num_steps=run_params['num_grad_steps'])

elif run_params['fit_type'] == 'none':
    # do nothing
    a=1

else:
    raise ValueError('Fit type not recognized')

# save the model and data
model_synth_true.save(path=run_params['model_save_folder'] + '/model_synth_true.pkl')
model_synth_trained.save(path=run_params['model_save_folder'] + '/model_synth_trained.pkl')

# convert the emissions, inputs, init_mean, and init_cov into tensors
emissions_torch, inputs_torch = model_synth_true.standardize_inputs(emissions, inputs)

init_mean_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :] for i in init_mean_true]
init_cov_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :, :] for i in init_cov_true]
init_mean_true_torch = torch.cat(init_mean_true_torch, dim=0)
init_cov_true_torch = torch.cat(init_cov_true_torch, dim=0)

# get the negative log-likelihood of the data given the true parameters
ll_true_params = model_synth_true.lgssm_filter(emissions_torch, inputs_torch, init_mean_true_torch,
                                               init_cov_true_torch)[0].detach().cpu().numpy()

# plotting
if run_params['plot_figures']:
    if run_params['model_type'] == 'simple':
        plotting.trained_on_synthetic_diag(model_synth_trained, model_synth_true, ll_true_params)
    elif run_params['model_type'] == 'full':
        plotting.trained_on_synthetic(model_synth_trained, model_synth_true, ll_true_params)
    else:
        raise ValueError('model_type not recognized')

