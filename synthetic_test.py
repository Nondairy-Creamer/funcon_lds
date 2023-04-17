import torch
from ssm_classes import Lgssm
import preprocessing as pp
import plotting
import pickle
import numpy as np

run_params = pp.get_params(param_name='params_synth')

device = run_params['device']
dtype = getattr(torch, run_params['dtype'])
rng = np.random.default_rng(run_params['random_seed'])

model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                   dtype=dtype, device=device, param_props=run_params['param_props'])
# randomize the parameters (defaults are nonrandom)
model_true.randomize_weights(rng=rng)
model_true.emissions_weights = torch.eye(model_true.dynamics_dim, device=device, dtype=dtype)
model_true.emissions_input_weights = torch.zeros((model_true.dynamics_dim, model_true.emissions_dim), device=device, dtype=dtype)

# sample from the randomized model
data_dict = \
    model_true.sample(init_mean=np.zeros((run_params['num_data_sets'], model_true.dynamics_dim)),
                      num_time=run_params['num_time'],
                      num_data_sets=run_params['num_data_sets'],
                      nan_freq=run_params['nan_freq'],
                      rng=rng)

emissions = data_dict['emissions']
inputs = data_dict['inputs']
latents_true = data_dict['latents']
init_mean_true = data_dict['init_mean']
init_cov_true = data_dict['init_cov']

# make a new model to fit to the random model
model_trained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                      dtype=dtype, device=device, verbose=run_params['verbose'], param_props=run_params['param_props'])

model_trained.emissions_weights = torch.eye(model_trained.dynamics_dim, device=model_trained.device, dtype=model_trained.dtype)
model_trained.emissions_input_weights = torch.zeros((model_trained.dynamics_dim, model_trained.emissions_dim), device=device, dtype=dtype)

# save the data
data_dict['params_init'] = model_trained.get_params()
data_dict['params_init']['init_mean'] = init_mean_true
data_dict['params_init']['init_cov'] = init_cov_true
data_dict['params_true'] = model_true.get_params()
data_dict['params_true']['init_mean'] = init_mean_true
data_dict['params_true']['init_cov'] = init_cov_true

save_file = open(run_params['synth_data_save_folder'] + '/data.pkl', 'wb')
pickle.dump(data_dict, save_file)
save_file.close()

# train the model
if run_params['fit_type'] == 'gd':
    model_trained.fit_gd(emissions, inputs, init_mean=init_mean_true, init_cov=init_cov_true,
                         learning_rate=run_params['learning_rate'],
                         num_steps=run_params['num_grad_steps'])

elif run_params['fit_type'] == 'em':
    model_trained.fit_em(emissions, inputs, init_mean=init_mean_true, init_cov=init_cov_true,
                         num_steps=run_params['num_grad_steps'])

elif run_params['fit_type'] == 'none':
    # do nothing
    a=1

else:
    raise ValueError('Fit type not recognized')

# save the model
model_true.save(path=run_params['model_save_folder'] + '/model_true.pkl')
model_trained.save(path=run_params['model_save_folder'] + '/model_trained.pkl')

# get the log-likelihood of the data given the true parameters
# convert the emissions, inputs, init_mean, and init_cov into tensors
emissions_torch, inputs_torch = model_true.standardize_inputs(emissions, inputs)

init_mean_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :] for i in init_mean_true]
init_cov_true_torch = [torch.tensor(i, dtype=dtype, device=device)[None, :, :] for i in init_cov_true]
init_mean_true_torch = torch.cat(init_mean_true_torch, dim=0)
init_cov_true_torch = torch.cat(init_cov_true_torch, dim=0)

ll_true_params = model_true.lgssm_filter(emissions_torch, inputs_torch, init_mean_true_torch,
                                         init_cov_true_torch)[0].detach().cpu().numpy()

# plotting
if run_params['plot_figures']:
    plotting.plot_model_params(model_trained, model_true, ll_true_params)

