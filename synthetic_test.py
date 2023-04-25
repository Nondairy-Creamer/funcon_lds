import torch
from ssm_classes import Lgssm
import preprocessing as pp
import plotting
import pickle
import numpy as np
import time

run_params = pp.get_params(param_name='params_synth')

device = run_params['device']
dtype = getattr(torch, run_params['dtype'])
rng = np.random.default_rng(run_params['random_seed'])

model_true = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                   dtype=dtype, device=device, param_props=run_params['param_props'],
                   num_lags=run_params['num_lags'])
# randomize the parameters (defaults are nonrandom)
model_true.randomize_weights(rng=rng)
model_true.emissions_weights = torch.eye(model_true.emissions_dim, model_true.dynamics_dim_full, device=device, dtype=dtype)
model_true.emissions_input_weights = torch.zeros((model_true.emissions_dim, model_true.input_dim_full), device=device, dtype=dtype)

start = time.time()
# sample from the randomized model
data_dict = \
    model_true.sample(num_time=run_params['num_time'],
                      num_data_sets=run_params['num_data_sets'],
                      nan_freq=run_params['nan_freq'],
                      rng=rng)
print('Time to sample:', time.time() - start, 's')

# data_end = [500, 300, 1000, 900, 800]
# for d in range(run_params['num_data_sets']):
    # data_dict['emissions'][d][data_end[d]:, :] = np.nan
    # data_dict['emissions'][d][:, d:d+3] = np.nan

emissions = data_dict['emissions']
inputs = data_dict['inputs']
latents_true = data_dict['latents']
init_mean_true = data_dict['init_mean']
init_cov_true = data_dict['init_cov']

# make a new model to fit to the random model
model_trained = Lgssm(run_params['dynamics_dim'], run_params['emissions_dim'], run_params['input_dim'],
                      dtype=dtype, device=device, verbose=run_params['verbose'], param_props=run_params['param_props'],
                      num_lags=run_params['num_lags'])
# model_trained.randomize_weights()
model_true.emissions_weights = torch.eye(model_true.emissions_dim, model_true.dynamics_dim_full, device=device, dtype=dtype)
model_true.emissions_input_weights = torch.zeros((model_true.emissions_dim, model_true.input_dim_full), device=device, dtype=dtype)

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
# model_trained.fit_em(emissions, inputs, init_mean=init_mean_true, init_cov=init_cov_true,
#                      num_steps=run_params['num_grad_steps'])
model_trained.fit_em(emissions, inputs,
                     num_steps=run_params['num_grad_steps'])

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
ll_true_params = 0
ll_trained_params = 0

for d in range(len(emissions_torch)):
    emi_in = emissions_torch[d]
    inp_in = inputs_torch[d]
    im_in = init_mean_true_torch[d]
    ic_in = init_cov_true_torch[d]
    ll_true_params += model_true.lgssm_filter(emi_in, inp_in, im_in, ic_in)[0].detach().cpu().numpy()
    ll_trained_params += model_trained.lgssm_filter(emi_in, inp_in, im_in, ic_in)[0].detach().cpu().numpy()

# plotting
if run_params['plot_figures']:
    plotting.plot_model_params(model_trained, model_true, ll_true_params)

