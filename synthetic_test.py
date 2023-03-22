import torch
from ssm_classes import LgssmSimple
import preprocessing as pp
import plotting
import pickle

profile_code = False
params = pp.get_params(param_name='params_synth')

device = params['device']
dtype = getattr(torch, params['dtype'])
random_seed = params['random_seed']

# initialize an linear gaussian ssm model
model_synth_true = LgssmSimple(params['latent_dim'], dtype=dtype, device=device)
# randomize the parameters (defaults are nonrandom)
model_synth_true.randomize_weights(random_seed=random_seed)
# sample from the randomized model
synth_data_dict = \
    model_synth_true.sample(num_time=params['num_time'],
                            num_data_sets=params['num_data_sets'],
                            nan_freq=params['nan_freq'])

emissions = synth_data_dict['emissions']
inputs = synth_data_dict['inputs']
init_mean_true = synth_data_dict['init_mean']
init_cov_true = synth_data_dict['init_cov']

# make a new model to fit to the random model
model_synth_trained = LgssmSimple(params['latent_dim'], dtype=dtype, device=device, verbose=params['verbose'])
model_synth_trained.randomize_weights()

if params['fit_type'] == 'gradient_descent':
    if profile_code:
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True
        ) as p:
            model_synth_trained.fit_gd(emissions, inputs, learning_rate=params['learning_rate'],
                                       num_steps=params['num_grad_steps'])

        print(p.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        print(p.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(p.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
    else:
        model_synth_trained.fit_gd(emissions, inputs, learning_rate=params['learning_rate'],
                                   num_steps=params['num_grad_steps'])

elif params['fit_type'] == 'batch_sgd':
    model_synth_trained.fit_batch_sgd(emissions, inputs, learning_rate=params['learning_rate'],
                                      num_steps=params['num_grad_steps'], batch_size=params['batch_size'],
                                      num_splits=params['num_splits'])
else:
    raise ValueError('Fit type not recognized')

# save the model and data
model_synth_true.save(path=params['model_save_folder'] + '/model_synth_true.pkl')
model_synth_trained.save(path=params['model_save_folder'] + '/model_synth_trained.pkl')

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

synth_data_dict['model'] = model_synth_true
synth_data_dict['ll_true_params'] = ll_true_params

save_file = open(params['synth_data_save_folder'] + '/synth_data.pkl', 'wb')
pickle.dump(synth_data_dict, save_file)
save_file.close()

# plotting
if params['plot_figures']:
    plotting.trained_on_synthetic(model_synth_trained, model_synth_true, ll_true_params)

