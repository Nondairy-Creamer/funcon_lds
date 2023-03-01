import torch
from ssm_classes import LgssmSimple
import preprocessing as pp
import plotting


params = pp.get_params(param_name='params_synth')

device = params['device']
dtype = getattr(torch, params['dtype'])

# initialize an linear gaussian ssm model
model_synth_true = LgssmSimple(params['latent_dim'], dtype=dtype, device=device, random_seed=params['random_seed'])
# randomize the parameters (defaults are nonrandom)
model_synth_true.randomize_weights()
# sample from the randomized model
emissions, inputs, latents_true, init_mean_true, init_cov_true = \
    model_synth_true.sample(num_time=params['num_time'],
                            num_data_sets=params['num_data_sets'],
                            nan_freq=params['nan_freq'])


# make a new model to fit to the random model
model_synth_trained = LgssmSimple(params['latent_dim'], dtype=dtype, device=device, verbose=True)

if params['fit_type'] == 'gradient_descent':
    model_synth_trained.fit_gd(emissions, inputs, learning_rate=params['learning_rate'],
                               num_steps=params['num_grad_steps'])
elif params['fit_type'] == 'batch_sgd':
    model_synth_trained.fit_batch_sgd(emissions, inputs, learning_rate=params['learning_rate'],
                                      num_steps=params['num_grad_steps'], batch_size=params['batch_size'],
                                      num_splits=params['num_splits'])
else:
    raise ValueError('Fit type not recognized')

if params['save_model']:
    model_synth_true.save(path=params['save_folder'] + '/model_synth_true.pkl')
    model_synth_trained.save(path=params['save_folder'] + '/model_synth_trained.pkl')

if params['plot_figures']:
    # get the negative log-likelihood of the data given the true parameters
    init_mean_true_torch = [torch.tensor(i, dtype=dtype, device=device) for i in init_mean_true]
    init_cov_true_torch = [torch.tensor(i, dtype=dtype, device=device) for i in init_cov_true]
    emissions_torch = [torch.tensor(i, dtype=dtype, device=device) for i in emissions]
    inputs_torch = [torch.tensor(i, dtype=dtype, device=device) for i in inputs]
    ll_true_params = model_synth_true.loss_fn(emissions_torch, inputs_torch, init_mean_true_torch,
                                              init_cov_true_torch).detach().cpu().numpy()

    plotting.trained_on_synthetic(model_synth_trained, model_synth_true, ll_true_params)

