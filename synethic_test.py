import numpy as np
import torch
from matplotlib import pyplot as plt
from ssm_classes import LgssmSimple
import yaml

with open('run_params_synth.yaml', 'r') as file:
    run_params = yaml.safe_load(file)

device = run_params['device']
dtype = getattr(torch, run_params['dtype'])

# initialize an linear gaussian ssm model
lgssm_synthetic = LgssmSimple(run_params['latent_dim'], dtype=dtype, device=device, random_seed=run_params['random_seed'])
# randomize the parameters (defaults are nonrandom)
lgssm_synthetic.randomize_weights()
# sample from the randomized model
emissions, inputs, latents_true, init_mean_true, init_cov_true = \
    lgssm_synthetic.sample(num_time=run_params['num_time'],
                           num_data_sets=run_params['num_data_sets'],
                           nan_freq=run_params['nan_freq'])

# get the negative log-likelihood of the data given the true parameters
init_mean_true_torch = [torch.tensor(i, dtype=dtype, device=device) for i in init_mean_true]
init_cov_true_torch = [torch.tensor(i, dtype=dtype, device=device) for i in init_cov_true]
emissions_torch = [torch.tensor(i, dtype=dtype, device=device) for i in emissions]
inputs_torch = [torch.tensor(i, dtype=dtype, device=device) for i in inputs]
ll_true_params = lgssm_synthetic.loss_fn(emissions_torch, inputs_torch, init_mean_true_torch, init_cov_true_torch).detach().cpu().numpy()

# make a new model to fit to the random model
lgssm_model = LgssmSimple(run_params['latent_dim'], dtype=dtype, device=device, verbose=True)

if run_params['fit_type'] == 'gradient_descent':
    loss_out = lgssm_model.fit_gd(emissions, inputs, learning_rate=run_params['learning_rate'],
                                  num_steps=run_params['num_grad_steps'])
elif run_params['fit_type'] == 'batch_sgd':
    loss_out = lgssm_model.fit_batch_sgd(emissions, inputs, learning_rate=run_params['learning_rate'],
                                         num_steps=run_params['num_grad_steps'], batch_size=run_params['batch_size'],
                                         num_splits=run_params['num_splits'])
else:
    raise ValueError('Fit type not recognized')

if run_params['save_model']:
    lgssm_model.save(path=run_params['save_path'])

if run_params['plot_figures']:
    # Plots
    # Plot the loss
    plt.figure()
    plt.plot(loss_out)
    plt.plot(np.zeros(len(loss_out)) + ll_true_params, 'k-')
    plt.xlabel('iterations')
    plt.ylabel('negative log likelihood')
    plt.legend({'ll over fitting', 'll for the true parameters'})
    plt.tight_layout()

    # Plot the dynamics weights
    plt.figure()
    colorbar_shrink = 0.4
    plt.subplot(2, 2, 1)
    plt.imshow(lgssm_model.dynamics_weights_init, interpolation='Nearest')
    plt.title('init dynamics weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')

    plt.subplot(2, 2, 2)
    plt.imshow(lgssm_model.dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
    plt.title('fit dynamics weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')

    plt.subplot(2, 2, 3)
    plt.imshow(lgssm_synthetic.dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
    plt.title('true dynamics weights')
    plt.colorbar()

    plt.tight_layout()


    # Plot the input weights
    plt.figure()
    plt.plot(np.exp(lgssm_model.inputs_weights_log_init))
    plt.plot(np.exp(lgssm_model.inputs_weights_log.detach().cpu().numpy()))
    plt.plot(np.exp(lgssm_synthetic.inputs_weights_log.detach().cpu().numpy()))
    plt.legend(['init', 'final', 'true'])
    plt.xlabel('neurons')
    plt.ylabel('input weights')

    # plot the covariances
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.exp(lgssm_model.dynamics_cov_log_init))
    plt.plot(np.exp(lgssm_model.dynamics_cov_log.detach().cpu().numpy()))
    plt.plot(np.exp(lgssm_synthetic.dynamics_cov_log.detach().cpu().numpy()))
    plt.legend(['init', 'final', 'true'])
    plt.xlabel('neurons')
    plt.ylabel('dynamics noise cov')

    plt.subplot(1, 2, 2)
    plt.plot(np.exp(lgssm_model.emissions_cov_log_init))
    plt.plot(np.exp(lgssm_model.emissions_cov_log.detach().cpu().numpy()))
    plt.plot(np.exp(lgssm_synthetic.emissions_cov_log.detach().cpu().numpy()))
    plt.xlabel('neurons')
    plt.ylabel('emissions noise cov')

    plt.tight_layout()

    plt.show()

