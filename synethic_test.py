import numpy as np
import torch
from matplotlib import pyplot as plt
from ssm_classes import LgssmSimple

# number of iteration through the full data set to run during gradient descent
num_grad_steps = 50
batch_size = 2
num_splits = 1

device = 'cpu'
# dtype = torch.float32
dtype = torch.float64
fit_type = 'batch_sgd'

learning_rate = 1e-2
latent_dim = 5
obs_dim = latent_dim
num_time = 1000
num_data_sets = 10
random_seed = 0
nan_freq = 0.1

# initialize an linear gaussian ssm model
lgssm_synthetic = LgssmSimple(latent_dim, dtype=dtype, device=device, random_seed=random_seed)
# randomize the parameters (defaults are nonrandom)
lgssm_synthetic.randomize_weights()
# sample from the randomized model
emissions, inputs, latents_true, init_mean_true, init_cov_true = lgssm_synthetic.sample(num_time=num_time, num_data_sets=num_data_sets, nan_freq=nan_freq)
# get the negative log-likelihood of the data given the true parameters


init_mean_true_torch = [torch.tensor(i, dtype=dtype, device=device) for i in init_mean_true]
init_cov_true_torch = [torch.tensor(i, dtype=dtype, device=device) for i in init_cov_true]
emissions_torch = [torch.tensor(i, dtype=dtype, device=device) for i in emissions]
inputs_torch = [torch.tensor(i, dtype=dtype, device=device) for i in inputs]
ll_true_params = lgssm_synthetic.loss_fn(emissions_torch, inputs_torch, init_mean_true_torch, init_cov_true_torch).detach().cpu().numpy()

# make a new model to fit to the random model
lgssm_trained = LgssmSimple(latent_dim, dtype=dtype, device=device, verbose=True)

if fit_type == 'gd':
    loss_out = lgssm_trained.fit_gd(emissions, inputs, learning_rate=learning_rate,
                                    num_steps=num_grad_steps)
elif fit_type == 'batch_sgd':
    loss_out = lgssm_trained.fit_batch_sgd(emissions, inputs, learning_rate=learning_rate,
                                           num_steps=num_grad_steps, batch_size=batch_size,
                                           num_splits=num_splits)
else:
    raise ValueError('Fit type not recognized')

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
plt.imshow(lgssm_trained.dynamics_weights_init, interpolation='Nearest')
plt.title('init dynamics weights')
plt.xlabel('input neurons')
plt.ylabel('output neurons')

plt.subplot(2, 2, 2)
plt.imshow(lgssm_trained.dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
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
plt.plot(np.exp(lgssm_trained.inputs_weights_log_init))
plt.plot(np.exp(lgssm_trained.inputs_weights_log.detach().cpu().numpy()))
plt.plot(np.exp(lgssm_synthetic.inputs_weights_log.detach().cpu().numpy()))
plt.legend(['init', 'final', 'true'])
plt.xlabel('neurons')
plt.ylabel('input weights')

# plot the covariances
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.exp(lgssm_trained.dynamics_cov_log_init))
plt.plot(np.exp(lgssm_trained.dynamics_cov_log.detach().cpu().numpy()))
plt.plot(np.exp(lgssm_synthetic.dynamics_cov_log.detach().cpu().numpy()))
plt.legend(['init', 'final', 'true'])
plt.xlabel('neurons')
plt.ylabel('dynamics noise cov')

plt.subplot(1, 2, 2)
plt.plot(np.exp(lgssm_trained.emissions_cov_log_init))
plt.plot(np.exp(lgssm_trained.emissions_cov_log.detach().cpu().numpy()))
plt.plot(np.exp(lgssm_synthetic.emissions_cov_log.detach().cpu().numpy()))
plt.xlabel('neurons')
plt.ylabel('emissions noise cov')

plt.tight_layout()

plt.show()

