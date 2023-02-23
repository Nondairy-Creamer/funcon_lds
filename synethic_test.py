import numpy as np
import torch
import preprocessing as pp
import inference as infer
from matplotlib import pyplot as plt
import time
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
# from torch.utils.tensorboard import SummaryWriter

# the goal of this function is to take the pairwise stimulation and response data from
# https://arxiv.org/abs/2208.04790
# this data is a collection of calcium recordings of ~100 neurons over ~5-15 minutes where individual neurons are
# randomly targeted and stimulated optogenetically
# We want to fit a linear dynamical system to the data in order to infer the connection weights between neurons
# The model is of the form
# x_t = A @ x_(t-1) + V @ u_t + b + w_t
# y_t = x_t + d + v_t

# the mapping of x_t to y_t is the identity
# V is diagonal
# w_t is a gaussian with 0 mean, and potentially diagonal covariance
# optionally would like to add a dependence on another time point in the past, i.e. H @ x_(t-2)
# fitting needs to handle missing data in y

# number of iteration through the full data set to run during gradient descent
num_grad_steps = 30
batch_size = 1

random_seed = 0
device = 'cpu'
# dtype = torch.float32
dtype = torch.float64

learning_rate = 1e-2
latent_dim = 100
obs_dim = latent_dim
num_time = 5000
num_datasets = 1
rng_seed = 0
cell_ids, data_final, stim_mat, params_true = pp.generate_data(latent_dim=latent_dim, obs_dim=obs_dim, num_time=num_time,
                                                               num_datasets=num_datasets, emission_eye=True, rng_seed=rng_seed+1)

# initialize parameters, see lds_construction.pdf for model structure
init_std = 0.1
var_log_offset = 0
dynamics_max_eig = 0.8

# initialize the parameters, see PDF for name definitions
rng_torch = torch.Generator(device=device)
rng_torch.manual_seed(random_seed)

dynamics_weights = torch.randn((latent_dim, latent_dim), device=device, dtype=dtype, generator=rng_torch) * init_std
dynamics_weights = dynamics_max_eig * dynamics_weights / torch.max(torch.abs(torch.linalg.eigvals(dynamics_weights)))

emissions_weights = torch.tensor(params_true['emissions_weights'], device=device, dtype=dtype)
inputs_weights_log = torch.randn(latent_dim, device=device, dtype=dtype, generator=rng_torch) * init_std - var_log_offset
emissions_input_weights = torch.zeros((obs_dim, latent_dim), dtype=dtype, device=device)
dynamics_cov_log = torch.randn(latent_dim, device=device, dtype=dtype, generator=rng_torch) * init_std - var_log_offset
emissions_cov_log = torch.randn(obs_dim, device=device, dtype=dtype, generator=rng_torch) * init_std - var_log_offset

# after initializing, require gradient on all the tensors
dynamics_weights.requires_grad = True
inputs_weights_log.requires_grad = True
dynamics_cov_log.requires_grad = True
emissions_cov_log.requires_grad = True

# save a copy of the initialization for plotting
dynamics_weights_init = dynamics_weights.detach().cpu().numpy().copy()
inputs_weights_log_init = inputs_weights_log.detach().cpu().numpy().copy()
dynamics_cov_log_init = dynamics_cov_log.detach().cpu().numpy().copy()
emissions_cov_log_init = emissions_cov_log.detach().cpu().numpy().copy()


def loss_fun(model_params, emissions, inputs):
    dynamics_weights = model_params[0]
    input_weights = torch.exp(model_params[1])
    dynamics_cov = torch.diag(torch.exp(model_params[2]))
    emissions_cov = torch.diag(torch.exp(model_params[3]))

    initial_weights_dict = {'mean': init_mean,
                            'cov': init_cov,
                            }

    dynamic_weights_dict = {'weights': dynamics_weights,
                            'input_weights': input_weights,
                            'bias': torch.zeros(latent_dim),
                            'cov': dynamics_cov,
                            }

    emission_weights_dict = {'weights': emissions_weights,
                             'input_weights': emissions_input_weights,
                             'bias': torch.zeros(obs_dim),
                             'cov': emissions_cov,
                             }

    params_example = {'initial': initial_weights_dict,
                      'dynamics': dynamic_weights_dict,
                      'emissions': emission_weights_dict,
                      }

    ll, _, _ = \
        infer.lgssm_filter(params_example, emissions, inputs=inputs)

    # ll, filtered_means, filtered_covs = \
    #     infer.lgssm_filter_simple(init_mean, init_cov, dynamics_weights, input_weights,
    #                               dynamics_cov, emissions_cov, emissions, inputs=inputs)

    return -ll


def loss_fun2(model_params, emissions, inputs):
    dynamics_weights = model_params[0]
    input_weights = torch.exp(model_params[1])
    dynamics_cov = torch.exp(model_params[2])
    emissions_cov = torch.exp(model_params[3])

    ll, filtered_means, filtered_covs = \
        infer.lgssm_filter_simple(init_mean, init_cov, dynamics_weights, input_weights, dynamics_cov, emissions_cov, emissions, inputs=inputs)

    return -ll


# perform gradient descent on ll with respect to the parameters
model_params = [dynamics_weights, inputs_weights_log, dynamics_cov_log, emissions_cov_log]
optimizer = torch.optim.Adam(model_params, lr=learning_rate)
loss_out = []

start = time.time()
emissions_torch = [torch.tensor(i, device=device, dtype=dtype) for i in data_final]
inputs_torch = [torch.tensor(i, device=device, dtype=dtype) for i in stim_mat]
rng_np = np.random.default_rng(random_seed)
init_cov = torch.tensor(params_true['init_cov'], device=device, dtype=dtype)
init_mean = torch.tensor(params_true['init_mean'], device=device, dtype=dtype)
this_data = emissions_torch[0]
this_input = inputs_torch[0]

for ep in range(num_grad_steps):
    optimizer.zero_grad()
    # loss = torch.tensor(0, device=device, dtype=dtype)

    loss = loss_fun(model_params, this_data, this_input)
    # loss = loss_fun2(model_params, this_data, this_input)

    # print(loss)
    # print(loss2)

    loss.backward()
    loss_out.append(loss.detach().cpu().numpy())
    optimizer.step()

    end = time.time()
    print('Finished step ' + str(ep + 1) + '/' + str(num_grad_steps))
    print('Loss = ' + str(loss_out[-1]))
    print('Time elapsed = ' + str(end - start))

# Plots
# Plot the loss
plt.figure()
plt.plot(loss_out)
plt.xlabel('iterations')
plt.ylabel('negative log likelihood')
plt.tight_layout()

# Plot the dynamics weights
plt.figure()
colorbar_shrink = 0.4
plt.subplot(2, 2, 1)
plt.imshow(dynamics_weights_init, interpolation='Nearest')
plt.title('init dynamics weights')
plt.xlabel('input neurons')
plt.ylabel('output neurons')

plt.subplot(2, 2, 2)
plt.imshow(dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
plt.title('fit dynamics weights')
plt.xlabel('input neurons')
plt.ylabel('output neurons')

plt.subplot(2, 2, 3)
plt.imshow(params_true['dynamics_weights'], interpolation='Nearest')
plt.title('true dynamics weights')
plt.colorbar()

plt.tight_layout()


# Plot the input weights
plt.figure()
plt.plot(np.exp(inputs_weights_log_init))
plt.plot(np.exp(inputs_weights_log.detach().cpu().numpy()))
plt.plot(params_true['input_weights'])
plt.legend(['init', 'final', 'true'])
plt.xlabel('neurons')
plt.ylabel('input weights')

# plot the covariances
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.exp(dynamics_cov_log_init))
plt.plot(np.exp(dynamics_cov_log.detach().cpu().numpy()))
plt.plot(params_true['dynamics_cov'])
plt.legend(['init', 'final', 'true'])
plt.xlabel('neurons')
plt.ylabel('dynamics noise cov')

plt.subplot(1, 2, 2)
plt.plot(np.exp(emissions_cov_log_init))
plt.plot(np.exp(emissions_cov_log.detach().cpu().numpy()))
plt.plot(params_true['emissions_cov'])
plt.xlabel('neurons')
plt.ylabel('emissions noise cov')

plt.tight_layout()

plt.show()

