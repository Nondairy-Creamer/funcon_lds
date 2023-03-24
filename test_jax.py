from jax import numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import monotonically_increasing
from ssm_jax import LgssmJax
import optax


def plot_learning_curve(marginal_lls, true_model, true_params, test_model, test_params, emissions, inputs):
    plt.figure()
    plt.xlabel("iteration")
    nsteps = len(marginal_lls)
    plt.plot(marginal_lls, label="estimated")
    true_logjoint = (true_model.log_prior(true_params) + true_model.marginal_log_prob(true_params, emissions, inputs=inputs))
    plt.axhline(-true_logjoint / emissions.size, color='k', linestyle=':', label="true")
    plt.ylabel("marginal joint probability")
    plt.legend()


def plot_predictions(true_model, true_params, test_model, test_params, emissions, inputs):
    smoothed_emissions, smoothed_emissions_std = test_model.posterior_predictive(test_params, emissions, inputs=inputs)

    spc = 3
    plt.figure(figsize=(10, 4))
    for i in range(emission_dim):
        plt.plot(emissions[:, i] + spc * i, "--k", label="observed" if i == 0 else None)
        ln = plt.plot(smoothed_emissions[:, i] + spc * i,
                        label="smoothed" if i == 0 else None)[0]
        plt.fill_between(
            jnp.arange(num_timesteps),
            spc * i + smoothed_emissions[:, i] - 2 * jnp.sqrt(smoothed_emissions_std[i]),
            spc * i + smoothed_emissions[:, i] + 2 * jnp.sqrt(smoothed_emissions_std[i]),
            color=ln.get_color(),
            alpha=0.25,
        )
    plt.xlabel("time")
    plt.xlim(0, num_timesteps - 1)
    plt.ylabel("true and predicted emissions")
    plt.legend()
    plt.show()


state_dim = 4
emission_dim = state_dim
input_dim = state_dim
num_timesteps = 1000
key = jr.PRNGKey(0)

true_model = LgssmJax(state_dim, emission_dim, input_dim=input_dim)

key, key_root = jr.split(key)
max_eig_param = 0.8
dynamics_weights_true = jr.normal(key, shape=[state_dim, state_dim])
max_eig = jnp.max(jnp.abs(jnp.linalg.eigvals(dynamics_weights_true)))
dynamics_weights_true = dynamics_weights_true / max_eig * max_eig_param

key, key_root = jr.split(key)
dynamics_inputs_true = jr.normal(key, shape=[state_dim]) / 10
key, key_root = jr.split(key)
dynamics_cov_true = jnp.diag(jnp.exp(jr.normal(key, shape=[state_dim]) / 10 - 1))

key, key_root = jr.split(key)
emissions_cov_true = jnp.diag(jnp.exp(jr.normal(key, shape=[emission_dim]) / 10 - 1))

key, key_root = jr.split(key)
inputs = jr.normal(key, shape=[num_timesteps, state_dim]) / 10

key, key_root = jr.split(key)
true_params, param_props = true_model.initialize(key,
                                                 dynamics_weights=dynamics_weights_true,
                                                 dynamics_input_weights=dynamics_inputs_true,
                                                 dynamics_covariance=dynamics_cov_true,
                                                 dynamics_bias=jnp.zeros(state_dim),
                                                 emission_weights=jnp.eye(emission_dim, state_dim),
                                                 emission_input_weights=jnp.zeros((emission_dim, input_dim)),
                                                 emission_covariance=emissions_cov_true,
                                                 emission_bias=jnp.zeros(emission_dim),
                                                 )

key, key_root = jr.split(key)
true_states, emissions = true_model.sample(true_params, key, num_timesteps, inputs=inputs)

# Plot the true states and emissions
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(emissions + 3 * jnp.arange(emission_dim))
ax.set_ylabel("data")
ax.set_xlabel("time")
ax.set_xlim(0, num_timesteps - 1)

# Plot predictions from a random, untrained model
test_model = LgssmJax(state_dim, emission_dim, input_dim=input_dim)
key = jr.PRNGKey(42)
test_params, param_props = test_model.initialize(key,
                                                 dynamics_bias=jnp.zeros(state_dim),
                                                 emission_weights=jnp.eye(state_dim),
                                                 emission_input_weights=jnp.zeros((emission_dim, input_dim)),
                                                 emission_bias=jnp.zeros(state_dim),
                                                 )


param_props.dynamics.bias.trainable = False
param_props.emissions.bias.trainable = False
param_props.emissions.weights.trainable = False

init_params = test_params

plot_predictions(true_model, true_params, test_model, init_params, emissions, inputs)

num_iters = 500
em_2 = jnp.array((emissions, emissions))
alpha = 1e-2
optimizer = optax.adam(alpha)
# test_params, marginal_lls = test_model.fit_em(test_params, param_props, emissions, num_iters=num_iters)
test_params, marginal_lls = test_model.fit_sgd(test_params, param_props, emissions,
                                               inputs=inputs, num_epochs=num_iters,
                                               optimizer=optimizer)

plot_learning_curve(marginal_lls, true_model, true_params, test_model, test_params, emissions, inputs)
plot_predictions(true_model, true_params, test_model, test_params, emissions, inputs)



# Plot the dynamics weights
plt.figure()
colorbar_shrink = 0.4
plt.subplot(2, 2, 1)
plt.imshow(init_params.dynamics.weights, interpolation='Nearest')
plt.title('init dynamics weights')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(test_params.dynamics.weights, interpolation='Nearest')
plt.title('fit dynamics weights')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(true_params.dynamics.weights, interpolation='Nearest')
plt.title('true dynamics weights')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow((true_params.dynamics.weights - test_params.dynamics.weights)**2, interpolation='Nearest')
plt.title('squared error trained vs true weights')
plt.colorbar()

plt.tight_layout()

# Plot the input weights
plt.figure()
plt.plot(init_params.dynamics.input_weights)
plt.plot(test_params.dynamics.input_weights)
plt.plot(true_params.dynamics.input_weights)
plt.legend(['init', 'final', 'true'])
plt.xlabel('neurons')
plt.ylabel('input weights')

# plot the covariances
plt.figure()
colorbar_shrink = 0.4
plt.subplot(2, 2, 1)
plt.imshow(init_params.dynamics.cov, interpolation='Nearest')
plt.title('init dynamics covariance')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(test_params.dynamics.cov, interpolation='Nearest')
plt.title('fit dynamics covariance')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(true_params.dynamics.cov, interpolation='Nearest')
plt.title('true dynamics covariance')
plt.colorbar()

plt.subplot(2, 2, 4)
dynamics_cov_error = true_params.dynamics.cov - test_params.dynamics.cov
max_abs_error = jnp.max(jnp.abs(dynamics_cov_error))
plt.imshow(dynamics_cov_error, interpolation='Nearest')
plt.title('true - trained')
plt.clim((-max_abs_error, max_abs_error))
plt.colorbar()

plt.tight_layout()


plt.figure()
colorbar_shrink = 0.4
plt.subplot(2, 2, 1)
plt.imshow(init_params.emissions.cov, interpolation='Nearest')
plt.title('init emissions covariance')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(test_params.emissions.cov, interpolation='Nearest')
plt.title('fit emissions covariance')
plt.xlabel('input neurons')
plt.ylabel('output neurons')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(true_params.emissions.cov, interpolation='Nearest')
plt.title('true emissions covariance')
plt.colorbar()

plt.subplot(2, 2, 4)
emissions_cov_error = true_params.emissions.cov - test_params.emissions.cov
max_abs_error = jnp.max(jnp.abs(emissions_cov_error))
plt.imshow(emissions_cov_error, interpolation='Nearest')
plt.title('true - trained')
plt.clim((-max_abs_error, max_abs_error))
plt.colorbar()

plt.tight_layout()

plt.show()