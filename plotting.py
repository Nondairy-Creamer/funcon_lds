from matplotlib import pyplot as plt
import numpy as np
import torch
import matplotlib as mpl


def trained_on_real(log_likelihood, trained_params, init_params):
    # Plot the log likelihood
    plt.figure()
    plt.plot(log_likelihood)
    plt.xlabel('iterations')
    plt.ylabel('log likelihood')
    plt.tight_layout()

    # Plot the dynamics weights
    plt.figure()
    colorbar_shrink = 0.4
    plt.subplot(1, 2, 1)
    plt.imshow(trained_params.dynamics.weights, interpolation='Nearest')
    plt.title('dynamics weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)

    plt.subplot(1, 2, 2)
    plt.imshow(trained_params.dynamics.weights - torch.eye(trained_params.dynamics.weights.shape[0]), interpolation='Nearest')
    plt.title('dynamics weights - I')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)
    plt.tight_layout()

    # Plot the input weights
    plt.figure()
    plt.plot(init_params.dynamics.input_weights)
    plt.plot(trained_params.dynamics.input_weights)
    plt.legend(['init', 'final'])
    plt.xlabel('neurons')
    plt.ylabel('input weights')

    # plot the dynamics covariance
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(init_params.dynamics.cov, interpolation='Nearest')
    plt.title('init dynamics cov')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)

    plt.subplot(1, 2, 2)
    plt.imshow(trained_params.dynamics.cov, interpolation='Nearest')
    plt.title('trained dynamics cov')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)
    plt.tight_layout()

    # plot the emissions covariance
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(init_params.emissions.cov, interpolation='Nearest')
    plt.title('init emissions cov')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)

    plt.subplot(1, 2, 2)
    plt.imshow(trained_params.emissions.cov, interpolation='Nearest')
    plt.title('trained emissions cov')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)
    plt.tight_layout()

    plt.show()


def trained_on_synthetic_diag(model_synth_trained, model_synth_true, ll_true_params=None):
    # Plots
    # Plot the log likelihood
    plt.figure()
    plt.plot(model_synth_trained.log_likelihood)

    if ll_true_params is not None:
        plt.axhline(ll_true_params, color='k', linestyle=':', label="true")
        plt.legend({'log likelihood for model', 'log likelihood for true parameters'})

    plt.xlabel('iterations')
    plt.ylabel('log likelihood')
    plt.tight_layout()

    # plot the time to train the model
    plt.figure()
    plt.plot(model_synth_trained.train_time)
    plt.xlabel('iterations')
    plt.ylabel('total_time')
    plt.tight_layout()

    # Plot the dynamics weights
    model_synth_true_np = model_synth_true.dynamics_weights.detach().cpu().numpy()
    model_synth_trained_np = model_synth_trained.dynamics_weights.detach().cpu().numpy()
    model_synth_init_np = model_synth_trained.dynamics_weights_init
    compare_2d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='dynamics weights')

    # Plot the input weights
    model_synth_true_np = np.exp(model_synth_true.inputs_weights_log.detach().cpu().numpy())
    model_synth_trained_np = np.exp(model_synth_trained.inputs_weights_log.detach().cpu().numpy())
    model_synth_init_np = np.exp(model_synth_trained.inputs_weights_log_init)
    compare_1d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='input weights')

    # plot the covariances
    model_synth_true_np = np.exp(model_synth_true.dynamics_cov_log.detach().cpu().numpy())
    model_synth_trained_np = np.exp(model_synth_trained.dynamics_cov_log.detach().cpu().numpy())
    model_synth_init_np = np.exp(model_synth_trained.dynamics_cov_log_init)
    compare_1d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='dynamics covariance')

    model_synth_true_np = np.exp(model_synth_true.emissions_cov_log.detach().cpu().numpy())
    model_synth_trained_np = np.exp(model_synth_trained.emissions_cov_log.detach().cpu().numpy())
    model_synth_init_np = np.exp(model_synth_trained.emissions_cov_log_init)
    compare_1d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='emissions covariance')


def trained_on_synthetic(model_synth_trained, model_synth_true, ll_true_params=None):
    # Plots
    # Plot the log likelihood
    plt.figure()
    plt.plot(model_synth_trained.log_likelihood)

    if ll_true_params is not None:
        plt.axhline(ll_true_params, color='k', linestyle=':', label="true")
        plt.legend({'log likelihood for model', 'log likelihood for true parameters'})

    plt.xlabel('iterations')
    plt.ylabel('log likelihood')
    plt.tight_layout()

    # plot the time to train the model
    plt.figure()
    plt.plot(model_synth_trained.train_time)
    plt.xlabel('iterations')
    plt.ylabel('total_time')
    plt.tight_layout()

    # Plot the dynamics weights
    model_synth_true_np = model_synth_true.dynamics_weights.detach().cpu().numpy()
    model_synth_trained_np = model_synth_trained.dynamics_weights.detach().cpu().numpy()
    model_synth_init_np = model_synth_trained.dynamics_weights_init
    compare_2d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='dynamics weights')

    # Plot the input weights
    model_synth_true_np = model_synth_true.inputs_weights.detach().cpu().numpy()
    model_synth_trained_np = model_synth_trained.inputs_weights.detach().cpu().numpy()
    model_synth_init_np = model_synth_trained.inputs_weights_init
    compare_2d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='input weights')

    # plot the covariances
    model_synth_true_np = model_synth_true.dynamics_cov.detach().cpu().numpy()
    model_synth_trained_np = model_synth_trained.dynamics_cov.detach().cpu().numpy()
    model_synth_init_np = model_synth_trained.dynamics_cov_init
    compare_2d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='dynamics covariance')

    model_synth_true_np = model_synth_true.emissions_cov.detach().cpu().numpy()
    model_synth_trained_np = model_synth_trained.emissions_cov.detach().cpu().numpy()
    model_synth_init_np = model_synth_trained.emissions_cov_init
    compare_2d(model_synth_true_np, model_synth_trained_np, model_synth_init_np, title='emissions covariance')


def compare_2d(true, fit, init, title=''):
    abs_max = np.max([np.max(np.abs(i)) for i in [init, true, fit, true - fit]])
    colormap = mpl.colormaps['coolwarm']

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(init, interpolation='Nearest', cmap=colormap)
    plt.title('init weights, ' + title)
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(fit, interpolation='Nearest', cmap=colormap)
    plt.title('fit weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(true - fit, interpolation='Nearest', cmap=colormap)
    plt.title('true - fit')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(true, interpolation='Nearest', cmap=colormap)
    plt.title('true weights')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def compare_1d(true, fit, init, title=''):
    # Plot the input weights
    plt.figure()
    plt.plot(init)
    plt.plot(fit)
    plt.plot(true)
    plt.legend(['init', 'final', 'true'])
    plt.xlabel('neurons')
    plt.ylabel('weights')
    plt.title(title)
    plt.show()

