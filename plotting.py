from matplotlib import pyplot as plt
import numpy as np
import torch


def trained_on_real(model_trained):
    # Plot the loss
    plt.figure()
    plt.plot(model_trained.loss)
    plt.xlabel('iterations')
    plt.ylabel('negative log likelihood')
    plt.tight_layout()

    # plot the time to train the model
    plt.figure()
    plt.plot(model_trained.train_time)
    plt.xlabel('iterations')
    plt.ylabel('total_time')
    plt.tight_layout()

    # Plot the dynamics weights
    plt.figure()
    colorbar_shrink = 0.4
    plt.subplot(1, 2, 1)
    plt.imshow(model_trained.dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
    plt.title('dynamics weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)

    plt.subplot(1, 2, 2)
    plt.imshow(model_trained.dynamics_weights.detach().cpu().numpy() - np.identity(model_trained.latent_dim), interpolation='Nearest')
    plt.title('dynamics weights - I')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.colorbar(shrink=colorbar_shrink)
    plt.tight_layout()

    # Plot the input weights
    plt.figure()
    plt.plot(np.exp(model_trained.inputs_weights_log_init))
    plt.plot(np.exp(model_trained.inputs_weights_log.detach().cpu().numpy()))
    plt.legend(['init', 'final'])
    plt.xlabel('neurons')
    plt.ylabel('input weights')

    # plot the covariances
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.exp(model_trained.dynamics_cov_log_init))
    plt.plot(np.exp(model_trained.dynamics_cov_log.detach().cpu().numpy()))
    plt.legend(['init', 'final'])
    plt.xlabel('neurons')
    plt.ylabel('dynamics noise cov')

    plt.subplot(1, 2, 2)
    plt.plot(np.exp(model_trained.emissions_cov_log_init))
    plt.plot(np.exp(model_trained.emissions_cov_log.detach().cpu().numpy()))
    plt.legend(['init', 'final'])
    plt.xlabel('neurons')
    plt.ylabel('emissions noise cov')
    plt.tight_layout()

    plt.show()


def trained_on_synthetic(model_synth_trained, model_synth_true, ll_true_params=None):
    import matplotlib as mpl
    mpl.use('TkAgg')

    # Plots
    # Plot the loss
    plt.figure()
    plt.plot(model_synth_trained.loss)

    if ll_true_params is not None:
        plt.plot(np.zeros(len(model_synth_trained.loss)) + ll_true_params, 'k-')
        plt.legend({'ll over fitting', 'll for the true parameters'})

    plt.xlabel('iterations')
    plt.ylabel('negative log likelihood')

    plt.tight_layout()

    # plot the time to train the model
    plt.figure()
    plt.plot(model_synth_trained.train_time)
    plt.xlabel('iterations')
    plt.ylabel('total_time')
    plt.tight_layout()

    # Plot the dynamics weights
    plt.figure()
    colorbar_shrink = 0.4
    plt.subplot(2, 2, 1)
    plt.imshow(model_synth_trained.dynamics_weights_init, interpolation='Nearest')
    plt.title('init dynamics weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')

    plt.subplot(2, 2, 2)
    plt.imshow(model_synth_trained.dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
    plt.title('fit dynamics weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')

    plt.subplot(2, 2, 3)
    plt.imshow(model_synth_true.dynamics_weights.detach().cpu().numpy(), interpolation='Nearest')
    plt.title('true dynamics weights')
    plt.colorbar()

    plt.tight_layout()

    # Plot the input weights
    plt.figure()
    plt.plot(np.exp(model_synth_trained.inputs_weights_log_init))
    plt.plot(np.exp(model_synth_trained.inputs_weights_log.detach().cpu().numpy()))
    plt.plot(np.exp(model_synth_true.inputs_weights_log.detach().cpu().numpy()))
    plt.legend(['init', 'final', 'true'])
    plt.xlabel('neurons')
    plt.ylabel('input weights')

    # plot the covariances
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.exp(model_synth_trained.dynamics_cov_log_init))
    plt.plot(np.exp(model_synth_trained.dynamics_cov_log.detach().cpu().numpy()))
    plt.plot(np.exp(model_synth_true.dynamics_cov_log.detach().cpu().numpy()))
    plt.legend(['init', 'final', 'true'])
    plt.xlabel('neurons')
    plt.ylabel('dynamics noise cov')

    plt.subplot(1, 2, 2)
    plt.plot(np.exp(model_synth_trained.emissions_cov_log_init))
    plt.plot(np.exp(model_synth_trained.emissions_cov_log.detach().cpu().numpy()))
    plt.plot(np.exp(model_synth_true.emissions_cov_log.detach().cpu().numpy()))
    plt.xlabel('neurons')
    plt.ylabel('emissions noise cov')

    plt.tight_layout()

    plt.show()