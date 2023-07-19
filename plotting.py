from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl


def plot_model_params(model_trained, model_true=None):
    # Takes in a model and plots out all its parameters
    ll_true_params = None

    if model_true is not None:
        params_true = model_true.get_params()
        ll_true_params = model_true.log_likelihood[0]

    # Plots
    # Plot the log likelihood
    plt.figure()
    plt.plot(model_trained.log_likelihood)

    if ll_true_params is not None:
        plt.axhline(ll_true_params, color='k', linestyle=':', label="true")
        plt.legend({'log likelihood for model', 'log likelihood for true parameters'})

    plt.xlabel('iterations')
    plt.ylabel('log likelihood')
    plt.tight_layout()

    # plot the time to train the model
    plt.figure()
    plt.plot(model_trained.train_time)
    plt.xlabel('iterations')
    plt.ylabel('total_time')
    plt.tight_layout()

    params_trained = model_trained.get_params()

    # plot each of the model parameters
    for k in params_trained['trained'].keys():
        if model_trained.param_props['update'][k]:
            # if there are multiple lags, then only the top left block of the covariance has any trained parameters
            # For the non covariance parameters (A, B) then the top row of blocks has trained parameters
            if k[-3:] == 'cov':
                num_cols = model_trained.dynamics_dim
            else:
                num_cols = params_trained['init'][k].shape[1]

            param_init = params_trained['init'][k][:model_trained.dynamics_dim, :num_cols]
            param_trained = params_trained['trained'][k][:model_trained.dynamics_dim, :num_cols]

            # for synthetic data we have a true model, get the true parameters here
            if model_true is not None:
                param_true = params_true['trained'][k][:model_true.dynamics_dim, :num_cols]
            else:
                param_true = None

            plot_params(param_init, param_trained, param_true, title=k)


def plot_params(param_init, param_trained, param_true, title=''):
    # plot a specific model parameter, usually called by plot_model_params
    colormap = mpl.colormaps['coolwarm']

    if param_true is not None:
        n_row = 2
        abs_max = np.max([np.max(np.abs(i)) for i in [param_trained, param_true, param_true - param_trained]])
    else:
        n_row = 1
        abs_max = np.max([np.max(np.abs(i)) for i in [param_trained]])

    plt.figure()
    plt.subplot(n_row, 2, 1)
    plt.imshow(param_init, interpolation='Nearest', cmap=colormap)
    plt.title('init weights, ' + title)
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    init_abs_max = np.max(np.abs(param_init))
    plt.clim((-init_abs_max, init_abs_max))
    plt.colorbar()

    plt.subplot(n_row, 2, 2)
    plt.imshow(param_trained, interpolation='Nearest', cmap=colormap)
    plt.title('fit weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()

    if param_true is not None:
        plt.subplot(n_row, 2, 3)
        plt.imshow(param_true - param_trained, interpolation='Nearest', cmap=colormap)
        plt.title('true - fit')
        plt.clim((-abs_max, abs_max))
        plt.colorbar()

        plt.subplot(n_row, 2, 4)
        plt.imshow(param_true, interpolation='Nearest', cmap=colormap)
        plt.title('true weights')
        plt.clim((-abs_max, abs_max))
        plt.colorbar()

    plt.tight_layout()
    plt.show()



