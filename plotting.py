from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl


def plot_model_params(model_synth_trained, model_synth_true=None, ll_true_params=None):
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

    if model_synth_true is not None:
        params_true = model_synth_true.get_params()
    params_trained = model_synth_trained.get_params()

    for k in params_trained['trained'].keys():
        if model_synth_trained.param_props['update'][k]:
            param_init = params_trained['init'][k]
            param_trained = params_trained['trained'][k]

            if model_synth_true is not None:
                param_true = params_true['trained'][k]
            else:
                param_true = None

            plot_params(param_init, param_trained, param_true, title=k)


def plot_params(true_param, fit_param, init_param, title=''):
    if true_param.ndim == 1:
        compare_1d(true_param, fit_param, init_param, title=title)
    else:
        compare_2d(true_param, fit_param, init_param, title=title)


def compare_1d(init_param, fit_param, true_param, title=''):
    # Plot the input weights
    plt.figure()
    plt.plot(init_param)
    plt.plot(fit_param)

    if true_param is not None:
        plt.plot(true_param)
        plt.legend(['init', 'final', 'true'])
    else:
        plt.legend(['init', 'final'])

    plt.xlabel('neurons')
    plt.ylabel('weights')
    plt.title(title)
    plt.show()


def compare_2d(init_param, fit_param, true_param, title=''):
    colormap = mpl.colormaps['coolwarm']

    if true_param is not None:
        n_row = 2
        abs_max = np.max([np.max(np.abs(i)) for i in [init_param, fit_param, true_param, true_param - fit_param]])
    else:
        n_row = 1
        abs_max = np.max([np.max(np.abs(i)) for i in [init_param, fit_param]])

    plt.figure()
    plt.subplot(n_row, 2, 1)
    plt.imshow(init_param, interpolation='Nearest', cmap=colormap)
    plt.title('init weights, ' + title)
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()

    plt.subplot(n_row, 2, 2)
    plt.imshow(fit_param, interpolation='Nearest', cmap=colormap)
    plt.title('fit weights')
    plt.xlabel('input neurons')
    plt.ylabel('output neurons')
    plt.clim((-abs_max, abs_max))
    plt.colorbar()

    if true_param is not None:
        plt.subplot(n_row, 2, 3)
        plt.imshow(true_param - fit_param, interpolation='Nearest', cmap=colormap)
        plt.title('true - fit')
        plt.clim((-abs_max, abs_max))
        plt.colorbar()

        plt.subplot(n_row, 2, 4)
        plt.imshow(true_param, interpolation='Nearest', cmap=colormap)
        plt.title('true weights')
        plt.clim((-abs_max, abs_max))
        plt.colorbar()

    plt.tight_layout()
    plt.show()


