import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import inference_utilities as iu
from ssm_classes import Lgssm
from pathlib import Path


def plot_model_params(model_folder):
    colormap = mpl.colormaps['coolwarm']

    model_path = model_folder / 'model_trained.pkl'

    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    model_params = model.get_run_params()
    A_full = model_params['trained']['dynamics_weights'][:model.dynamics_dim, :]
    # A_full = A_full - np.eye(A_full.shape[0], A_full.shape[1])
    A = np.split(A_full, model.dynamics_lags, axis=1)
    cmax = np.max(np.abs(A_full))

    for ai, aa in enumerate(A):
        plt.figure()
        plt.imshow(aa, cmap=colormap)
        plt.clim((-cmax, cmax))
        plt.colorbar()
        plt.title('A' + str(ai))

    B_full = model_params['trained']['dynamics_input_weights'][:model.dynamics_dim, :]
    B = np.split(B_full, model.dynamics_input_lags, axis=1)
    cmax = np.max(np.abs(B_full))

    for bi, bb in enumerate(B):
        plt.figure()
        plt.imshow(bb, cmap=colormap)
        plt.clim((-cmax, cmax))
        plt.colorbar()
        plt.title('B' + str(bi))

    Q = model_params['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim]
    cmax = np.max(np.abs(Q))
    plt.figure()
    plt.imshow(model_params['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim], cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('Q')
    plt.colorbar()

    R = model_params['trained']['emissions_cov']
    cmax = np.max(np.abs(R))
    plt.figure()
    plt.imshow(model_params['trained']['emissions_cov'], cmap=colormap)
    plt.clim((-cmax, cmax))
    plt.title('R')
    plt.colorbar()

    plt.show()


def predict_from_model(model_folder):
    model_path = model_folder / 'model_trained.pkl'
    data_path = model_folder / 'data.pkl'

    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    data_file = open(data_path, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    emissions = data['emissions']
    inputs = data['inputs']
    cell_ids = data['cell_ids']
    inputs_lagged = [Lgssm.get_lagged_data(i, model.dynamics_input_lags) for i in inputs]

    has_stims = np.any(np.concatenate(inputs, axis=0), axis=0)
    inputs = [i[:, has_stims] for i in inputs]

    num_emissions = emissions[0].shape[1]
    num_inputs = inputs[0].shape[1]
    model_dynamics_dim = model.dynamics_dim
    model_input_dim = model.input_dim

    sample_time = 500
    sampled_model = model.sample(sample_time, inputs=inputs_lagged[0][:sample_time, :])

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(emissions[0][:sample_time, :].T)
    plt.subplot(2, 1, 2)
    plt.imshow(sampled_model[0].T)
    plt.show()


model_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models')
model_name = Path('local')

predict_from_model(model_folder / model_name)
plot_model_params(model_folder / model_name)
