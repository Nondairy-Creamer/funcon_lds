import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl


colormap = mpl.colormaps['coolwarm']

model_folder = '/home/mcreamer/Documents/data_sets/fun_con_models'
model_name = 'model_47900336_trained.pkl'
model_path = model_folder + '/' + model_name

model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()

model_params = model.get_params()
A = model_params['trained']['dynamics_weights']

plt.figure()
plt.hist(A.reshape(-1), 100)

plt.figure()
plt.imshow(model_params['trained']['dynamics_weights'][:model.dynamics_dim, :], cmap=colormap)
plt.clim([-1, 1])
plt.colorbar()

plt.figure()
plt.imshow(model_params['trained']['dynamics_input_weights'][:model.dynamics_dim, :], cmap=colormap)
plt.clim([-0.05, 0.05])
plt.colorbar()

plt.figure()
plt.imshow(model_params['trained']['dynamics_cov'][:model.dynamics_dim, :model.dynamics_dim], cmap=colormap)
plt.clim([-0.1, 0.1])
plt.colorbar()

plt.figure()
plt.imshow(model_params['trained']['emissions_cov'][:model.dynamics_dim, :], cmap=colormap)
plt.clim([-1, 1])
plt.colorbar()

plt.show()
