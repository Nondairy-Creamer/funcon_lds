import numpy as np
import pickle
import torch
import matplotlib
import matplotlib.pyplot as plt
from ssm_classes import LgssmSimple
import preprocessing as pp


params = pp.get_params(param_name='params_synth')

device = params["device"]
dtype = getattr(torch, params["dtype"])

model_synth_true = LgssmSimple(params["latent_dim"], dtype=dtype, device=device)
# randomize the parameters (defaults are nonrandom)
model_synth_true.randomize_weights(random_seed=params["random_seed"])
# sample from the randomized model
synth_data_dict = model_synth_true.sample(
    num_time=params["num_time"],
    num_data_sets=params["num_data_sets"],
    nan_freq=params["nan_freq"],
    random_seed=params["random_seed"],
)

emissions = synth_data_dict["emissions"][0]
inputs = synth_data_dict["inputs"]
init_mean_true = synth_data_dict["init_mean"]
init_cov_true = synth_data_dict["init_cov"]
latents = synth_data_dict["latents"][0]

A = model_synth_true.dynamics_weights.detach().numpy()
print(A)

# fit A_hat with p time lags
# X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_1^L A_tau*X(t-tau) + noise(t)

mse = []

num_lags = 3
num_time, num_neurons = emissions.shape
# y_target is the time series we are trying to predict from A_hat @ y_history
# y_target should start at t=0+num_lags
y_target = np.zeros((num_time - num_lags, num_neurons))
# y_target is the lagged time series, should start at t=0+num_lags-1
# we will concatenate each of the columns of the y_history matrix where each column corresponds to a lagged time series
y_history = np.zeros((num_time - num_lags, 0))

# note this goes from time point num_lags to T
y_target = emissions[:-num_lags, :]

for p in range(1, num_lags+1):
    if p-num_lags:
        y_history = np.concatenate((y_history, emissions[p:p-num_lags, :]), axis=1)
    else:
        y_history = np.concatenate((y_history, emissions[p:, :]), axis=1)

A_hat = np.linalg.solve(y_history, y_target).T

y_hat = y_target @ A_hat.T

# mse.append(np.mean((y - yhat) ** 2))

# fig, axs = plt.subplots(1, 3)
# Apos = axs[0].imshow(A)
# bhatpos = axs[1].imshow(bhat[0])
# fig.colorbar(Apos, ax=axs[0])
# fig.colorbar(bhatpos, ax=axs[1])
# axs[2].plot(range(1, 100), mse)
#
# plt.show()
