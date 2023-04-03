import numpy as np
import pickle
import torch
import matplotlib
import matplotlib.pyplot as plt

# create fake data
with open('./example_data/synth_data.pkl', 'rb') as f:
    data = pickle.load(f)

latents = data['latents'][0]
model = data['model']
A = model.dynamics_weights.detach().numpy()
print(A)

#fit A_tau with L time lags
#X_i is a granger cause of another time series X_j if at least 1 element A_tau(j,i)
# for tau=1,...,L is signif larger than 0
# X_t = sum_tau=1^L A_tau*X(t-tau) + noise(t)

#show results for lags p

mse = []
bhat = []

for p in range(1,100):
    yvar = latents.T[:,p:]
    temp = np.ones(latents.shape[0]-p)
    zvar = np.vstack((temp,latents.T[:,:-p]))
    bhat.append(yvar @ zvar.T @ np.linalg.inv(zvar @ zvar.T))
    yhat = bhat[p-1] @ zvar
    mse.append(np.mean((yvar - yhat)**2))

fig, axs = plt.subplots(1,3)
Apos = axs[0].imshow(A)
bhatpos = axs[1].imshow(bhat[0])
fig.colorbar(Apos, ax=axs[0])
fig.colorbar(bhatpos, ax=axs[1])
axs[2].plot(range(1,100),mse)

plt.show()


