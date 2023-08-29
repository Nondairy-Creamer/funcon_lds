import numpy as np
from matplotlib import pyplot as plt


# this will convert from x to x_bar
def get_lagged_data(data, lags, add_pad=True):
    num_time, num_neurons = data.shape

    if add_pad:
        final_time = num_time
        pad = np.zeros((lags - 1, num_neurons))
        data = np.concatenate((pad, data), axis=0)
    else:
        final_time = num_time - lags + 1

    lagged_data = np.zeros((final_time, 0))

    for tau in reversed(range(lags)):
        if tau == lags - 1:
            lagged_data = np.concatenate((lagged_data, data[tau:, :]), axis=1)
        else:
            lagged_data = np.concatenate((lagged_data, data[tau:-lags + tau + 1, :]), axis=1)

    return lagged_data


# this will convert from a list of [A1, A2... An] to A_bar
def get_lagged_weights(weights, lags_out, fill='eye'):
    lagged_weights = np.concatenate(np.split(weights, weights.shape[0], 0), 2)[0, :, :]

    if fill == 'eye':
        fill_mat = np.eye(lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1])
    elif fill == 'zeros':
        fill_mat = np.zeros((lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1]))
    else:
        raise Exception('fill value not recognized')

    lagged_weights = np.concatenate((lagged_weights, fill_mat), 0)

    return lagged_weights


rng = np.random.default_rng(0)

size_data_t = 1000
num_neurons = 5

num_simulation_sample = 10
num_lags = 2

# get the location in the data where a stimulation event occured.
# this is your np.where operation on u
stim_index = 3

# initialize our measured neural data
x = rng.standard_normal((size_data_t, num_neurons))

# now set up a new matrix which is our prediction, starting at stim_index
# this prediction will be in -bar- space, where lags are in the state space.
# we'll move it back to normal space later
# for bayesian, this would be called the posterior predictive. You're not bayesian tho, so you're just predicting
num_timepoints_to_predict = 100
pred_x_bar = np.zeros((num_timepoints_to_predict, num_neurons*num_lags))

# need to get the initial value of our prediction
# a smart thing to do, is to initialize with the real data, so you have the true state of the system prior
# to stimulation
# this is a (num_lags x num_neurons matrix)
# we need a (num_lags * num_neurons matrix, 1) vector because our -bar- space includes the lags in the state space
pred_x_0 = x[stim_index-num_lags:stim_index, :]

# now turn this into a vector. Being very careful that we know the order np.reshape is accessing the matrix in
pred_x_0_column = np.reshape(pred_x_0, (num_lags * num_neurons))

# if you're having trouble getting the initial conditions for pred_x from the data, just generate random noise
# pred_x_0_column = rng.standard_normal((num_lags * num_neurons, 1))

# put initial value of the x_bar prediction as the first time point
pred_x_bar[0, :] = pred_x_0_column

# here we'll generate random A and B matricies.
# weights should be closer to zero the farther in the past they are
tau = 1
matrix_std = 0.2
A_bar = matrix_std * rng.standard_normal((num_lags, num_neurons, num_neurons)) * np.exp(-np.arange(num_lags) / tau)[:, None, None]
A_bar = get_lagged_weights(A_bar, num_lags)
B_bar = matrix_std * rng.standard_normal((num_lags, num_neurons, num_neurons)) * np.exp(-np.arange(num_lags) / tau)[:, None, None] / 4
B_bar = get_lagged_weights(B_bar, num_lags, fill='zeros')

# generate some inputs for our data
# here our inputs are random, but it should be a shifted version of the same matrix
inputs = rng.standard_normal((size_data_t, num_neurons))
inputs_bar = get_lagged_data(inputs, num_lags)
# this calculates the inputs that the state space will receive
# it is size (size_data_t x num_lags * num_neurons)
system_inputs = inputs_bar @ B_bar.T

for t in range(1, num_timepoints_to_predict):
    pred_x_bar[t, :] = A_bar @ pred_x_bar[t-1, :] + system_inputs[t]

pred_x = pred_x_bar[:, :num_neurons]

plt.figure()
plt.imshow(pred_x.T, aspect='auto', interpolation='nearest')

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(A_bar)
plt.subplot(1, 2, 2)
plt.imshow(B_bar)

plt.show()
