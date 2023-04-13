import numpy as np
import matlab_implementation as mi
from scipy import linalg
import scipy.io as sio
import pickle


# TODO: convert matricies from neurons x time to time x neurons
# TODO: convert sampling so that the first time point comes from a step forward with A?
# test_LDSgaussian_EMfitting.m
#
# Sample from a latent Gaussian linear dynamical system (LDS) model, then
# run EM to estimate the model parameters

# Basic equations:
# -----------------
# X_t = A@X_{t-1} + eps_x,  eps_x ~ N(0,Q)  # latents
# Y_t = C@X_t + eps_y,      eps_y ~ N(0,R)  # observations
#
# With X_1 ~ N(0,Q0)    initial condition:

# Set dimensions
nz = 5  # dimensionality of latent z
ny = 5  # dimensionality of observation y
nu = 5  # dimensionality of external inputs
nT = 1000 * 1  # number of time steps
rng = np.random.default_rng(2)

# Set model parameters
# --------------------
#
# Set dynamics matrix A
if nz == 2:
    # Use rotation matrix if nz = 2
    thet = np.pi / 25
    A = 0.99 * np.array(((np.cos(thet), np.sin(thet)), (-np.sin(thet), np.cos(thet))))
else:
    # Generate random stable A
    A = rng.standard_normal((nz, nz))
    s, u = np.linalg.eig(A)  # get eigenvectors and eigenvals
    s = s / np.max(np.abs(s)) * 0.98  # set largest eigenvalue to lie inside unit circle (enforcing stability)
    s[np.real(s) < 0] = -s[np.real(s) < 0]  # set real parts to be positive (encouraging smoothness)
    A = np.real(u @ np.linalg.solve(u.T, np.diag(s)).T)  # reconstruct A

# Set observation matrix C
C = np.eye(nz)  # loading weights

# Set input matrices B and D
B = 0.5 * rng.standard_normal((nz, nu))  # weights from inputs to latents
D = 0.5 * rng.standard_normal((ny, nu))  # weights from inputs to observed

# Dynamics noise covariance
Q = rng.standard_normal((nz, nz))
Q = 0.1 * (Q.T @ Q + np.eye(nz))  # dynamics noise covariance
R = np.diag(1 * rng.uniform(size=ny) + 0.1)  # Y noise covariance
Q0 = 2 * np.eye(nz)  # Covariance for latent in first time step

# Use discrete Lyapunov equation solver to compute asymptotic covariance
P = linalg.solve_discrete_lyapunov(A, Q)

## Sample data from LDS model

uu = rng.standard_normal((nu, nT))  # external inputs

mmtrue = {'A': A,
          'B': B,
          'C': C,
          'D': D,
          'Q': Q,
          'R': R,
          'Q0': Q0,
          }

yy, zz = mi.sampleLDSgauss(mmtrue, nT, uu, rng)[:2]  # sample from model

## Compute latents and log-marginal likelihood given true params

# Run Kalman Filter-Smoother to get posterior over latents given true data
# zzmutrue, loglitrue, zzcovtrue = mi.runKalmanSmooth(yy, uu, mmtrue)[:3]
# print('Log-evidence at true params: ' + str(loglitrue))

## Compute ML estimate for model params using EM
#
# Set options for EM
optsEM = {'maxiter': 10,  # maximum # of iterations
          'dlogptol': 1e-4,  # stopping tolerance
          'display': 10,  # display frequency
          # Specify which parameters to learn.  (Set to '0' or 'false' to NOT update).
          'update': {'A': True,
                     'B': True,
                     'C': False,
                     'D': True,
                     'Q': True,
                     'R': True,
                     'Q0': True,
                     },
          }

# Initialize fitting struct
mm0 = {}

if optsEM['update']['A']: mm0['A'] = 0.5 * A + 0.1 * rng.standard_normal((nz, nz))
mm0['C'] = np.eye(nz)
if optsEM['update']['Q']: mm0['Q'] = 1.33 * Q
if optsEM['update']['R']: mm0['R'] = 1.5 * R
if optsEM['update']['B']: mm0['B'] = 0.5 * B
if optsEM['update']['D']: mm0['D'] = 0.5 * D
if optsEM['update']['Q0']: mm0['Q0'] = Q0

optsEM['update']['Dynam'] = True
optsEM['update']['Obs'] = True

# Run EM
# Run EM inference for model parameters
# save the generated data
save_folder = '/home/mcreamer/Documents/data_sets/matlab_kalman_data.mat'
save_dict = {'yy': yy,
             'mm0': mm0,
             'uu': uu,
             'optsEM': optsEM,
             'mmtrue': mmtrue,
             }
sio.savemat(save_folder, save_dict)


###### load data from python
load_file = open('/home/mcreamer/Documents/python/funcon_lds/example_data/data.pkl', 'rb')
data_in = pickle.load(load_file)

yy = data_in['emissions'][1].T
uu = data_in['inputs'][1].T
mm0 = {'A': data_in['params_init']['dynamics']['weights'],
       'B': data_in['params_init']['dynamics']['input_weights'],
       'C': data_in['params_init']['emissions']['weights'],
       'D': data_in['params_init']['emissions']['input_weights'],
       'Q': data_in['params_init']['dynamics']['cov'],
       'R': data_in['params_init']['emissions']['cov'],
       'Q0': data_in['params_init']['init_cov'][1],
       }

import time
start = time.time()
mm1, logEvTrace = mi.runEM_LDSgaussian(yy, mm0, uu, optsEM)
print('time to train', time.time()-start)

## Examine fitted model

# Compute posterior mean and cov of latents, and log-evidence at optimum
zzmu1, logli1, zzcov1 = mi.runKalmanSmooth(yy, uu, mm1)[:3]

from matplotlib import pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(mmtrue['A'])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(mm1['A'])
plt.colorbar()
plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(mmtrue['B'])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(mm1['B'])
plt.colorbar()
plt.show()


#
# # Align fitted model with true model (so we can compare params)
# mm1a = mi.alignLDSmodels(zzmu1, zzmutrue, mm1)
# zzmu1a, logli1a, zzcov1a = mi.runKalmanSmooth(yy, uu, mm1a)[:3]  # recompute logli (just to make sure it hasn't changed)
#
# # Report whether optimization succeeded in finding a posible global optimum
# print('Log-evidence at true params:', loglitrue)
# print('Log-evidence at inferred params:', logli1)
# # Report if we found the global optimum
# if logli1 >= loglitrue:
#     print('(found better optimum -- SUCCESS!)')
# else:
#     print('(FAILED to find optimum!)')

