import numpy as np
import matlab_implementation as mi
from scipy import linalg


# test_LDSgaussian_EMfitting.m
#
# Sample from a latent Gaussian linear dynamical system (LDS) model, then
# run EM to estimate the model parameters

# Basic equations:
# -----------------
# X_t = A*X_{t-1} + eps_x,  eps_x ~ N(0,Q)  # latents
# Y_t = C*X_t + eps_y,      eps_y ~ N(0,R)  # observations
#
# With X_1 ~ N(0,Q0)    initial condition:

# Set dimensions
nz = 100  # dimensionality of latent z
ny = 100  # dimensionality of observation y
nu = 3  # dimensionality of external inputs
nT = 5000 * 10  # number of time steps

# Set model parameters
# --------------------

# Set dynamics matrix A
if nz == 2:
    # Use rotation matrix if nz = 2
    thet = np.pi / 25
    A = 0.99 * np.array((np.cos(thet), np.sin(thet), -np.sin(thet), np.cos(thet)))
else:
    # Generate random stable A
    A = np.randn(nz)
    u, s = np.linalg.eig(A, 'vector')  # get eigenvectors and eigenvals
    s = s / np.max(np.abs(s)) * 0.98  # set largest eigenvalue to lie inside unit circle (enforcing stability)
    s[np.real(s) < 0] = -s(np.real(s) < 0)  # set real parts to be positive (encouraging smoothness)
    A = np.real(u * (np.diag(s) / u))  # reconstruct A

rng = np.random.default_rng(0)
# Set observation matrix C
C = 0.5 * rng.standard_normal(ny, nz)  # loading weights

# Set input matrices B and D
B = 0.5 * rng.standard_normal((nz, nu))  # weights from inputs to latents
D = 0.5 * rng.standard_normal((ny, nu))  # weights from inputs to observed

# Dynamics noise covariance
Q = rng.standard_normal(nz)
Q = 0.1 * (Q.T * Q + np.eye(nz))  # dynamics noise covariance
R = np.diag(1 * np.standard_normal((ny, 1)) + 0.1)  # Y noise covariance
Q0 = 2 * np.eye(nz)  # Covariance for latent in first time step

# Use discrete Lyapunov equation solver to compute asymptotic covariance
P = linalg.solve_discrete_lyapunov(A, Q)

## Sample data from LDS model

uu = rng.standard_normal(nu, nT)  # external inputs

mmtrue = {'A': A,
          'B': B,
          'C': C,
          'D': D,
          'Q': Q,
          'R': R,
          'Q0': Q0,
          }

yy, zz = mi.sampleLDSgauss(mmtrue, nT, uu)  # sample from model

## Compute latents and log-marginal likelihood given true params

# Run Kalman Filter-Smoother to get posterior over latents given true data
zzmutrue, loglitrue, zzcovtrue = mi.runKalmanSmooth(yy, uu, mmtrue)
print('Log-evidence at true params: ' + str(loglitrue))

## Compute ML estimate for model params using EM

# Set options for EM
optsEM = {'maxiter': 250,  # maximum # of iterations
          'dlogptol': 1e-4,  # stopping tolerance
          'display': 10,  # display frequency
          # Specify which parameters to learn.  (Set to '0' or 'false' to NOT update).
          'update': {'A': True,
                     'B': True,
                     'C': True,
                     'D': True,
                     'Q': True,
                     'R': True,
                     },
          }

# Initialize fitting struct
mm0 = {'A': A,
       'B': B,
       'C': C,
       'D': D,
       'Q': Q,
       'R': R,
       'Q0': Q0,
       }

# make struct with initial params
if optsEM['update']['A']:
    mm0.A = A * 0.5 + rng.standard_normal(nz) * 0.1 # initial A param

if optsEM['update']['C']:
    mm0.C = C * .9 + rng.standard_normal(ny, nz) * 0.1 # initial C param

if optsEM['update']['Q']:
    mm0.Q = Q * 1.33 # initial Q param

if optsEM['update']['R']:
    mm0.R = R * 1.5 # initial R param

if optsEM['update']['B']:
    mm0.B = B * 0.5 # initial B param

if optsEM['update']['D']:
    mm0.D = D * 0.5 # initial D param

## Run EM

# Run EM inference for model parameters
mm1, logEvTrace = mi.runEM_LDSgaussian(yy, mm0, uu, optsEM)

## Examine fitted model

# Compute posterior mean and cov of latents, and log-evidence at optimum
zzmu1, logli1, zzcov1 = mi.runKalmanSmooth(yy, uu, mm1)

# Align fitted model with true model (so we can compare params)
mm1a = mi.alignLDSmodels(zzmu1, zzmutrue, mm1)
zzmu1a, logli1a, zzcov1a = mi.runKalmanSmooth(yy, uu, mm1a) # recompute logli (just to make sure it hasn't changed)

# Report whether optimization succeeded in finding a posible global optimum
print('Log-evidence at true params: ' + str(loglitrue))
print('Log-evidence at inferred params: ' + str(logli1))
# Report if we found the global optimum
if logli1 >= loglitrue:
    print('(found better optimum -- SUCCESS!)')
else:
    print('(FAILED to find optimum!)')

