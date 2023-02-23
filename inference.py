import torch
import numpy as np
import time


# @torch.compile
def lgssm_filter_simple(init_mean, init_cov, dynamics_weights, input_weights,
                        dynamics_cov, emissions_cov, emissions,
                        inputs=None, nan_fill=1e8):
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
    adapted from Dynamax.
    This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions

    Args:
        params: model parameters
        emissions: array of observations.
        inputs: optional array of inputs.
        nan_fill: value to put in place of nans in the emissions covariance diagonal

    Returns:
        PosteriorGSSMFiltered: filtered posterior object
    """
    dtype = emissions.dtype
    device = emissions.device
    num_timesteps = len(emissions)

    if inputs is None:
        inputs = torch.zeros((num_timesteps, 0), dtype=dtype, device=device)

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        y = emissions[t]

        # added by msc to attempt to perform inference over nans
        nan_loc = torch.isnan(y)
        y = torch.where(nan_loc, 0, y)
        R = torch.where(nan_loc, nan_fill, emissions_cov)

        # Update the log likelihood
        mu = pred_mean
        cov = pred_cov + torch.diag(R)

        # due to floating point precision, cov can end up not being psd
        # cov = nearest_psd(cov)

        # start = time.time()
        # ll += torch.distributions.MultivariateNormal(mu, cov).log_prob(y)
        # print('built in: ' + str(time.time() - start))
        # start = time.time()
        ll += -torch.linalg.slogdet(cov)[1] - torch.dot(y - mu, torch.linalg.solve(cov, y - mu))
        # print('hand made in: ' + str(time.time() - start))


        # Condition on this emission
        # Compute the Kalman gain
        S = torch.diag(R) + pred_cov
        K = torch.linalg.solve(S, pred_cov).T
        filtered_cov = pred_cov - K @ S @ K.T
        filtered_mean = pred_mean + K @ (y - pred_mean)

        # Predict the next state
        pred_mean = dynamics_weights @ filtered_mean + input_weights * inputs[t]
        pred_cov = dynamics_weights @ filtered_cov @ dynamics_weights.T + torch.diag(dynamics_cov)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, init_mean, init_cov)
    (ll, _, _), (filtered_means, filtered_covs) = scan(_step, carry, torch.arange(num_timesteps))

    return ll, filtered_means, filtered_covs


def lgssm_filter_simple_batched(init_mean, init_cov, dynamics_weights, input_weights,
                                dynamics_cov, emissions_cov, emissions,
                                inputs=None, nan_fill=1e8):
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
    adapted from Dynamax.
    This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions

    Args:
        params: model parameters
        emissions: array of observations.
        inputs: optional array of inputs.
        nan_fill: value to put in place of nans in the emissions covariance diagonal

    Returns:
        PosteriorGSSMFiltered: filtered posterior object
    """
    dtype = emissions.dtype
    device = emissions.device
    num_timesteps = len(emissions)

    if inputs is None:
        inputs = torch.zeros((num_timesteps, 0), dtype=dtype, device=device)

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        y = emissions[t]

        # added by msc to attempt to perform inference over nans
        nan_loc = torch.isnan(y)
        y = torch.where(nan_loc, 0, y)
        R = torch.where(nan_loc, nan_fill, emissions_cov)

        # Update the log likelihood
        mu = pred_mean
        cov = pred_cov + torch.diag(R)

        ll += torch.distributions.MultivariateNormal(mu, cov).log_prob(y)

        # Condition on this emission
        # Compute the Kalman gain
        S = torch.diag(R) + pred_cov
        K = torch.linalg.solve(S, pred_cov).T
        filtered_cov = pred_cov - K @ S @ K.T
        filtered_mean = pred_mean + K @ (y - pred_mean)

        # Predict the next state
        pred_mean = dynamics_weights @ filtered_mean + input_weights * inputs
        pred_cov = dynamics_weights @ filtered_cov @ dynamics_weights.T + torch.diag(dynamics_cov)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, init_mean, init_cov)
    (ll, _, _), (filtered_means, filtered_covs) = scan(_step, carry, torch.arange(num_timesteps))

    return ll, filtered_means, filtered_covs


def lgssm_filter(params, emissions, inputs=None):
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
    adapted from Dynamax

    Args:
        params: model parameters
        emissions: array of observations.
        inputs: optional array of inputs.

    Returns:
        PosteriorGSSMFiltered: filtered posterior object
    """
    dtype = emissions.dtype
    device = emissions.device
    num_timesteps = len(emissions)

    if inputs is None:
        inputs = torch.zeros((num_timesteps, 0), dtype=dtype, device=device)

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        F = params['dynamics']['weights']
        B = params['dynamics']['input_weights']
        b = params['dynamics']['bias']
        Q = params['dynamics']['cov']
        H = params['emissions']['weights']
        D = params['emissions']['input_weights']
        d = params['emissions']['bias']
        R = params['emissions']['cov']
        u = inputs[t]
        y = emissions[t]

        # added by msc to attempt to perform inference over nans
        nan_loc = torch.isnan(y)
        y = torch.where(nan_loc, 0, y)
        R = torch.where(torch.diag(nan_loc), 1e8, R)

        # Update the log likelihood
        mu = H @ pred_mean + D @ u + d
        cov = H @ pred_cov @ H.T + R
        ll += torch.distributions.MultivariateNormal(mu, cov).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, F, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, params['initial']['mean'], params['initial']['cov'])
    (ll, _, _), (filtered_means, filtered_covs) = scan(_step, carry, torch.arange(num_timesteps))

    return ll, filtered_means, filtered_covs


def lgssm_smoother(params, emissions, inputs=None):
    """Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.
    adapted from Dynamax

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        emissions: array of observations.
        inputs: array of inputs.

    Returns:
        PosteriorGSSMSmoothed: smoothed posterior object.

    """
    num_timesteps = len(emissions)
    model_dim = emissions.shape[1]
    dtype = emissions.dtype
    device = emissions.device

    if inputs is None:
        inputs = torch.zeros((num_timesteps, 0), dtype=dtype, device=device)

    # Run the Kalman filter
    ll, filtered_means, filtered_covs = lgssm_filter(params, emissions, inputs)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        F = params['dynamics']['weights']
        B = params['dynamics']['input_weights']
        b = params['dynamics']['bias']
        Q = params['dynamics']['cov']
        u = inputs[t]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = torch.linalg.solve(Q + F @ filtered_cov @ F.T, F @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - F @ filtered_mean - B @ u - b)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - F @ filtered_cov @ F.T - Q) @ G.T

        # Compute the smoothed expectation of x_t x_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + torch.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    filtered_means_flipped = torch.flip(filtered_means[:-1], [0])
    filtered_covs_flipped = torch.flip(filtered_covs[:-1], [0])
    args = (torch.arange(num_timesteps - 2, -1, -1), filtered_means_flipped, filtered_covs_flipped)
    _, (smoothed_means, smoothed_covs, smoothed_cross) = scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = torch.flip(smoothed_means, [0])
    smoothed_covs = torch.flip(smoothed_covs, [0])
    smoothed_cross = torch.flip(smoothed_cross, [0])
    smoothed_means = torch.row_stack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = torch.row_stack((smoothed_covs, filtered_covs[-1][None, ...]))

    return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_cross


def _condition_on(m, P, H, D, d, R, u, y):
    r"""Condition a Gaussian potential on a new linear Gaussian observation
       p(x_t \mid y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t \mid y_{1:t-1}, u_{1:t-1}) p(y_t \mid x_t, u_t)
         = N(x_t \mid m, P) N(y_t \mid H_t x_t + D_t u_t + d_t, R_t)
         = N(x_t \mid mm, PP)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = H*m + D*u + d
         S = (R + H * P * H')
         K = P * H' * S^{-1}
         PP = P - K S K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         H (D_obs,D_hid): emission matrix.
         D (D_obs,D_in): emission input weights.
         u (D_in,): inputs.
         d (D_obs,): emission bias.
         R (D_obs,D_obs): emission covariance matrix.
         y (D_obs,): observation.

     Returns:
         mu_pred (D_hid,): predicted mean.
         Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    # Compute the Kalman gain
    S = R + H @ P @ H.T
    K = torch.linalg.solve(S, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - D @ u - d - H @ m)

    return mu_cond, Sigma_cond


def _predict(m, S, F, B, b, Q, u):
    r"""Predict next mean and covariance under a linear Gaussian model.

        p(x_{t+1}) = int N(x_t \mid m, S) N(x_{t+1} \mid Fx_t + Bu + b, Q)
                    = N(x_{t+1} \mid Fm + Bu, F S F^T + Q)

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        F (D_hid,D_hid): dynamics matrix.
        B (D_hid,D_in): dynamics input matrix.
        u (D_in,): inputs.
        Q (D_hid,D_hid): dynamics covariance matrix.
        b (D_hid,): dynamics bias.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    mu_pred = F @ m + B @ u + b
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred


def psd_check(a):
    eigs_are_real = torch.all(torch.isreal(torch.linalg.eigvals(a)))
    vals_are_positive = torch.all(torch.real(torch.linalg.eigvals(a)) > 0)
    symmetric = torch.isclose(a, a.mT, atol=1e-6).all(-2).all(-1)
    cholesky = torch.linalg.cholesky_ex(a).info.eq(0)

    return torch.tensor((eigs_are_real, vals_are_positive, symmetric, cholesky))


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    if type(xs) is not tuple and type(xs) is not list:
        xs = [xs]

    for x in range(xs[0].shape[0]):
        args = []
        for i in xs:
            args.append(i[x])

        carry, y = f(carry, args)
        ys.append(y)

    out = []
    for j in range(len(ys[0])):
        var = []

        for i in ys:
            var.append(i[j])

        out.append(torch.stack(var))

    return carry, out


def nearest_psd(a, precision=1e-8):
    # assume that a is nearly symmetric
    [eig_vals, eig_vec] = torch.linalg.eigh(a)
    # set all eig values to be positive
    torch.where(eig_vals < precision, precision, eig_vals)

    return eig_vec @ torch.diag(eig_vals) @ eig_vec.T

