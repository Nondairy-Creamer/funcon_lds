import torch
import time
import utilities as uu


def lgssm_filter(params, emissions, inputs, init_mean, init_cov, nan_fill=1e8):
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
    adapted from Dynamax.
    This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """
    dynamics_weights = params['dynamics']['weights']
    dynamics_input_weights = params['dynamics']['input_weights']
    dynamics_offset = params['dynamics']['offset']
    dynamics_cov = params['dynamics']['cov']

    emissions_weights = params['emissions']['weights']
    emissions_input_weights = params['emissions']['input_weights']
    emissions_offset = params['emissions']['offset']
    emissions_cov = params['emissions']['cov']

    num_data_sets, num_timesteps, _ = emissions.shape
    device = dynamics_weights.device
    dtype = dynamics_weights.dtype

    ll = torch.zeros(num_data_sets, device=device, dtype=dtype)
    pred_mean = init_mean
    pred_cov = init_cov

    dynamics_inputs = (dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]
    emissions_inputs = (emissions_input_weights @ inputs[:, :, :, None])[:, :, :, 0]

    filtered_means_list = []
    filtered_covs_list = []

    for t in range(num_timesteps):
        # Shorthand: get parameters and input for time index t
        y = emissions[:, t, :]

        # locate nans and set covariance at their location to a large number to marginalize over them
        nan_loc = torch.isnan(y)
        y = torch.where(nan_loc, 0, y)
        R = torch.where(torch.diag_embed(nan_loc), nan_fill, torch.tile(emissions_cov, (num_data_sets, 1, 1)))

        # Update the log likelihood
        ll_mu = (emissions_weights @ pred_mean[:, :, None])[:, :, 0] + emissions_inputs[:, t, :] + emissions_offset[None, :]
        ll_cov = emissions_weights @ pred_cov @ emissions_weights.T + R

        ll += torch.distributions.multivariate_normal.MultivariateNormal(ll_mu, ll_cov).log_prob(y)

        # Condition on this emission
        # Compute the Kalman gain
        S = R + emissions_weights @ pred_cov @ emissions_weights.T
        K = uu.batch_trans(torch.linalg.solve(S, emissions_weights @ pred_cov))

        filtered_cov = pred_cov - K @ S @ uu.batch_trans(K)
        filtered_mean = pred_mean + (K @ (y - ll_mu)[:, :, None])[:, :, 0]

        filtered_covs_list.append(filtered_cov)
        filtered_means_list.append(filtered_mean)

        # Predict the next state
        pred_mean = (dynamics_weights @ filtered_mean[:, :, None])[:, :, 0] + dynamics_inputs[:, t, :] + dynamics_offset
        pred_cov = dynamics_weights @ filtered_cov @ dynamics_weights.T + dynamics_cov

    filtered_means = torch.permute(torch.stack(filtered_means_list), (1, 0, 2))
    filtered_covs = torch.permute(torch.stack(filtered_covs_list), (1, 0, 2, 3))

    ll = torch.sum(ll) / emissions.numel()

    return ll, filtered_means, filtered_covs


def lgssm_smoother(params, emissions, inputs, init_mean, init_cov):
    r"""Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.
    Adopted from Dynamax
    """
    dynamics_weights = params['dynamics']['weights']
    dynamics_input_weights = params['dynamics']['input_weights']
    dynamics_offset = params['dynamics']['offset']
    dynamics_cov = params['dynamics']['cov']

    emissions_weights = params['emissions']['weights']
    emissions_input_weights = params['emissions']['input_weights']
    emissions_offset = params['emissions']['offset']
    emissions_cov = params['emissions']['cov']

    num_timesteps = emissions.shape[1]

    # Run the Kalman filter
    ll, filtered_means, filtered_covs = lgssm_filter(params, emissions, inputs, init_mean, init_cov)

    dynamics_inputs = (dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]
    emissions_inputs = (emissions_input_weights @ inputs[:, :, :, None])[:, :, :, 0]

    smoothed_mean_next = filtered_means[:, -1, :]
    smoothed_cov_next = filtered_covs[:, -1, :, :]

    smoothed_means_list = []
    smoothed_covs_list = []
    smoothed_cross_list = []

    # Run the smoother backward in time
    for t in reversed(range(num_timesteps - 1)):
        # Unpack the input
        filtered_mean = filtered_means[:, t, :]
        filtered_cov = filtered_covs[:, t, :, :]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        a = dynamics_cov + dynamics_weights @ filtered_cov @ dynamics_weights.T
        b = dynamics_weights @ filtered_cov
        G = uu.batch_trans(torch.linalg.solve(a, b))

        # Compute the smoothed mean and covariance
        pred_mean = (dynamics_weights @ filtered_mean[:, :, None])[:, :, 0] + dynamics_inputs[:, t, :] + dynamics_offset
        smoothed_mean = (filtered_mean[:, :, None] + G @ (smoothed_mean_next - pred_mean)[:, :, None])[:, :, 0]
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - dynamics_weights @ filtered_cov @ dynamics_weights.T - dynamics_cov) @ uu.batch_trans(G)

        # Compute the smoothed expectation of x_t x_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + smoothed_mean[:, :, None] * smoothed_mean_next[:, None, :]

        smoothed_means_list.append(smoothed_mean)
        smoothed_covs_list.append(smoothed_cov)
        smoothed_cross_list.append(smoothed_cross)

    # Reverse the arrays and return
    smoothed_means_reversed = torch.permute(torch.stack(list(reversed(smoothed_means_list))), (1, 0, 2))
    smoothed_covs_reversed = torch.permute(torch.stack(list(reversed(smoothed_covs_list))), (1, 0, 2, 3))
    smoothed_means = torch.cat((smoothed_means_reversed, filtered_means[:, -1, None, :]), dim=1)
    smoothed_covs = torch.cat((smoothed_covs_reversed, filtered_covs[:, -1, None, :, :]), dim=1)
    smoothed_crosses = torch.permute(torch.stack(list(reversed(smoothed_cross_list))), (1, 0, 2, 3))

    return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses


def e_step(params, emissions, input, init_mean, init_cov):
    num_data_sets = emissions.shape[0]
    num_timesteps = emissions.shape[1]

    # Run the smoother to get posterior expectations
    ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses = \
        lgssm_smoother(params, emissions, input, init_mean, init_cov)

    # shorthand
    Ex = smoothed_means
    Exp = smoothed_means[:, :-1, :]
    Exn = smoothed_means[:, 1:, :]
    Vx = smoothed_covs
    Vxp = smoothed_covs[:, :-1, :, :]
    Vxn = smoothed_covs[:, 1:, :, :]
    Expxn = smoothed_crosses

    # Append bias to the input
    input = torch.cat((input, torch.ones((num_data_sets, num_timesteps, 1))), dim=2)
    up = input[:, :-1, :]
    u = input
    y = emissions
    y = torch.where(torch.isnan(y), 0, y)

    # expected sufficient statistics for the initial tfd.Distribution
    Ex0 = smoothed_means[:, 0, :]
    Ex0x0T = smoothed_covs[:, 0, :] + Ex0[:, :, None] @ Ex0[:, None, :]
    init_stats = (Ex0, Ex0x0T, torch.tensor(1, dtype=self.dtype, device=self.device))

    # expected sufficient statistics for the dynamics tfd.Distribution
    # let zp[t] = [x[t], u[t]] for t = 0...T-2
    # let xn[t] = x[t+1]          for t = 0...T-2
    m11 = uu.batch_trans(Exp) @ Exp
    m12 = uu.batch_trans(Exp) @ up
    m21 = uu.batch_trans(up) @ Exp
    m22 = uu.batch_trans(up) @ up
    sum_zpzpT = uu.block([[m11, m12], [m21, m22]])
    sum_zpzpT[:, :self.dynamics_dim, :self.dynamics_dim] += Vxp.sum(1)
    sum_zpxnT = uu.block([[Expxn.sum(1)], [uu.batch_trans(up) @ Exn]])
    sum_xnxnT = Vxn.sum(1) + uu.batch_trans(Exn) @ Exn
    dynamics_stats = (sum_zpzpT[:, :-1, :-1], sum_zpxnT[:, :-1, :], sum_xnxnT,
                      torch.tensor(num_timesteps - 1, dtype=self.dtype, device=self.device))

    # more expected sufficient statistics for the emissions
    # let z[t] = [x[t], u[t]] for t = 0...T-1
    n11 = uu.batch_trans(Ex) @ Ex
    n12 = uu.batch_trans(Ex) @ u
    n21 = uu.batch_trans(u) @ Ex
    n22 = uu.batch_trans(u) @ u
    sum_zzT = uu.block([[n11, n12], [n21, n22]])
    sum_zzT[:, :self.dynamics_dim, :self.dynamics_dim] += Vx.sum(1)
    sum_zyT = uu.block([[uu.batch_trans(Ex) @ y], [uu.batch_trans(u) @ y]])
    sum_yyT = uu.batch_trans(y) @ y
    emission_stats = (sum_zzT[:, :-1, :-1], sum_zyT[:, :-1, :], sum_yyT,
                      torch.tensor(num_timesteps, dtype=self.dtype, device=self.device))

    return ll, init_stats, dynamics_stats, emission_stats


def m_step(self, init_stats, dynamics_stats, emission_stats):
    def fit_linear_regression(ExxT, ExyT, EyyT, N):
        # Solve a linear regression given sufficient statistics
        W = torch.linalg.solve(ExxT, ExyT).T

        Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
        return W, Sigma

    # Sum the statistics across all batches
    init_stats = [i.sum(0) for i in init_stats]
    dynamics_stats = [i.sum(0) for i in dynamics_stats]
    emission_stats = [i.sum(0) for i in emission_stats]

    # Perform MLE estimation jointly
    sum_x0, sum_x0x0T, N = init_stats
    S = (sum_x0x0T - torch.outer(sum_x0, sum_x0)) / N
    m = sum_x0 / N

    FB, self.dynamics_cov = fit_linear_regression(*dynamics_stats)
    self.dynamics_weights = FB[:, :self.dynamics_dim]
    self.input_weights, b = (FB[:, self.dynamics_dim:], None)

    HD, self.emissions_cov = fit_linear_regression(*emission_stats)
    H = HD[:, :self.dynamics_dim]
    D, d = (HD[:, self.dynamics_dim:], None)


def fit_gd(self, emissions_list, input_list, learning_rate=1e-2, num_steps=50):
    """ This function will fit the model using gradient descent on the entire data set
    """
    self.dynamics_weights.requires_grad = True
    self.input_weights.requires_grad = True
    self.dynamics_cov.requires_grad = True
    self.emissions_cov.requires_grad = True

    emissions, input = self.standardize_input(emissions_list, input_list)
    init_mean, init_cov = self.estimate_init(emissions)

    # initialize the optimizer with the parameters we're going to optimize
    model_params = [self.dynamics_weights, self.input_weights, self.dynamics_cov, self.emissions_cov]

    optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    log_likelihood_out = []
    time_out = []

    start = time.time()
    for ep in range(num_steps):
        optimizer.zero_grad()
        loss = -self.lgssm_filter(emissions, input, init_mean, init_cov)[0]
        loss.backward()
        log_likelihood_out.append(-loss.detach().cpu().numpy())
        optimizer.step()

        time_out.append(time.time() - start)

        if self.verbose:
            print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
            print('log likelihood = ' + str(log_likelihood_out[-1]))
            print('Time elapsed = ' + str(time_out[-1]))

    self.log_likelihood = log_likelihood_out
    self.train_time = time_out

    self.dynamics_weights.requires_grad = False
    self.input_weights.requires_grad = False
    self.dynamics_cov.requires_grad = False
    self.emissions_cov.requires_grad = False