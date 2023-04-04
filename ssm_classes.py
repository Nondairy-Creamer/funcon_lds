import torch
import numpy as np
import pickle
import time


class Lgssm:
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """

    def __init__(self, latent_dim, dtype=torch.float64, device='cpu', verbose=True):
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.log_likelihood = None
        self.train_time = None

        self.dynamics_weights_init = 0.9 * np.eye(self.latent_dim)
        self.inputs_weights_init = np.zeros((self.latent_dim, self.latent_dim))
        self.dynamics_cov_init = np.eye(self.latent_dim)
        self.emissions_cov_init = np.eye(self.latent_dim)

        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device, dtype=self.dtype)
        self.inputs_weights = torch.tensor(self.inputs_weights_init, device=self.device, dtype=self.dtype)
        self.dynamics_cov = torch.tensor(self.dynamics_cov_init, device=self.device, dtype=self.dtype)
        self.emissions_cov = torch.tensor(self.emissions_cov_init, device=self.device, dtype=self.dtype)

    def save(self, path='trained_models/trained_model.pkl'):
        save_file = open(path, 'wb')
        pickle.dump(self, save_file)
        save_file.close()

    def randomize_weights(self, max_eig_allowed=0.8, init_std=0.1, random_seed=None):
        rng = np.random.default_rng(random_seed)

        self.dynamics_weights_init = rng.standard_normal((self.latent_dim, self.latent_dim))
        max_eig_in_mat = np.max(np.abs(np.linalg.eigvals(self.dynamics_weights_init)))
        self.dynamics_weights_init = max_eig_allowed * self.dynamics_weights_init / max_eig_in_mat

        self.inputs_weights_init = init_std * rng.standard_normal((self.latent_dim, self.latent_dim))
        self.dynamics_cov_init = rng.standard_normal((self.latent_dim, self.latent_dim))
        self.dynamics_cov_init = init_std * self.dynamics_cov_init.T @ self.dynamics_cov_init / self.latent_dim
        self.emissions_cov_init = rng.standard_normal((self.latent_dim, self.latent_dim))
        self.emissions_cov_init = init_std * self.emissions_cov_init.T @ self.emissions_cov_init / self.latent_dim

        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device, dtype=self.dtype)
        self.inputs_weights = torch.tensor(self.inputs_weights_init, device=self.device, dtype=self.dtype)
        self.dynamics_cov = torch.tensor(self.dynamics_cov_init, device=self.device, dtype=self.dtype)
        self.emissions_cov = torch.tensor(self.emissions_cov_init, device=self.device, dtype=self.dtype)

    def set_device(self, new_device):
        self.device = new_device
        self.dynamics_weights = self.dynamics_weights.to(new_device)
        self.inputs_weights = self.inputs_weights.to(new_device)
        self.dynamics_cov = self.dynamics_cov.to(new_device)
        self.emissions_cov = self.emissions_cov.to(new_device)

    def sample(self, num_time=100, num_data_sets=None, nan_freq=0.0, random_seed=None):
        rng = np.random.default_rng(random_seed)

        dynamics_weights = self.dynamics_weights.detach().cpu().numpy().copy()
        inputs_weights = self.inputs_weights.detach().cpu().numpy().copy()
        dynamics_cov = self.dynamics_cov.detach().cpu().numpy().copy()
        emissions_cov = self.emissions_cov.detach().cpu().numpy().copy()

        # generate a random initial mean and covariance
        init_mean = rng.standard_normal((num_data_sets, self.latent_dim))
        init_cov = rng.standard_normal((num_data_sets, self.latent_dim, self.latent_dim))
        init_cov = np.transpose(init_cov, [0, 2, 1]) @ init_cov / self.latent_dim

        latents = np.zeros((num_data_sets, num_time, self.latent_dim))
        emissions = np.zeros((num_data_sets, num_time, self.latent_dim))

        inputs = rng.standard_normal((num_data_sets, num_time, self.latent_dim))

        # get the initial observations
        emissions_noise_init = rng.multivariate_normal(np.zeros(self.latent_dim), emissions_cov, size=num_data_sets)
        dynamics_noise_init = rng.multivariate_normal(np.zeros(self.latent_dim), emissions_cov, size=num_data_sets)
        for d in range(num_data_sets):
            latent_init = rng.multivariate_normal(init_mean[d, :], init_cov[d, :, :])

            latents[d, 0, :] = (dynamics_weights @ latent_init[:, None] + \
                               inputs_weights @ inputs[d, 0, :, None] + dynamics_noise_init[d, :, None])[:, 0]
            emissions[d, 0, :] = latents[d, 0, :] + emissions_noise_init[d, :]

        for t in range(1, num_time):
            dynamics_noise = rng.multivariate_normal(np.zeros(self.latent_dim), dynamics_cov, size=num_data_sets)
            emissions_noise = rng.multivariate_normal(np.zeros(self.latent_dim), emissions_cov, size=num_data_sets)

            latents[:, t, :] = (dynamics_weights[None, :, :] @ latents[:, t - 1, :, None] +
                               inputs_weights[None, :, :] @ inputs[:, t, :, None] + dynamics_noise[:, :, None])[:, :, 0]
            emissions[:, t, :] = latents[:, t, :] + emissions_noise

        # add in nans
        nan_mask = rng.random((num_data_sets, num_time, self.latent_dim)) <= nan_freq
        emissions[nan_mask] = np.nan

        emissions = [i for i in emissions]
        inputs = [i for i in inputs]
        init_mean = [i for i in init_mean]
        init_cov = [i for i in init_cov]

        data_dict = {'emissions': emissions,
                     'inputs': inputs,
                     'latents': latents,
                     'init_mean': init_mean,
                     'init_cov': init_cov,
                     }

        return data_dict

    def lgssm_filter(self, emissions, inputs, init_mean, init_cov, nan_fill=1e8):
        """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
        """
        num_data_sets, num_timesteps, _ = emissions.shape

        ll = torch.zeros(num_data_sets, device=self.device, dtype=self.dtype)
        pred_mean = init_mean
        pred_cov = init_cov

        filtered_means_list = []
        filtered_covs_list = []

        for t in range(num_timesteps):
            # Shorthand: get parameters and inputs for time index t
            y = emissions[:, t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)
            R = torch.where(torch.diag_embed(nan_loc), nan_fill, torch.tile(self.emissions_cov, (num_data_sets, 1, 1)))

            # Update the log likelihood
            ll_mu = pred_mean
            ll_cov = pred_cov + R

            y_mean_sub = y - ll_mu
            const = -y.shape[1] * torch.log(2 * torch.tensor(torch.pi, device=self.device, dtype=self.dtype))
            ll += const + -torch.linalg.slogdet(ll_cov)[1] - \
                  (y_mean_sub[:, None, :] @ torch.linalg.solve(ll_cov, y_mean_sub[:, :, None]))[:, 0, 0]

            # Condition on this emission
            # Compute the Kalman gain
            K = self._batch_trans(torch.linalg.solve(ll_cov, pred_cov))

            filtered_cov = pred_cov - K @ ll_cov @ self._batch_trans(K)
            filtered_mean = pred_mean + (K @ y_mean_sub[:, :, None])[:, :, 0]

            filtered_covs_list.append(filtered_cov)
            filtered_means_list.append(filtered_mean)

            # Predict the next state
            pred_mean = (self.dynamics_weights[None, :, :] @ filtered_mean[:, :, None] +
                        self.inputs_weights[None, :, :] @ inputs[:, t, :, None])[:, :, 0]
            pred_cov = self.dynamics_weights[None, :, :] @ filtered_cov @ self.dynamics_weights.T[None, :, :] + \
                       self.dynamics_cov[None, :, :]

        filtered_means = torch.permute(torch.stack(filtered_means_list), (1, 0, 2))
        filtered_covs = torch.permute(torch.stack(filtered_covs_list), (1, 0, 2, 3))

        # TODO go back to normalizing the loss
        ll = torch.sum(ll) #/ emissions.numel()

        return ll, filtered_means, filtered_covs

    def lgssm_smoother(self, emissions, inputs, init_mean, init_cov):
        r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states. Technically, this
        implements the Rauch-Tung-Striebel (RTS) smoother.
        Adopted from Dynamax
        """
        num_timesteps = emissions.shape[1]

        # Run the Kalman filter
        ll, filtered_means, filtered_covs = self.lgssm_filter(emissions, inputs, init_mean, init_cov)

        smoothed_mean_next = filtered_means[:, -1, :]
        smoothed_cov_next = filtered_covs[:, -1, :, :]

        smoothed_means_list = []
        smoothed_covs_list = []
        smoothed_cross_list = []

        # Run the smoother backward in time
        for t in reversed(range(num_timesteps-1)):
            # Unpack the inputs
            filtered_mean = filtered_means[:, t, :]
            filtered_cov = filtered_covs[:, t, :, :]

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            a = self.dynamics_cov[None, :, :] + self.dynamics_weights[None, :, :] @ filtered_cov @ self.dynamics_weights.T[None, :, :]
            b = self.dynamics_weights[None, :, :] @ filtered_cov
            G = self._batch_trans(torch.linalg.solve(a, b))

            # Compute the smoothed mean and covariance
            smoothed_mean = filtered_mean[:, :, None] + G @ (smoothed_mean_next[:, :, None] - self.dynamics_weights[None, :, :] @ filtered_mean[:, :, None] - self.inputs_weights[None, :, :] @ inputs[:, t, :, None])
            smoothed_cov = filtered_cov + G @ (smoothed_cov_next - self.dynamics_weights[None, :, :] @ filtered_cov @ self.dynamics_weights.T[None, :, :] - self.dynamics_cov[None, :, :]) @ self._batch_trans(G)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            smoothed_cross = G @ smoothed_cov_next + smoothed_mean @ smoothed_mean_next[:, None, :]

            smoothed_means_list.append(smoothed_mean[:, :, 0])
            smoothed_covs_list.append(smoothed_cov)
            smoothed_cross_list.append(smoothed_cross)

        # Reverse the arrays and return
        smoothed_means_reversed = torch.permute(torch.stack(list(reversed(smoothed_means_list))), (1, 0, 2))
        smoothed_covs_reversed = torch.permute(torch.stack(list(reversed(smoothed_covs_list))), (1, 0, 2, 3))
        smoothed_means = torch.cat((smoothed_means_reversed, filtered_means[:, -1, None, :]), dim=1)
        smoothed_covs = torch.cat((smoothed_covs_reversed, filtered_covs[:, -1, None, :, :]), dim=1)
        smoothed_crosses = torch.permute(torch.stack(list(reversed(smoothed_cross_list))), (1, 0, 2, 3))

        return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses

    def fit_gd(self, emissions_list, inputs_list, learning_rate=1e-2, num_steps=50):
        """ This function will fit the model using gradient descent on the entire data set
        """
        self.dynamics_weights.requires_grad = True
        self.inputs_weights.requires_grad = True
        self.dynamics_cov.requires_grad = True
        self.emissions_cov.requires_grad = True

        emissions, inputs = self.standardize_inputs(emissions_list, inputs_list)
        init_mean, init_cov = self.estimate_init(emissions)

        # initialize the optimizer with the parameters we're going to optimize
        model_params = [self.dynamics_weights, self.inputs_weights, self.dynamics_cov, self.emissions_cov]

        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        log_likelihood_out = []
        time_out = []

        start = time.time()
        for ep in range(num_steps):
            optimizer.zero_grad()
            loss = -self.lgssm_filter(emissions, inputs, init_mean, init_cov)[0]
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
        self.inputs_weights.requires_grad = False
        self.dynamics_cov.requires_grad = False
        self.emissions_cov.requires_grad = False

    def fit_em(self, emissions_list, inputs_list, num_steps=10):
        emissions, inputs = self.standardize_inputs(emissions_list, inputs_list)
        init_mean, init_cov = self.estimate_init(emissions)

        log_likelihood_out = []
        time_out = []

        start = time.time()
        for ep in range(num_steps):
            ll, init_stats, dynamics_stats, emission_stats = self.e_step(emissions, inputs, init_mean, init_cov)
            self.m_step(init_stats, dynamics_stats, emission_stats)

            log_likelihood_out.append(ll.detach().cpu().numpy())
            time_out.append(time.time() - start)

            if self.verbose:
                print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
                print('log likelihood = ' + str(log_likelihood_out[-1]))
                print('Time elapsed = ' + str(time_out[-1]))

        self.log_likelihood = log_likelihood_out
        self.train_time = time_out

    def e_step(self, emissions, inputs, init_mean, init_cov):
        num_data_sets = emissions.shape[0]
        num_timesteps = emissions.shape[1]

        # Run the smoother to get posterior expectations
        ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses = \
            self.lgssm_smoother(emissions, inputs, init_mean, init_cov)

        # shorthand
        Ex = smoothed_means
        Exp = smoothed_means[:, :-1, :]
        Exn = smoothed_means[:, 1:, :]
        Vx = smoothed_covs
        Vxp = smoothed_covs[:, :-1, :, :]
        Vxn = smoothed_covs[:, 1:, :, :]
        Expxn = smoothed_crosses

        # Append bias to the inputs
        inputs = torch.cat((inputs, torch.ones((num_data_sets, num_timesteps, 1))), dim=2)
        up = inputs[:, :-1, :]
        u = inputs
        y = emissions
        y = torch.where(torch.isnan(y), 0, y)

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = smoothed_means[:, 0, :]
        Ex0x0T = smoothed_covs[:, 0, :] + Ex0[:, :, None] @ Ex0[:, None, :]
        init_stats = (Ex0, Ex0x0T, torch.tensor(1, dtype=self.dtype, device=self.device))

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        m11 = self._batch_trans(Exp) @ Exp
        m12 = self._batch_trans(Exp) @ up
        m21 = self._batch_trans(up) @ Exp
        m22 = self._batch_trans(up) @ up
        sum_zpzpT = self._block([[m11, m12], [m21, m22]])
        sum_zpzpT[:, :self.latent_dim, :self.latent_dim] += Vxp.sum(1)
        sum_zpxnT = self._block([[Expxn.sum(1)], [self._batch_trans(up) @ Exn]])
        sum_xnxnT = Vxn.sum(1) + self._batch_trans(Exn) @ Exn
        dynamics_stats = (sum_zpzpT[:, :-1, :-1], sum_zpxnT[:, :-1, :], sum_xnxnT,
                          torch.tensor(num_timesteps - 1, dtype=self.dtype, device=self.device))

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        n11 = self._batch_trans(Ex) @ Ex
        n12 = self._batch_trans(Ex) @ u
        n21 = self._batch_trans(u) @ Ex
        n22 = self._batch_trans(u) @ u
        sum_zzT = self._block([[n11, n12], [n21, n22]])
        sum_zzT[:, :self.latent_dim, :self.latent_dim] += Vx.sum(1)
        sum_zyT = self._block([[self._batch_trans(Ex) @ y], [self._batch_trans(u) @ y]])
        sum_yyT = self._batch_trans(y) @ y
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
        self.dynamics_weights = FB[:, :self.latent_dim]
        self.inputs_weights, b = (FB[:, self.latent_dim:], None)

        HD, self.emissions_cov = fit_linear_regression(*emission_stats)
        H = HD[:, :self.latent_dim]
        D, d = (HD[:, self.latent_dim:], None)

    def standardize_inputs(self, emissions_list, inputs_list):
        assert(type(emissions_list) is list)
        assert(type(inputs_list) is list)

        if emissions_list[0].dtype is not self.dtype:
            emissions_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions_list]
            inputs_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in inputs_list]

        emissions, inputs = self._stack_data(emissions_list, inputs_list)

        return emissions, inputs

    @staticmethod
    def _block(block_list, dims=(2, 1)):
        layer = []
        for i in block_list:
            layer.append(torch.cat(i, dim=dims[0]))

        return torch.cat(layer, dim=dims[1])

    @staticmethod
    def _batch_trans(batch_matrix):
        return torch.permute(batch_matrix, (0, 2, 1))

    @staticmethod
    def _get_batches_inds(num_data, batch_size, generator):
        num_batches = np.ceil(num_data / batch_size)
        data_inds = np.arange(num_data)
        generator.shuffle(data_inds)
        batch_data_inds = np.array_split(data_inds, num_batches)

        return batch_data_inds

    @staticmethod
    def estimate_init(emissions):
        init_mean = torch.nanmean(emissions, dim=1)
        init_cov = [LgssmSimple._estimate_cov(i)[None, :, :] for i in emissions]
        init_cov = torch.cat(init_cov, dim=0)

        return init_mean, init_cov

    @staticmethod
    def _stack_data(emissions_list, inputs_list):
        device = emissions_list[0].device
        dtype = inputs_list[0].dtype

        data_set_time = [i.shape[0] for i in emissions_list]
        max_time = np.max(data_set_time)
        num_data_sets = len(emissions_list)
        num_neurons = emissions_list[0].shape[1]

        emissions = torch.empty((num_data_sets, max_time, num_neurons), device=device, dtype=dtype)
        emissions[:] = torch.nan
        inputs = torch.zeros((num_data_sets, max_time, num_neurons), device=device, dtype=dtype)

        for d in range(num_data_sets):
            emissions[d, :data_set_time[d], :] = emissions_list[d]
            inputs[d, :data_set_time[d], :] = inputs_list[d]

        return emissions, inputs

    @staticmethod
    def _split_data(emissions, inputs, num_splits=2):
        # split the data in half for lower memory usage
        emissions_split = []
        inputs_split = []
        for ems, ins in zip(emissions, inputs):
            time_inds = np.arange(ems.shape[0])
            split_inds = np.array_split(time_inds, num_splits)

            for s in split_inds:
                emissions_split.append(ems[s, :])
                inputs_split.append(ins[s, :])

        return emissions_split, inputs_split

    @staticmethod
    def _estimate_cov(a):
        def nan_matmul(a, b, impute_val=0):
            a_no_nan = torch.where(torch.isnan(a), impute_val, a)
            b_no_nan = torch.where(torch.isnan(b), impute_val, b)

            return a_no_nan @ b_no_nan

        a_mean_sub = a - torch.nanmean(a, dim=0, keepdim=True)
        # estimate the covariance from the data in a
        cov = nan_matmul(a_mean_sub.T, a_mean_sub) / a.shape[0]

        # some columns will be all 0s due to missing data
        # replace those diagonals with the mean covariance
        cov_diag = torch.diag(cov)
        cov_diag_mean = torch.mean(cov_diag[cov_diag != 0])
        cov_diag = torch.where(cov_diag == 0, cov_diag_mean, cov_diag)

        cov[torch.eye(a.shape[1], dtype=torch.bool)] = cov_diag

        return cov


class LgssmSimple:
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """

    def __init__(self, latent_dim, dtype=torch.float64, device='cpu', verbose=True):
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.log_likelihood = None
        self.train_time = None

        self.dynamics_weights_init = 0.9 * np.eye(self.latent_dim)
        self.inputs_weights_log_init = np.zeros(self.latent_dim)
        self.dynamics_cov_log_init = np.zeros(self.latent_dim)
        self.emissions_cov_init = np.zeros(self.latent_dim)

        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device, dtype=self.dtype)
        self.inputs_weights_log = torch.tensor(self.inputs_weights_log_init, device=self.device, dtype=self.dtype)
        self.dynamics_cov_log = torch.tensor(self.dynamics_cov_log_init, device=self.device, dtype=self.dtype)
        self.emissions_cov_log = torch.tensor(self.emissions_cov_init, device=self.device, dtype=self.dtype)

    def save(self, path='trained_models/trained_model.pkl'):
        save_file = open(path, 'wb')
        pickle.dump(self, save_file)
        save_file.close()

    def randomize_weights(self, max_eig_allowed=0.8, init_std=0.1, cov_log_offset=-1, random_seed=None):
        rng = np.random.default_rng(random_seed)

        self.dynamics_weights_init = rng.standard_normal((self.latent_dim, self.latent_dim))
        max_eig_in_mat = np.max(np.abs(np.linalg.eigvals(self.dynamics_weights_init)))
        self.dynamics_weights_init = max_eig_allowed * self.dynamics_weights_init / max_eig_in_mat

        self.inputs_weights_log_init = init_std * rng.standard_normal(self.latent_dim) + cov_log_offset
        self.dynamics_cov_log_init = init_std * rng.standard_normal(self.latent_dim) + cov_log_offset
        self.emissions_cov_log_init = init_std * rng.standard_normal(self.latent_dim) + cov_log_offset

        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device, dtype=self.dtype)
        self.inputs_weights_log = torch.tensor(self.inputs_weights_log_init, device=self.device, dtype=self.dtype)
        self.dynamics_cov_log = torch.tensor(self.dynamics_cov_log_init, device=self.device, dtype=self.dtype)
        self.emissions_cov_log = torch.tensor(self.emissions_cov_log_init, device=self.device, dtype=self.dtype)

    def set_device(self, new_device):
        self.device = new_device
        self.dynamics_weights = self.dynamics_weights.to(new_device)
        self.inputs_weights_log = self.inputs_weights_log.to(new_device)
        self.dynamics_cov_log = self.dynamics_cov_log.to(new_device)
        self.emissions_cov_log = self.emissions_cov_log.to(new_device)

    def sample(self, num_time=100, num_data_sets=None, nan_freq=0.0, random_seed=None):
        rng = np.random.default_rng(random_seed)

        dynamics_weights = self.dynamics_weights.detach().cpu().numpy().copy()
        inputs_weights = np.exp(self.inputs_weights_log.detach().cpu().numpy().copy())
        dynamics_cov = np.exp(self.dynamics_cov_log.detach().cpu().numpy().copy())
        emissions_cov = np.exp(self.emissions_cov_log.detach().cpu().numpy().copy())

        # generate a random initial mean and covariance
        init_mean = rng.standard_normal((num_data_sets, self.latent_dim))
        init_cov = rng.standard_normal((num_data_sets, self.latent_dim, self.latent_dim))
        init_cov = np.transpose(init_cov, [0, 2, 1]) @ init_cov / self.latent_dim

        latents = np.zeros((num_data_sets, num_time, self.latent_dim))
        emissions = np.zeros((num_data_sets, num_time, self.latent_dim))

        inputs = rng.standard_normal((num_data_sets, num_time, self.latent_dim))

        # get the initial observations
        emissions_noise_init = rng.multivariate_normal(np.zeros(self.latent_dim), np.diag(emissions_cov),
                                                       size=num_data_sets)
        dynamics_noise_init = rng.multivariate_normal(np.zeros(self.latent_dim), np.diag(emissions_cov),
                                                      size=num_data_sets)
        for d in range(num_data_sets):
            latent_init = rng.multivariate_normal(init_mean[d, :], init_cov[d, :, :])

            latents[d, 0, :] = dynamics_weights @ latent_init + \
                               inputs_weights * inputs[d, 0, :] + dynamics_noise_init[d, :]
            emissions[d, 0, :] = latents[d, 0, :] + emissions_noise_init[d, :]

        for t in range(1, num_time):
            dynamics_noise = rng.standard_normal((num_data_sets, self.latent_dim)) * np.sqrt(dynamics_cov)[None, :]
            emissions_noise = rng.standard_normal((num_data_sets, self.latent_dim)) * np.sqrt(emissions_cov)[None, :]

            latents[:, t, :] = (dynamics_weights[None, :, :] @ latents[:, t - 1, :, None])[:, :, 0] + \
                               inputs_weights[None, :] * inputs[:, t, :] + dynamics_noise
            emissions[:, t, :] = latents[:, t, :] + emissions_noise

        # add in nans
        nan_mask = rng.random((num_data_sets, num_time, self.latent_dim)) <= nan_freq
        emissions[nan_mask] = np.nan

        emissions = [i for i in emissions]
        inputs = [i for i in inputs]
        init_mean = [i for i in init_mean]
        init_cov = [i for i in init_cov]

        data_dict = {'emissions': emissions,
                     'inputs': inputs,
                     'latents': latents,
                     'init_mean': init_mean,
                     'init_cov': init_cov,
                     }

        return data_dict

    def lgssm_filter(self, emissions, inputs, init_mean, init_cov, nan_fill=1e8):
        """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
        """
        num_data_sets, num_timesteps, _ = emissions.shape
        dynamics_weights = self.dynamics_weights
        inputs_weights = torch.exp(self.inputs_weights_log)
        dynamics_cov = torch.exp(self.dynamics_cov_log)
        emissions_cov = torch.exp(self.emissions_cov_log)

        ll = torch.zeros(num_data_sets, device=self.device, dtype=self.dtype)
        pred_mean = init_mean
        pred_cov = init_cov

        filtered_means_list = []
        filtered_covs_list = []

        for t in range(num_timesteps):
            # Shorthand: get parameters and inputs for time index t
            y = emissions[:, t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)
            R = torch.where(nan_loc, nan_fill, torch.tile(emissions_cov[None, :], (num_data_sets, 1)))

            # Update the log likelihood
            mu = pred_mean
            cov = pred_cov + torch.diag_embed(R)

            y_mean_sub = y - mu
            ll += -torch.linalg.slogdet(cov)[1] - \
                  (y_mean_sub[:, None, :] @ torch.linalg.solve(cov, y_mean_sub[:, :, None]))[:, 0, 0]

            # Condition on this emission
            # Compute the Kalman gain
            K = self._batch_trans(torch.linalg.solve(cov, pred_cov))

            filtered_cov = pred_cov - K @ cov @ self._batch_trans(K)
            filtered_mean = (pred_mean[:, :, None] + K @ y_mean_sub[:, :, None])[:, :, 0]

            filtered_covs_list.append(filtered_cov)
            filtered_means_list.append(filtered_mean)

            # Predict the next state
            pred_mean = (dynamics_weights[None, :, :] @ filtered_mean[:, :, None])[:, :, 0] + \
                        inputs_weights[None, :] * inputs[:, t, :]
            pred_cov = dynamics_weights[None, :, :] @ filtered_cov @ dynamics_weights.T[None, :, :] + \
                       torch.diag_embed(dynamics_cov[None, :])

        filtered_means = torch.permute(torch.stack(filtered_means_list), (1, 0, 2))
        filtered_covs = torch.permute(torch.stack(filtered_covs_list), (1, 0, 2, 3))

        ll = torch.sum(ll) / emissions.numel()

        return ll, filtered_means, filtered_covs

    def lgssm_smoother(self, emissions, inputs, init_mean, init_cov):
        r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states. Technically, this
        implements the Rauch-Tung-Striebel (RTS) smoother.
        Adopted from Dynamax
        """
        num_timesteps = emissions.shape[1]
        dynamics_weights = self.dynamics_weights
        inputs_weights = torch.diag(torch.exp(self.inputs_weights_log))
        dynamics_cov = torch.diag(torch.exp(self.dynamics_cov_log))

        # Run the Kalman filter
        ll, filtered_means, filtered_covs = self.lgssm_filter(emissions, inputs, init_mean, init_cov)

        smoothed_mean_next = filtered_means[:, -1, :]
        smoothed_cov_next = filtered_covs[:, -1, :, :]

        smoothed_means_list = []
        smoothed_covs_list = []
        smoothed_cross_list = []

        # Run the smoother backward in time
        for t in reversed(range(num_timesteps-1)):
            # Unpack the inputs
            filtered_mean = filtered_means[:, t, :]
            filtered_cov = filtered_covs[:, t, :, :]

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            a = dynamics_cov[None, :, :] + dynamics_weights[None, :, :] @ filtered_cov @ dynamics_weights.T[None, :, :]
            b = dynamics_weights[None, :, :] @ filtered_cov
            G = self._batch_trans(torch.linalg.solve(a, b))

            # Compute the smoothed mean and covariance
            smoothed_mean = filtered_mean[:, :, None] + G @ (smoothed_mean_next[:, :, None] - dynamics_weights[None, :, :] @ filtered_mean[:, :, None] - inputs_weights[None, :, :] @ inputs[:, t, :, None])
            smoothed_cov = filtered_cov + G @ (smoothed_cov_next - dynamics_weights[None, :, :] @ filtered_cov @ dynamics_weights.T[None, :, :] - dynamics_cov[None, :, :]) @ self._batch_trans(G)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            smoothed_cross = G @ smoothed_cov_next + smoothed_mean @ smoothed_mean_next[:, None, :]

            smoothed_means_list.append(smoothed_mean[:, :, 0])
            smoothed_covs_list.append(smoothed_cov)
            smoothed_cross_list.append(smoothed_cross)

        # Reverse the arrays and return
        smoothed_means_reversed = torch.permute(torch.stack(list(reversed(smoothed_means_list))), (1, 0, 2))
        smoothed_covs_reversed = torch.permute(torch.stack(list(reversed(smoothed_covs_list))), (1, 0, 2, 3))
        smoothed_means = torch.cat((smoothed_means_reversed, filtered_means[:, -1, None, :]), dim=1)
        smoothed_covs = torch.cat((smoothed_covs_reversed, filtered_covs[:, -1, None, :, :]), dim=1)
        smoothed_crosses = torch.permute(torch.stack(list(reversed(smoothed_cross_list))), (1, 0, 2, 3))

        return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses

    def fit_gd(self, emissions_list, inputs_list, learning_rate=1e-2, num_steps=50):
        """ This function will fit the model using gradient descent on the entire data set
        """
        self.dynamics_weights.requires_grad = True
        self.inputs_weights_log.requires_grad = True
        self.dynamics_cov_log.requires_grad = True
        self.emissions_cov_log.requires_grad = True

        emissions, inputs = self.standardize_inputs(emissions_list, inputs_list)
        init_mean, init_cov = self.estimate_init(emissions)

        # initialize the optimizer with the parameters we're going to optimize
        model_params = [self.dynamics_weights, self.inputs_weights_log, self.dynamics_cov_log, self.emissions_cov_log]

        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        log_likelihood_out = []
        time_out = []

        start = time.time()
        for ep in range(num_steps):
            optimizer.zero_grad()
            loss = -self.lgssm_filter(emissions, inputs, init_mean, init_cov)[0]
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
        self.inputs_weights_log.requires_grad = False
        self.dynamics_cov_log.requires_grad = False
        self.emissions_cov_log.requires_grad = False

    def fit_em(self, emissions_list, inputs_list, num_steps=10):
        emissions, inputs = self.standardize_inputs(emissions_list, inputs_list)
        init_mean, init_cov = self.estimate_init(emissions)

        log_likelihood_out = []
        time_out = []

        start = time.time()
        for ep in range(num_steps):
            ll, init_stats, dynamics_stats, emission_stats = self.e_step(emissions, inputs, init_mean, init_cov)
            self.m_step(init_stats, dynamics_stats, emission_stats)

            log_likelihood_out.append(ll.detach().cpu().numpy())
            time_out.append(time.time() - start)

            if self.verbose:
                print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
                print('log likelihood = ' + str(log_likelihood_out[-1]))
                print('Time elapsed = ' + str(time_out[-1]))

        self.log_likelihood = log_likelihood_out
        self.train_time = time_out

    def e_step(self, emissions, inputs, init_mean, init_cov):
        num_data_sets = emissions.shape[0]
        num_timesteps = emissions.shape[1]

        # Run the smoother to get posterior expectations
        ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses = \
            self.lgssm_smoother(emissions, inputs, init_mean, init_cov)

        # shorthand
        Ex = smoothed_means
        Exp = smoothed_means[:, :-1, :]
        Exn = smoothed_means[:, 1:, :]
        Vx = smoothed_covs
        Vxp = smoothed_covs[:, :-1, :, :]
        Vxn = smoothed_covs[:, 1:, :, :]
        Expxn = smoothed_crosses

        # Append bias to the inputs
        inputs = torch.cat((inputs, torch.ones((num_data_sets, num_timesteps, 1))), dim=2)
        up = inputs[:, :-1, :]
        u = inputs
        y = emissions
        y = torch.where(torch.isnan(y), 0, y)

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = smoothed_means[:, 0, :]
        Ex0x0T = smoothed_covs[:, 0, :] + Ex0[:, :, None] @ Ex0[:, None, :]
        init_stats = (Ex0, Ex0x0T, torch.tensor(1, dtype=self.dtype, device=self.device))

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        m11 = self._batch_trans(Exp) @ Exp
        m12 = self._batch_trans(Exp) @ up
        m21 = self._batch_trans(up) @ Exp
        m22 = self._batch_trans(up) @ up
        sum_zpzpT = self._block([[m11, m12], [m21, m22]])
        sum_zpzpT[:, :self.latent_dim, :self.latent_dim] += Vxp.sum(1)
        sum_zpxnT = self._block([[Expxn.sum(1)], [self._batch_trans(up) @ Exn]])
        sum_xnxnT = Vxn.sum(1) + self._batch_trans(Exn) @ Exn
        dynamics_stats = (sum_zpzpT[:, :-1, :-1], sum_zpxnT[:, :-1, :], sum_xnxnT,
                          torch.tensor(num_timesteps - 1, dtype=self.dtype, device=self.device))

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        n11 = self._batch_trans(Ex) @ Ex
        n12 = self._batch_trans(Ex) @ u
        n21 = self._batch_trans(u) @ Ex
        n22 = self._batch_trans(u) @ u
        sum_zzT = self._block([[n11, n12], [n21, n22]])
        sum_zzT[:, :self.latent_dim, :self.latent_dim] += Vx.sum(1)
        sum_zyT = self._block([[self._batch_trans(Ex) @ y], [self._batch_trans(u) @ y]])
        sum_yyT = self._batch_trans(y) @ y
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

        FB, dcl = fit_linear_regression(*dynamics_stats)
        self.dynamics_cov_log = torch.log(torch.diag(dcl))
        self.dynamics_weights = FB[:, :self.latent_dim]
        iwl, b = (FB[:, self.latent_dim:], None)
        self.inputs_weights_log = torch.log(torch.diag(iwl))

        HD, ecl = fit_linear_regression(*emission_stats)
        self.emissions_cov_log = torch.log(torch.diag(ecl))
        H = HD[:, :self.latent_dim]
        D, d = (HD[:, self.latent_dim:], None)

    def standardize_inputs(self, emissions_list, inputs_list):
        assert(type(emissions_list) is list)
        assert(type(inputs_list) is list)

        if emissions_list[0].dtype is not self.dtype:
            emissions_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions_list]
            inputs_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in inputs_list]

        emissions, inputs = self._stack_data(emissions_list, inputs_list)

        return emissions, inputs

    @staticmethod
    def _block(block_list, dims=(2, 1)):
        layer = []
        for i in block_list:
            layer.append(torch.cat(i, dim=dims[0]))

        return torch.cat(layer, dim=dims[1])

    @staticmethod
    def _batch_trans(batch_matrix):
        return torch.permute(batch_matrix, (0, 2, 1))

    @staticmethod
    def _get_batches_inds(num_data, batch_size, generator):
        num_batches = np.ceil(num_data / batch_size)
        data_inds = np.arange(num_data)
        generator.shuffle(data_inds)
        batch_data_inds = np.array_split(data_inds, num_batches)

        return batch_data_inds

    @staticmethod
    def estimate_init(emissions):
        init_mean = torch.nanmean(emissions, dim=1)
        init_cov = [LgssmSimple._estimate_cov(i)[None, :, :] for i in emissions]
        init_cov = torch.cat(init_cov, dim=0)

        return init_mean, init_cov

    @staticmethod
    def _stack_data(emissions_list, inputs_list):
        device = emissions_list[0].device
        dtype = inputs_list[0].dtype

        data_set_time = [i.shape[0] for i in emissions_list]
        max_time = np.max(data_set_time)
        num_data_sets = len(emissions_list)
        num_neurons = emissions_list[0].shape[1]

        emissions = torch.empty((num_data_sets, max_time, num_neurons), device=device, dtype=dtype)
        emissions[:] = torch.nan
        inputs = torch.zeros((num_data_sets, max_time, num_neurons), device=device, dtype=dtype)

        for d in range(num_data_sets):
            emissions[d, :data_set_time[d], :] = emissions_list[d]
            inputs[d, :data_set_time[d], :] = inputs_list[d]

        return emissions, inputs

    @staticmethod
    def _split_data(emissions, inputs, num_splits=2):
        # split the data in half for lower memory usage
        emissions_split = []
        inputs_split = []
        for ems, ins in zip(emissions, inputs):
            time_inds = np.arange(ems.shape[0])
            split_inds = np.array_split(time_inds, num_splits)

            for s in split_inds:
                emissions_split.append(ems[s, :])
                inputs_split.append(ins[s, :])

        return emissions_split, inputs_split

    @staticmethod
    def _estimate_cov(a):
        def nan_matmul(a, b, impute_val=0):
            a_no_nan = torch.where(torch.isnan(a), impute_val, a)
            b_no_nan = torch.where(torch.isnan(b), impute_val, b)

            return a_no_nan @ b_no_nan

        a_mean_sub = a - torch.nanmean(a, dim=0, keepdim=True)
        # estimate the covariance from the data in a
        cov = nan_matmul(a_mean_sub.T, a_mean_sub) / a.shape[0]

        # some columns will be all 0s due to missing data
        # replace those diagonals with the mean covariance
        cov_diag = torch.diag(cov)
        cov_diag_mean = torch.mean(cov_diag[cov_diag != 0])
        cov_diag = torch.where(cov_diag == 0, cov_diag_mean, cov_diag)

        cov[torch.eye(a.shape[1], dtype=torch.bool)] = cov_diag

        return cov
