import torch
import numpy as np
import pickle
import time
import utilities as utils


class Lgssm:
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """

    def __init__(self, dynamics_dim, emissions_dim, input_dim, dtype=torch.float64, device='cpu', verbose=True):
        self.dynamics_dim = dynamics_dim
        self.emissions_dim = emissions_dim
        self.input_dim = input_dim
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.log_likelihood = None
        self.train_time = None

        # initialize dynamics weights
        self.dynamics_weights_init = 0.9 * np.eye(self.dynamics_dim)
        self.dynamics_input_weights_init = np.zeros((self.dynamics_dim, self.input_dim))
        self.dynamics_offset_init = np.zeros(self.dynamics_dim)
        self.dynamics_cov_init = np.eye(self.dynamics_dim)

        # initialize emissions weights
        self.emissions_weights_init = np.eye(self.emissions_dim)
        self.emissions_input_weights_init = np.zeros((self.emissions_dim, self.input_dim))
        self.emissions_offset_init = np.zeros(self.emissions_dim)
        self.emissions_cov_init = np.eye(self.emissions_dim)

        # define the weights here, but set them to tensor versions of the intial values with _set_to_init()
        self.dynamics_weights = None
        self.dynamics_input_weights = None
        self.dynamics_offset = None
        self.dynamics_cov = None

        self.emissions_weights = None
        self.emissions_input_weights = None
        self.emissions_offset = None
        self.emissions_cov = None

        # convert to tensors
        self._set_to_init()

    def save(self, path='trained_models/trained_model.pkl'):
        save_file = open(path, 'wb')
        pickle.dump(self, save_file)
        save_file.close()

    def randomize_weights(self, max_eig_allowed=0.8, init_std=1.0, rng=np.random.default_rng()):
        # randomize dynamics weights
        self.dynamics_weights_init = rng.standard_normal((self.dynamics_dim, self.dynamics_dim))
        max_eig_in_mat = np.max(np.abs(np.linalg.eigvals(self.dynamics_weights_init)))
        self.dynamics_weights_init = max_eig_allowed * self.dynamics_weights_init / max_eig_in_mat

        self.dynamics_input_weights_init = init_std * rng.standard_normal((self.dynamics_dim, self.input_dim))
        self.dynamics_offset_init = 0 * rng.standard_normal(self.dynamics_dim)
        self.dynamics_cov_init = init_std * rng.standard_normal((self.dynamics_dim, self.dynamics_dim))
        self.dynamics_cov_init = self.dynamics_cov_init.T @ self.dynamics_cov_init / self.dynamics_dim

        # randomize emissions weights
        self.emissions_weights_init = init_std * rng.standard_normal((self.emissions_dim, self.dynamics_dim))
        self.emissions_input_weights_init = init_std * rng.standard_normal((self.dynamics_dim, self.input_dim))
        self.emissions_offset_init = 0 * rng.standard_normal(self.dynamics_dim)
        self.emissions_cov_init = init_std * rng.standard_normal((self.emissions_dim, self.emissions_dim))
        self.emissions_cov_init = self.emissions_cov_init.T @ self.emissions_cov_init / self.emissions_dim

        self._set_to_init()

    def _set_to_init(self, update=None):
        if update is None:
            update = self._get_default_update()

        if update['dynamics']['weights']:
            self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device, dtype=self.dtype)
        if update['dynamics']['input_weights']:
            self.dynamics_input_weights = torch.tensor(self.dynamics_input_weights_init, device=self.device, dtype=self.dtype)
        if update['dynamics']['offset']:
            self.dynamics_offset = torch.tensor(self.dynamics_offset_init, device=self.device, dtype=self.dtype)
        if update['dynamics']['cov']:
            self.dynamics_cov = torch.tensor(self.dynamics_cov_init, device=self.device, dtype=self.dtype)

        if update['emissions']['weights']:
            self.emissions_weights = torch.tensor(self.emissions_weights_init, device=self.device, dtype=self.dtype)
        if update['emissions']['input_weights']:
            self.emissions_input_weights = torch.tensor(self.emissions_input_weights_init, device=self.device, dtype=self.dtype)
        if update['emissions']['offset']:
            self.emissions_offset = torch.tensor(self.emissions_offset_init, device=self.device, dtype=self.dtype)
        if update['emissions']['cov']:
            self.emissions_cov = torch.tensor(self.emissions_cov_init, device=self.device, dtype=self.dtype)

    def set_params(self, params):
        update = self._get_default_update(set_value=False)

        # set dynamics weights
        if params['dynamics']['weights'] is not None:
            self.dynamics_weights_init = params['dynamics']['weights']
            update['dynamics']['weights'] = True

        if params['dynamics']['input_weights'] is not None:
            self.dynamics_input_weights_init = params['dynamics']['input_weights']
            update['dynamics']['input_weights'] = True

        if params['dynamics']['offset'] is not None:
            self.dynamics_offset_init = params['dynamics']['offset']
            update['dynamics']['offset'] = True

        if params['dynamics']['cov'] is not None:
            self.dynamics_cov_init = params['dynamics']['cov']
            update['dynamics']['cov'] = True

        # set emissions weights
        if params['emissions']['weights'] is not None:
            self.emissions_weights_init = params['emissions']['weights']
            update['emissions']['weights'] = True

        if params['emissions']['input_weights'] is not None:
            self.emissions_input_weights_init = params['emissions']['input_weights']
            update['emissions']['input_weights'] = True

        if params['emissions']['offset'] is not None:
            self.emissions_offset_init = params['emissions']['offset']
            update['emissions']['offset'] = True

        if params['emissions']['cov'] is not None:
            self.emissions_cov_init = params['emissions']['cov']
            update['emissions']['cov'] = True

        self._set_to_init(update)

    def standardize_inputs(self, emissions_list, input_list):
        assert(type(emissions_list) is list)
        assert(type(input_list) is list)

        if emissions_list[0].dtype is not self.dtype:
            emissions_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions_list]
            input_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in input_list]

        emissions, input = self._stack_data(emissions_list, input_list)

        return emissions, input

    def get_params(self):
        params_out = {'dynamics': {'weights': self.dynamics_weights.detach().cpu().numpy(),
                                   'input_weights': self.dynamics_input_weights.detach().cpu().numpy(),
                                   'offset': self.dynamics_offset.detach().cpu().numpy(),
                                   'cov': self.dynamics_cov.detach().cpu().numpy(),
                                   },
                      'emissions': {'weights': self.emissions_weights.detach().cpu().numpy(),
                                    'input_weights': self.emissions_input_weights.detach().cpu().numpy(),
                                    'offset': self.emissions_offset.detach().cpu().numpy(),
                                    'cov': self.emissions_cov.detach().cpu().numpy(),
                                    },
                      }

        return params_out

    def set_device(self, new_device):
        self.device = new_device

        self.dynamics_weights = self.dynamics_weights.to(new_device)
        self.dynamics_input_weights = self.dynamics_input_weights.to(new_device)
        self.dynamics_offset = self.dynamics_offset.to(new_device)
        self.dynamics_cov = self.dynamics_cov.to(new_device)

        self.emissions_weights = self.emissions_weights.to(new_device)
        self.emissions_input_weights = self.emissions_input_weights.to(new_device)
        self.emissions_offset = self.emissions_offset.to(new_device)
        self.emissions_cov = self.emissions_cov.to(new_device)

    def sample(self, num_time=100, num_data_sets=None, nan_freq=0.0, rng=np.random.default_rng()):
        # generate a random initial mean and covariance
        init_mean = rng.standard_normal((num_data_sets, self.dynamics_dim))
        init_cov = rng.standard_normal((num_data_sets, self.dynamics_dim, self.dynamics_dim))
        init_cov = np.transpose(init_cov, [0, 2, 1]) @ init_cov / self.dynamics_dim

        latents = np.zeros((num_data_sets, num_time, self.dynamics_dim))
        emissions = np.zeros((num_data_sets, num_time, self.dynamics_dim))
        inputs = rng.standard_normal((num_data_sets, num_time, self.dynamics_dim))

        # get the initial observations
        dynamics_noise = rng.multivariate_normal(np.zeros(self.dynamics_dim), self.dynamics_cov, size=(num_data_sets, num_time))
        emissions_noise = rng.multivariate_normal(np.zeros(self.dynamics_dim), self.emissions_cov, size=(num_data_sets, num_time))
        dynamics_inputs = (self.dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]
        emissions_inputs = (self.emissions_input_weights @ inputs[:, :, :, None])[:, :, :, 0]
        for d in range(num_data_sets):
            latent_init = rng.multivariate_normal(init_mean[d, :], init_cov[d, :, :])

            latents[d, 0, :] = self.dynamics_weights @ latent_init + \
                               dynamics_inputs[d, 0, :] + \
                               self.dynamics_offset + \
                               dynamics_noise[d, 0, :]

            emissions[d, 0, :] = self.emissions_weights @ latents[d, 0, :] + \
                                 emissions_inputs[d, 0, :] + \
                                 self.emissions_offset + \
                                 emissions_noise[d, 0, :]

        # loop through time and generate the latents and emissions
        for t in range(1, num_time):
            a=1
            latents[:, t, :] = (self.dynamics_weights @ latents[:, t-1, :, None])[:, :, 0] + \
                                dynamics_inputs[:, t, :] + \
                                self.dynamics_offset[None, :] + \
                                dynamics_noise[:, t, :]

            emissions[:, t, :] = (self.emissions_weights @ latents[:, t, :, None])[:, :, 0] + \
                                  emissions_inputs[:, t, :] + \
                                  self.emissions_offset[None, :] + \
                                  emissions_noise[:, t, :]

        # add in nans
        nan_mask = rng.random((num_data_sets, num_time, self.dynamics_dim)) <= nan_freq
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

    def fit_em(self, emissions_list, input_list, init_mean=None, init_cov=None, num_steps=10):
        emissions, input = self.standardize_inputs(emissions_list, input_list)

        if init_mean is None:
            init_mean = self.estimate_init_mean(emissions)
        else:
            init_mean = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in init_mean]
            init_mean = torch.stack(init_mean)

        if init_cov is None:
            init_cov = self.estimate_init_cov(emissions)
        else:
            init_cov = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in init_cov]
            init_cov = torch.stack(init_cov)

        params = {'dynamics': {'weights': self.dynamics_weights,
                               'input_weights': self.dynamics_input_weights,
                               'offset': self.dynamics_offset,
                               'cov': self.dynamics_cov,
                               },
                  'emissions': {'weights': self.emissions_weights,
                                'input_weights': self.emissions_input_weights,
                                'offset': self.emissions_offset,
                                'cov': self.emissions_cov,
                                }
                  }

        log_likelihood_out = []
        time_out = []

        start = time.time()
        for ep in range(num_steps):
            ll, init_stats, dynamics_stats, emission_stats = self.e_step(emissions, input, init_mean, init_cov)
            self.m_step(init_stats, dynamics_stats, emission_stats)

            log_likelihood_out.append(ll.detach().cpu().numpy())
            time_out.append(time.time() - start)

            if self.verbose:
                print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
                print('log likelihood = ' + str(log_likelihood_out[-1]))
                print('Time elapsed = ' + str(time_out[-1]))

        self.log_likelihood = log_likelihood_out
        self.train_time = time_out

    def lgssm_filter(self, emissions, inputs, init_mean, init_cov, nan_fill=1e8):
        """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
        """
        num_data_sets, num_timesteps = emissions.shape[:2]

        ll = torch.zeros(num_data_sets, device=self.device, dtype=self.dtype)
        pred_mean = init_mean
        pred_cov = init_cov

        dynamics_inputs = (self.dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]
        emissions_inputs = (self.emissions_input_weights @ inputs[:, :, :, None])[:, :, :, 0]

        filtered_means_list = []
        filtered_covs_list = []

        for t in range(num_timesteps):
            # Shorthand: get parameters and input for time index t
            y = emissions[:, t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)
            R = torch.where(torch.diag_embed(nan_loc), nan_fill, torch.tile(self.emissions_cov, (num_data_sets, 1, 1)))

            # Update the log likelihood
            ll_mu = utils.batch_Ax(self.emissions_weights, pred_mean) + emissions_inputs[:, t, :] + self.emissions_offset[None, :]
            ll_cov = self.emissions_weights @ pred_cov @ self.emissions_weights.T + R

            ll += torch.distributions.multivariate_normal.MultivariateNormal(ll_mu, ll_cov).log_prob(y)

            # Condition on this emission
            # Compute the Kalman gain
            S = R + self.emissions_weights @ pred_cov @ self.emissions_weights.T
            K = utils.batch_trans(torch.linalg.solve(S, self.emissions_weights @ pred_cov))

            filtered_cov = pred_cov - K @ S @ utils.batch_trans(K)
            filtered_mean = pred_mean + utils.batch_Ax(K, (y - ll_mu))

            filtered_covs_list.append(filtered_cov)
            filtered_means_list.append(filtered_mean)

            # Predict the next state
            pred_mean = utils.batch_Ax(self.dynamics_weights, filtered_mean) + dynamics_inputs[:, t, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

        filtered_means = torch.permute(torch.stack(filtered_means_list), (1, 0, 2))
        filtered_covs = torch.permute(torch.stack(filtered_covs_list), (1, 0, 2, 3))

        # ll = torch.sum(ll) / emissions.numel()
        ll = torch.sum(ll)

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

        dynamics_inputs = (self.dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]

        smoothed_means = filtered_means.detach().clone()
        smoothed_covs = filtered_covs.detach().clone()
        smoothed_crosses = filtered_covs[:, :-1, :, :].detach().clone()

        # Run the smoother backward in time
        for t in reversed(range(num_timesteps - 1)):
            # Unpack the input
            filtered_mean = filtered_means[:, t, :]
            filtered_cov = filtered_covs[:, t, :, :]
            smoothed_cov_next = smoothed_covs[:, t + 1, :, :]
            smoothed_mean_next = smoothed_means[:, t + 1, :]

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            a = self.dynamics_cov + self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T
            b = self.dynamics_weights @ filtered_cov
            G = utils.batch_trans(torch.linalg.solve(a, b))

            # Compute the smoothed mean and covariance
            pred_mean = utils.batch_Ax(self.dynamics_weights, filtered_mean) + dynamics_inputs[:, t, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov
            smoothed_covs[:, t, :, :] = filtered_cov + G @ (smoothed_cov_next - pred_cov) @ utils.batch_trans(G)
            smoothed_means[:, t, :] = filtered_mean + utils.batch_Ax(G, smoothed_mean_next - pred_mean)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            smoothed_crosses[:, t, :, :] = G @ smoothed_cov_next + smoothed_means[:, t, :, None] * smoothed_mean_next[:,
                                                                                                   None, :]

        return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses

    def e_step(self, emissions, input, init_mean, init_cov):
        num_data_sets, num_timesteps = emissions.shape[:2]

        # Run the smoother to get posterior expectations
        ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses = \
            self.lgssm_smoother(emissions, input, init_mean, init_cov)

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
        m11 = utils.batch_trans(Exp) @ Exp
        m12 = utils.batch_trans(Exp) @ up
        m21 = utils.batch_trans(up) @ Exp
        m22 = utils.batch_trans(up) @ up
        sum_zpzpT = utils.block([[m11, m12], [m21, m22]])
        sum_zpzpT[:, :self.dynamics_dim, :self.dynamics_dim] += Vxp.sum(1)
        sum_zpxnT = utils.block([[Expxn.sum(1)], [utils.batch_trans(up) @ Exn]])
        sum_xnxnT = Vxn.sum(1) + utils.batch_trans(Exn) @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, torch.tensor(num_timesteps - 1, device=self.device, dtype=self.dtype))

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        n11 = utils.batch_trans(Ex) @ Ex
        n12 = utils.batch_trans(Ex) @ u
        n21 = utils.batch_trans(u) @ Ex
        n22 = utils.batch_trans(u) @ u
        sum_zzT = utils.block([[n11, n12], [n21, n22]])
        sum_zzT[:, :self.dynamics_dim, :self.dynamics_dim] += Vx.sum(1)
        sum_zyT = utils.block([[utils.batch_trans(Ex) @ y], [utils.batch_trans(u) @ y]])
        sum_yyT = utils.batch_trans(y) @ y
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, torch.tensor(num_timesteps, device=self.device, dtype=self.dtype))

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

        FB, Q = fit_linear_regression(*dynamics_stats)
        A = FB[:, :self.dynamics_dim]
        B, b = (FB[:, self.dynamics_dim:-1], FB[:, -1])

        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :self.dynamics_dim]
        D, d = (HD[:, self.dynamics_dim:-1], HD[:, -1])

        self.dynamics_weights = A
        self.dynamics_input_weights = B
        self.dynamics_offset = b
        self.dynamics_cov = Q

        self.emissions_weights = H
        self.emissions_input_weights = D
        self.emissions_offset = d
        self.emissions_cov = R

        return

    def fit_gd(self, emissions_list, input_list, learning_rate=1e-2, num_steps=50):
        """ This function will fit the model using gradient descent on the entire data set
        """
        self.dynamics_weights.requires_grad = True
        self.input_weights.requires_grad = True
        self.dynamics_cov.requires_grad = True
        self.emissions_cov.requires_grad = True

        emissions, input = self.standardize_inputs(emissions_list, input_list)
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

    @staticmethod
    def _get_default_update(set_value=True):
        update = {'dynamics': {'weights': set_value,
                               'input_weights': set_value,
                               'offset': set_value,
                               'cov': set_value,
                               },
                  'emissions': {'weights': set_value,
                                'input_weights': set_value,
                                'offset': set_value,
                                'cov': set_value,
                                },
                  }

        return update

    @staticmethod
    def estimate_init_mean(emissions):
        init_mean = torch.nanmean(emissions, dim=1)

        return init_mean

    @staticmethod
    def estimate_init_cov(emissions):
        init_cov = [utils.estimate_cov(i)[None, :, :] for i in emissions]
        init_cov = torch.cat(init_cov, dim=0)

        return init_cov

    @staticmethod
    def _stack_data(emissions_list, input_list):
        device = emissions_list[0].device
        dtype = input_list[0].dtype

        data_set_time = [i.shape[0] for i in emissions_list]
        max_time = np.max(data_set_time)
        num_data_sets = len(emissions_list)
        num_neurons = emissions_list[0].shape[1]

        emissions = torch.empty((num_data_sets, max_time, num_neurons), device=device, dtype=dtype)
        emissions[:] = torch.nan
        input = torch.zeros((num_data_sets, max_time, num_neurons), device=device, dtype=dtype)

        for d in range(num_data_sets):
            emissions[d, :data_set_time[d], :] = emissions_list[d]
            input[d, :data_set_time[d], :] = input_list[d]

        return emissions, input

    @staticmethod
    def _split_data(emissions, input, num_splits=2):
        # split the data in half for lower memory usage
        emissions_split = []
        input_split = []
        for ems, ins in zip(emissions, input):
            time_inds = np.arange(ems.shape[0])
            split_inds = np.array_split(time_inds, num_splits)

            for s in split_inds:
                emissions_split.append(ems[s, :])
                input_split.append(ins[s, :])

        return emissions_split, input_split

