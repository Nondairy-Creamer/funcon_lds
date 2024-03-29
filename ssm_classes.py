import torch
import numpy as np
import pickle
import inference_utilities as iu
import warnings


class Lgssm:
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """

    def __init__(self, dynamics_dim, emissions_dim, input_dim, dynamics_lags=1, dynamics_input_lags=1, cell_ids=None,
                 param_props=None, dtype=torch.float64, device='cpu', verbose=True, nan_fill=1e8, ridge_lambda=0):
        self.dynamics_lags = dynamics_lags
        self.dynamics_input_lags = dynamics_input_lags
        self.dynamics_dim = dynamics_dim
        self.emissions_dim = emissions_dim
        self.input_dim = input_dim
        self.dynamics_dim_full = self.dynamics_dim * self.dynamics_lags
        self.input_dim_full = self.input_dim * self.dynamics_input_lags
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.log_likelihood = None
        self.train_time = None
        self.epsilon = nan_fill
        self.sample_rate = 0.5  # default is 2 Hz
        self.ridge_lambda = ridge_lambda
        if cell_ids is None:
            self.cell_ids = ['S' + str(i) for i in range(self.dynamics_dim)]
        else:
            self.cell_ids = cell_ids

        # define the weights here, but set them to tensor versions of the intial values with _set_to_init()
        self.dynamics_weights = None
        self.dynamics_input_weights = None
        self.dynamics_offset = None
        self.dynamics_cov = None

        self.emissions_weights = None
        self.emissions_input_weights = None
        self.emissions_offset = None
        self.emissions_cov = None

        self.param_props = {'update': {'dynamics_weights': True,
                                       'dynamics_input_weights': True,
                                       'dynamics_offset': False,
                                       'dynamics_cov': True,
                                       'emissions_weights': True,
                                       'emissions_input_weights': True,
                                       'emissions_offset': False,
                                       'emissions_cov': True,
                                       },

                            'shape': {'dynamics_weights': 'full',
                                      'dynamics_input_weights': 'full',
                                      'dynamics_offset': 'full',
                                      'dynamics_cov': 'full',
                                      'emissions_weights': 'full',
                                      'emissions_input_weights': 'full',
                                      'emissions_offset': 'full',
                                      'emissions_cov': 'full',
                                      },

                            'mask': {'dynamics_weights': None,
                                     'dynamics_input_weights': None,
                                     'dynamics_offset': None,
                                     'dynamics_cov': None,
                                     'emissions_weights': None,
                                     'emissions_input_weights': None,
                                     'emissions_offset': None,
                                     'emissions_cov': None,
                                     },
                            }

        if param_props is not None:
            for k in self.param_props.keys():
                self.param_props[k].update(param_props[k])

        # initialize dynamics weights
        tau = self.dynamics_lags / 3
        const = (np.exp(3) - 1) * np.exp(1 / tau - 3) / (np.exp(1 / tau) - 1)
        time_decay = np.exp(-np.arange(self.dynamics_lags) / tau) / const
        self.dynamics_weights_init = 0.9 * np.tile(np.eye(self.dynamics_dim), (self.dynamics_lags, 1, 1))
        self.dynamics_weights_init = self.dynamics_weights_init * time_decay[:, None, None]
        self.dynamics_cov_init = np.eye(self.dynamics_dim)
        self.dynamics_input_weights_init = np.zeros((self.dynamics_input_lags, self.dynamics_dim, self.input_dim))
        self.dynamics_offset_init = np.zeros(self.dynamics_dim)

        # initialize emissions weights
        self.emissions_weights_init = np.eye(self.emissions_dim)
        self.emissions_input_weights_init = np.zeros((self.emissions_dim, self.input_dim))
        self.emissions_offset_init = np.zeros(self.emissions_dim)
        self.emissions_cov_init = np.eye(self.emissions_dim)

        self._pad_init_for_lags()
        # convert to tensors
        self.set_to_init()

    def save(self, path='trained_models/trained_model.pkl'):
        save_file = open(path, 'wb')
        pickle.dump(self, save_file)
        save_file.close()

    def randomize_weights(self, max_eig_allowed=0.9, rng=np.random.default_rng()):
        input_weights_std = 1
        noise_std = 0.1

        # randomize dynamics weights
        dynamics_tau = self.dynamics_lags / 3
        dynamics_const = (np.exp(3) - 1) * np.exp(1 / dynamics_tau - 3) / (np.exp(1 / dynamics_tau) - 1)
        dynamics_time_decay = np.exp(-np.arange(self.dynamics_lags) / dynamics_tau) / dynamics_const
        # self.dynamics_weights_init = rng.standard_normal((self.dynamics_lags, self.dynamics_dim, self.dynamics_dim))
        self.dynamics_weights_init = np.tile(rng.standard_normal((1, self.dynamics_dim, self.dynamics_dim)), (self.dynamics_lags, 1, 1))
        eig_vals, eig_vects = np.linalg.eig(self.dynamics_weights_init)
        eig_vals = eig_vals / np.max(np.abs(eig_vals)) * max_eig_allowed
        negative_real_eigs = np.real(eig_vals) < 0
        eig_vals[negative_real_eigs] = -eig_vals[negative_real_eigs]
        eig_vals_mat = np.zeros((self.dynamics_lags, self.dynamics_dim, self.dynamics_dim), dtype=np.cdouble)
        for i in range(self.dynamics_lags):
            eig_vals_mat[i, :, :] = np.diag(eig_vals[i, :])
        self.dynamics_weights_init = np.real(eig_vects @ np.transpose(np.linalg.solve(np.transpose(eig_vects, (0, 2, 1)), eig_vals_mat), (0, 2, 1)))
        self.dynamics_weights_init = self.dynamics_weights_init * dynamics_time_decay[:, None, None]

        dynamics_input_tau = self.dynamics_input_lags / 3
        dynamics_input_const = (np.exp(3) - 1) * np.exp(1 / dynamics_input_tau - 3) / (np.exp(1 / dynamics_input_tau) - 1)
        dynamics_input_time_decay = np.exp(-np.arange(self.dynamics_input_lags) / dynamics_input_tau) / dynamics_input_const
        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            # dynamics_input_weights_init_diag = init_std * rng.standard_normal((self.dynamics_input_lags, self.input_dim))
            dynamics_input_weights_init_diag = input_weights_std * np.tile(np.exp(rng.standard_normal(self.input_dim)), (self.dynamics_input_lags, 1))
            self.dynamics_input_weights_init = np.zeros((self.dynamics_input_lags, self.dynamics_dim, self.input_dim))
            for i in range(self.dynamics_input_lags):
                self.dynamics_input_weights_init[i, :self.input_dim, :] = np.diag(dynamics_input_weights_init_diag[i, :])
        else:
            self.dynamics_input_weights_init = input_weights_std * rng.standard_normal((self.dynamics_input_lags, self.dynamics_dim, self.input_dim))
        self.dynamics_input_weights_init = self.dynamics_input_weights_init * dynamics_input_time_decay[:, None, None]

        self.dynamics_offset_init = np.zeros(self.dynamics_dim)

        if self.param_props['shape']['dynamics_cov'] == 'diag':
            self.dynamics_cov_init = np.diag(np.exp(noise_std * rng.standard_normal(self.dynamics_dim)))
        else:
            self.dynamics_cov_init = rng.standard_normal((self.dynamics_dim, self.dynamics_dim))
            self.dynamics_cov_init = noise_std * (self.dynamics_cov_init.T @ self.dynamics_cov_init / self.dynamics_dim + np.eye(self.dynamics_dim))

        # randomize emissions weights
        self.emissions_weights_init = rng.standard_normal((self.emissions_dim, self.dynamics_dim))
        self.emissions_input_weights_init = input_weights_std * rng.standard_normal((self.emissions_dim, self.input_dim))
        self.emissions_offset_init = np.zeros(self.emissions_dim)

        if self.param_props['shape']['emissions_cov'] == 'diag':
            self.emissions_cov_init = np.diag(np.exp(noise_std * rng.standard_normal(self.dynamics_dim)))
        else:
            self.emissions_cov_init = rng.standard_normal((self.emissions_dim, self.emissions_dim))
            self.emissions_cov_init = noise_std * (self.emissions_cov_init.T @ self.emissions_cov_init / self.emissions_dim + np.eye(self.emissions_dim))

        self._pad_init_for_lags()
        self.set_to_init()

    def set_to_init(self):
        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device, dtype=self.dtype)
        self.dynamics_input_weights = torch.tensor(self.dynamics_input_weights_init, device=self.device, dtype=self.dtype)
        self.dynamics_offset = torch.tensor(self.dynamics_offset_init, device=self.device, dtype=self.dtype)
        self.dynamics_cov = torch.tensor(self.dynamics_cov_init, device=self.device, dtype=self.dtype)

        self.emissions_weights = torch.tensor(self.emissions_weights_init, device=self.device, dtype=self.dtype)
        self.emissions_input_weights = torch.tensor(self.emissions_input_weights_init, device=self.device, dtype=self.dtype)
        self.emissions_offset = torch.tensor(self.emissions_offset_init, device=self.device, dtype=self.dtype)
        self.emissions_cov = torch.tensor(self.emissions_cov_init, device=self.device, dtype=self.dtype)

    def standardize_inputs(self, emissions_list, input_list):
        assert(type(emissions_list) is list)
        assert(type(input_list) is list)

        if emissions_list[0].dtype is not self.dtype:
            emissions_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions_list]
            input_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in input_list]

        return emissions_list, input_list

    def get_params(self):
        params_out = {'init': {'dynamics_weights': self.dynamics_weights_init,
                               'dynamics_input_weights': self.dynamics_input_weights_init,
                               'dynamics_offset': self.dynamics_offset_init,
                               'dynamics_cov': self.dynamics_cov_init,
                               'emissions_weights': self.emissions_weights_init,
                               'emissions_input_weights': self.emissions_input_weights_init,
                               'emissions_offset': self.emissions_offset_init,
                               'emissions_cov': self.emissions_cov_init,
                               },

                      'trained': {'dynamics_weights': self.dynamics_weights.detach().cpu().numpy(),
                                  'dynamics_input_weights': self.dynamics_input_weights.detach().cpu().numpy(),
                                  'dynamics_offset': self.dynamics_offset.detach().cpu().numpy(),
                                  'dynamics_cov': self.dynamics_cov.detach().cpu().numpy(),
                                  'emissions_weights': self.emissions_weights.detach().cpu().numpy(),
                                  'emissions_input_weights': self.emissions_input_weights.detach().cpu().numpy(),
                                  'emissions_offset': self.emissions_offset.detach().cpu().numpy(),
                                  'emissions_cov': self.emissions_cov.detach().cpu().numpy(),
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

    def sample(self, num_time=100, init_mean=None, init_cov=None, num_data_sets=1, input_time_scale=0, inputs_list=None,
               scattered_nan_freq=0.0, lost_emission_freq=0.0, rng=np.random.default_rng(), add_noise=True):
        latents_list = []
        emissions_list = []
        init_mean_list = []
        init_cov_list = []
        if add_noise:
            noise_mult = 1
        else:
            noise_mult = 0

        if type(num_time) is not list:
            num_time = [num_time] * num_data_sets

        for nt in num_time:
            if inputs_list is None:
                if input_time_scale != 0:
                    stims_per_data_set = int(nt / input_time_scale)
                    num_stims = num_data_sets * stims_per_data_set
                    sparse_inputs_init = np.eye(self.input_dim)[rng.choice(self.input_dim, num_stims, replace=True)]

                    # upsample to full time
                    total_time = nt * num_data_sets
                    inputs_list = np.zeros((total_time, self.emissions_dim))
                    inputs_list[::input_time_scale, :] = sparse_inputs_init
                    inputs_list = np.split(inputs_list, num_data_sets)
                else:
                    inputs_list = [rng.standard_normal((nt, self.input_dim)) for i in range(num_data_sets)]

        inputs_lagged = [self.get_lagged_data(i, self.dynamics_input_lags, add_pad=True) for i in inputs_list]

        for d in range(num_data_sets):
            # generate a random initial mean and covariance
            if init_mean is None:
                init_mean = rng.standard_normal(self.dynamics_dim_full)

            if init_cov is None:
                init_cov = rng.standard_normal((self.dynamics_dim_full, self.dynamics_dim_full))
                init_cov = init_cov.T @ init_cov

            latents = np.zeros((num_time[d], self.dynamics_dim_full))
            emissions = np.zeros((num_time[d], self.emissions_dim))
            inputs = inputs_lagged[d]

            # get the initial observations
            dynamics_noise = noise_mult * rng.multivariate_normal(np.zeros(self.dynamics_dim_full), self.dynamics_cov, size=num_time[d])
            emissions_noise = noise_mult * rng.multivariate_normal(np.zeros(self.emissions_dim), self.emissions_cov, size=num_time[d])
            dynamics_inputs = (self.dynamics_input_weights @ inputs[:, :, None])[:, :, 0]
            emissions_inputs = (self.emissions_input_weights @ inputs[:, :, None])[:, :, 0]
            latent_init = rng.multivariate_normal(init_mean, init_cov)

            latents[0, :] = self.dynamics_weights @ latent_init + \
                            dynamics_inputs[0, :] + \
                            self.dynamics_offset + \
                            dynamics_noise[0, :]

            emissions[0, :] = self.emissions_weights @ latents[0, :] + \
                              emissions_inputs[0, :] + \
                              self.emissions_offset[:] + \
                              emissions_noise[0, :]

            # loop through time and generate the latents and emissions
            for t in range(1, num_time[d]):
                latents[t, :] = (self.dynamics_weights @ latents[t-1, :]) + \
                                 dynamics_inputs[t, :] + \
                                 self.dynamics_offset + \
                                 dynamics_noise[t, :]

                emissions[t, :] = (self.emissions_weights @ latents[t, :]) + \
                                   emissions_inputs[t, :] + \
                                   self.emissions_offset + \
                                   emissions_noise[t, :]

            latents_list.append(latents)
            emissions_list.append(emissions)
            init_mean_list.append(init_mean)
            init_cov_list.append(init_cov)

        # add in nans
        scattered_nans_mask = [rng.random((num_time[i], self.emissions_dim)) < scattered_nan_freq for i in range(num_data_sets)]
        lost_emission_mask = [rng.random((1, self.emissions_dim)) < lost_emission_freq for i in range(num_data_sets)]
        nan_mask = [scattered_nans_mask[i] | lost_emission_mask[i] for i in range(num_data_sets)]

        # make sure each data set has at least one measurement and that all emissions were measured at least once
        for d in range(num_data_sets):
            if np.all(nan_mask[d]):
                nan_mask[d][:, rng.integers(0, self.emissions_dim)] = False

        neurons_measured = np.zeros(self.emissions_dim, dtype=bool)
        for d in range(num_data_sets):
            neurons_measured = neurons_measured | ~np.all(nan_mask[d], axis=0)

        for nmi, nm in enumerate(neurons_measured):
            if not nm:
                random_data_set = rng.integers(0, num_data_sets)
                nan_mask[random_data_set][:, nmi] = False

        for d in range(num_data_sets):
            emissions_list[d][nan_mask[d]] = np.nan

        data_dict = {'latents': latents_list,
                     'inputs': inputs_list,
                     'emissions': emissions_list,
                     'init_mean': init_mean_list,
                     'init_cov': init_cov_list,
                     'cell_ids': self.cell_ids,
                     }

        return data_dict

    def lgssm_filter(self, emissions, inputs, init_mean, init_cov):
        """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This function can deal with missing data if the data is missing the entire time trace
        """
        num_timesteps = emissions.shape[0]

        if inputs.shape[1] < self.input_dim_full:
            inputs = self.get_lagged_data(inputs, self.dynamics_input_lags)

        ll = torch.tensor(0, device=self.device, dtype=self.dtype)
        filtered_mean = init_mean
        filtered_cov = init_cov

        dynamics_inputs = inputs @ self.dynamics_input_weights.T
        emissions_inputs = inputs @ self.emissions_input_weights.T

        filtered_means_list = []
        filtered_covs_list = []
        converge_t = num_timesteps - 1

        # determine if you're going to check for covariance convergence
        # save a lot of time and memory, but can only be done if each neuron is either all or no nans
        check_convergence = self._has_no_scattered_nans(emissions)

        # step through the loop and keep calculating the covariances until they converge
        for t in range(num_timesteps):
            # Shorthand: get parameters and input for time index t
            y = emissions[t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)
            R = torch.where(torch.diag(nan_loc), self.epsilon, self.emissions_cov)

            # Predict the next state
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

            # Update the log likelihood
            ll_mu = self.emissions_weights @ pred_mean + emissions_inputs[t, :] + self.emissions_offset

            ll_cov = self.emissions_weights @ pred_cov @ self.emissions_weights.T + R
            ll_cov_logdet = torch.linalg.slogdet(ll_cov)[1]
            ll_cov_inv = torch.linalg.inv(ll_cov)

            # ll += torch.distributions.multivariate_normal.MultivariateNormal(ll_mu, ll_cov).log_prob(y)
            mean_diff = y - ll_mu
            ll += -1/2 * (emissions.shape[1] * np.log(2*np.pi) + ll_cov_logdet +
                          torch.dot(mean_diff, ll_cov_inv @ mean_diff))

            # Condition on this emission
            # Compute the Kalman gain
            K = pred_cov.T @ self.emissions_weights.T @ ll_cov_inv
            # K = torch.linalg.solve(ll_cov, self.emissions_weights @ pred_cov).T
            filtered_cov = pred_cov - K @ ll_cov @ K.T

            filtered_mean = pred_mean + K @ mean_diff
            filtered_means_list.append(filtered_mean)
            filtered_covs_list.append(filtered_cov)

            # check if covariance has converged
            if check_convergence and t > 0:
                max_abs_diff_cov = torch.max(torch.abs(filtered_covs_list[-1] - filtered_covs_list[-2]))
                if max_abs_diff_cov < 1 / self.epsilon:
                    converge_t = t
                    break

        # once the covariances converge, don't recalculate them
        for t in range(converge_t + 1, num_timesteps):
            # Shorthand: get parameters and input for time index t
            y = emissions[t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)

            # Predict the next state
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t, :] + self.dynamics_offset

            # Update the log likelihood
            ll_mu = self.emissions_weights @ pred_mean + emissions_inputs[t, :] + self.emissions_offset

            # ll += torch.distributions.multivariate_normal.MultivariateNormal(ll_mu, ll_cov).log_prob(y)
            mean_diff = y - ll_mu
            ll += -1 / 2 * (emissions.shape[1] * np.log(2 * np.pi) + ll_cov_logdet +
                            torch.dot(mean_diff, ll_cov_inv @ mean_diff))

            filtered_mean = pred_mean + K @ mean_diff

            filtered_means_list.append(filtered_mean)

        filtered_means_list = torch.stack(filtered_means_list)
        filtered_covs_list = torch.stack(filtered_covs_list)

        return ll, filtered_means_list, filtered_covs_list, converge_t

    def lgssm_smoother(self, emissions, inputs, init_mean, init_cov):
        if inputs.shape[1] < self.input_dim_full:
            inputs = self.get_lagged_data(inputs, self.dynamics_input_lags)

        ll, filtered_means, filtered_covs, converge_t = self.lgssm_filter(emissions, inputs, init_mean, init_cov)

        if converge_t < emissions.shape[0] / 2:
            smoothed_means, smoothed_covs, smoothed_crosses = \
                self.lgssm_smoother_fast(emissions, inputs, filtered_means, filtered_covs, converge_t)
        else:
            pad_covs = torch.tile(filtered_covs[None, -1, :, :], (emissions.shape[0] - converge_t - 1, 1, 1))
            filtered_covs = torch.cat((filtered_covs, pad_covs), dim=0)
            smoothed_means, smoothed_covs, smoothed_crosses = \
                self.lgssm_smoother_complete(emissions, inputs, filtered_means, filtered_covs)

        return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses

    def lgssm_smoother_complete(self, emissions, inputs, filtered_means, filtered_covs):
        r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states. Technically, this
        implements the Rauch-Tung-Striebel (RTS) smoother.
        Adopted from Dynamax
        """
        num_timesteps = emissions.shape[0]

        # Run the Kalman filter
        dynamics_inputs = inputs @ self.dynamics_input_weights.T

        smoothed_means = filtered_means.detach().clone()
        smoothed_covs = filtered_covs.detach().clone()
        smoothed_crosses = filtered_covs[:-1, :, :].detach().clone()

        # Run the smoother backward in time
        for t in reversed(range(num_timesteps - 1)):
            # Unpack the input
            filtered_mean = filtered_means[t, :]
            filtered_cov = filtered_covs[t, :, :]
            smoothed_cov_next = smoothed_covs[t + 1, :, :]
            smoothed_mean_next = smoothed_means[t + 1, :]

            # Compute the smoothed mean and covariance
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t+1, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            G = torch.linalg.solve(pred_cov, self.dynamics_weights @ filtered_cov).T
            smoothed_covs[t, :, :] = filtered_cov + G @ (smoothed_cov_next - pred_cov) @ G.T
            smoothed_means[t, :] = filtered_mean + G @ (smoothed_mean_next - pred_mean)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            # TODO: ask why the second expression is not in jonathan's code
            smoothed_crosses[t, :, :] = G @ smoothed_cov_next #+ smoothed_means[:, t, :, None] * smoothed_mean_next[:, None, :]

        return smoothed_means, smoothed_covs, smoothed_crosses

    def lgssm_smoother_fast(self, emissions, inputs, filtered_means, filtered_covs, converge_t):
        r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states. Technically, this
        implements the Rauch-Tung-Striebel (RTS) smoother.
        Adopted from Dynamax
        """
        # runs the kalman smooth in 3 dicrete chunks
        # 1) at the end of the data where the smoothed covariance has not converged
        # 2) in the middle of the data where both the filtered and smoothed covs have converged
        #    so we don't need to keep recalculating the covariance and inverting it
        # 3) at the beginning of the data where the filtered covariance and smoothed covariance have not converged

        num_timesteps = emissions.shape[0]

        # Run the Kalman filter
        dynamics_inputs = inputs @ self.dynamics_input_weights.T

        smoothed_means = filtered_means.detach().clone()
        smoothed_covs_end = [filtered_covs[-1, :, :]]
        smoothed_covs_beginning = []
        smoothed_crosses_end = []
        smoothed_crosses_beginning = []

        # Run the smoother backward in time
        # converge_t + 1 is the number of time steps before convergence
        for ti, t in enumerate(reversed(range(num_timesteps - (converge_t + 1), num_timesteps - 1))):
            # Unpack the input
            filtered_mean = filtered_means[t, :]
            filtered_cov = filtered_covs[-1, :, :]
            smoothed_cov_next = smoothed_covs_end[ti]
            smoothed_mean_next = smoothed_means[t + 1, :]

            # Compute the smoothed mean and covariance
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t+1, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            G = torch.linalg.solve(pred_cov, self.dynamics_weights @ filtered_cov).T
            smoothed_covs_end.append(filtered_cov + G @ (smoothed_cov_next - pred_cov) @ G.T)
            smoothed_means[t, :] = filtered_mean + G @ (smoothed_mean_next - pred_mean)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            smoothed_crosses_end.append(G @ smoothed_cov_next)

        for t in reversed(range(converge_t, num_timesteps - (converge_t + 1))):
            # Unpack the input
            filtered_mean = filtered_means[t, :]
            smoothed_mean_next = smoothed_means[t + 1, :]

            # Compute the smoothed mean and covariance
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t+1, :] + self.dynamics_offset

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            smoothed_means[t, :] = filtered_mean + G @ (smoothed_mean_next - pred_mean)

        smoothed_covs_beginning.append(smoothed_covs_end[-1])

        for ti, t in enumerate(reversed(range(0, converge_t))):
            # Unpack the input
            filtered_mean = filtered_means[t, :]
            filtered_cov = filtered_covs[t, :, :]
            smoothed_cov_next = smoothed_covs_beginning[ti]
            smoothed_mean_next = smoothed_means[t + 1, :]

            # Compute the smoothed mean and covariance
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t+1, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            G = torch.linalg.solve(pred_cov, self.dynamics_weights @ filtered_cov).T
            smoothed_covs_beginning.append(filtered_cov + G @ (smoothed_cov_next - pred_cov) @ G.T)
            smoothed_means[t, :] = filtered_mean + G @ (smoothed_mean_next - pred_mean)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            smoothed_crosses_beginning.append(G @ smoothed_cov_next)

        smoothed_covs_beginning = torch.stack(smoothed_covs_beginning).flip(0)
        smoothed_covs_end = torch.stack(smoothed_covs_end).flip(0)
        smoothed_crosses_beginning = torch.stack(smoothed_crosses_beginning).flip(0)
        smoothed_crosses_end = torch.stack(smoothed_crosses_end).flip(0)

        return smoothed_means, (smoothed_covs_beginning, smoothed_covs_end), (smoothed_crosses_beginning, smoothed_crosses_end)

    def get_ll(self, emissions, inputs, init_mean, init_cov):
        # get the log-likelihood of the data
        ll = 0

        for d in range(len(emissions)):
            emissions_torch = torch.tensor(emissions[d], dtype=self.dtype, device=self.device)
            inputs_torch = torch.tensor(inputs[d], dtype=self.dtype, device=self.device)
            init_mean_torch = torch.tensor(init_mean[d], dtype=self.dtype, device=self.device)
            init_cov_torch = torch.tensor(init_cov[d], dtype=self.dtype, device=self.device)

            ll += self.lgssm_filter(emissions_torch, inputs_torch, init_mean_torch, init_cov_torch)[0].detach().cpu().numpy()

        return ll

    def parallel_suff_stats(self, data):
        emissions = data[0]
        inputs = data[1]
        init_mean = data[2]
        init_cov = data[3]
        ll, suff_stats, smoothed_means, smoothed_covs = self.get_suff_stats(emissions, inputs, init_mean, init_cov)

        return ll, suff_stats, smoothed_means, smoothed_covs

    def em_step(self, emissions_list, inputs_list, init_mean_list, init_cov_list, cpu_id=0, num_cpus=1):
        # mm = runMstep_LDSgaussian(yy,uu,mm,zzmu,zzcov,zzcov_d1,optsEM)
        #
        # Run M-step updates for LDS-Gaussian model
        #
        # Inputs
        # =======
        #     yy [ny x T] - Bernoulli observations- design matrix
        #     uu [ns x T] - external inputs
        #     mm [struct] - model structure with fields
        #              .A [nz x nz] - dynamics matrix
        #              .B [nz x ns] - input matrix (optional)
        #              .C [ny x nz] - latents-to-observations matrix
        #              .D [ny x ns] - input-to-observations matrix (optional)
        #              .Q [nz x nz] - latent noise covariance
        #              .Q0 [ny x ny] - latent noise covariance for first latent sample
        #     zzmu [nz x T]        - posterior mean of latents
        #    zzcov [nz*T x nz*T]   -  diagonal blocks of posterior cov over latents
        # zzcov_d1 [nz*T x nz*T-1] - above-diagonal blocks of posterior covariance
        #   optsEM [struct] - optimization params (optional)
        #       .maxiter - maximum # of iterations
        #       .dlogptol - stopping tol for change in log-likelihood
        #       .display - how often to report log-li
        #       .update  - specify which params to update during M step
        #
        # Output
        # =======
        #  mmnew - new model struct with updated parameters
        nz = self.dynamics_dim_full  # number of latents

        if cpu_id == 0:
            data_out = list(zip(emissions_list, inputs_list, init_mean_list, init_cov_list))
            num_data = len(emissions_list)
            chunk_size = int(np.ceil(num_data / num_cpus))
            # split data out into a list of inputs
            data_out = [data_out[i:i+chunk_size] for i in range(0, num_data, chunk_size)]
        else:
            data_out = None

        ll_suff_stats_smoothed_means = []

        data = iu.individual_scatter(data_out, root=0)

        for d in data:
            ll_suff_stats_smoothed_means.append(self.parallel_suff_stats(d))

        ll_suff_stats_smoothed_means = iu.individual_gather(ll_suff_stats_smoothed_means, root=0)

        if cpu_id == 0:
            ll_suff_stats_out = []
            for i in ll_suff_stats_smoothed_means:
                for j in i:
                    ll_suff_stats_out.append(j)

            ll_suff_stats_smoothed_means = ll_suff_stats_out

        if cpu_id == 0:
            log_likelihood = [i[0] for i in ll_suff_stats_smoothed_means]
            log_likelihood = torch.sum(torch.stack(log_likelihood))
            suff_stats = [i[1] for i in ll_suff_stats_smoothed_means]
            smoothed_means = [i[2] for i in ll_suff_stats_smoothed_means]
            smoothed_covs = [i[3] for i in ll_suff_stats_smoothed_means]

            Mz1_list = [i['Mz1'] for i in suff_stats]
            Mz2_list = [i['Mz2'] for i in suff_stats]
            Mz12_list = [i['Mz12'] for i in suff_stats]
            Mu1_list = [i['Mu1'] for i in suff_stats]
            Muz2_list = [i['Muz2'] for i in suff_stats]
            Muz21_list = [i['Muz21'] for i in suff_stats]
            Mzy_list = [i['Mzy'] for i in suff_stats]
            Muy_list = [i['Muy'] for i in suff_stats]
            Mz_list = [i['Mz'] for i in suff_stats]
            Mu2_list = [i['Mu2'] for i in suff_stats]
            Muz_list = [i['Muz'] for i in suff_stats]
            My_list = [i['My'] for i in suff_stats]
            nt = [i['nt'] for i in suff_stats]

            total_time = np.sum(nt)

            Mz1 = torch.sum(torch.stack(Mz1_list), dim=0) / (total_time - len(emissions_list))
            Mz2 = torch.sum(torch.stack(Mz2_list), dim=0) / (total_time - len(emissions_list))
            Mz12 = torch.sum(torch.stack(Mz12_list), dim=0) / (total_time - len(emissions_list))
            Mu1 = torch.sum(torch.stack(Mu1_list), dim=0) / (total_time - len(emissions_list))
            Muz2 = torch.sum(torch.stack(Muz2_list), dim=0) / (total_time - len(emissions_list))
            Muz21 = torch.sum(torch.stack(Muz21_list), dim=0) / (total_time - len(emissions_list))

            Mz = torch.sum(torch.stack(Mz_list), dim=0) / total_time
            Mu2 = torch.sum(torch.stack(Mu2_list), dim=0) / total_time
            Muz = torch.sum(torch.stack(Muz_list), dim=0) / total_time

            Mzy = torch.sum(torch.stack(Mzy_list), dim=0) / total_time
            Muy = torch.sum(torch.stack(Muy_list), dim=0) / total_time
            My = torch.sum(torch.stack(My_list), dim=0) / total_time

            # update dynamics matrix A & input matrix B
            # append the trivial parts of the weights from input lags
            dynamics_eye_pad = torch.eye(self.dynamics_dim * (self.dynamics_lags - 1), device=self.device, dtype=self.dtype)
            dynamics_zeros_pad = torch.zeros((self.dynamics_dim * (self.dynamics_lags - 1), self.dynamics_dim), device=self.device, dtype=self.dtype)
            dynamics_pad = torch.cat((dynamics_eye_pad, dynamics_zeros_pad), dim=1)
            dynamics_inputs_zeros_pad = torch.zeros((self.dynamics_dim * (self.dynamics_lags - 1), self.input_dim_full), device=self.device, dtype=self.dtype)

            ridge_penalty = self.ridge_lambda * torch.diag(self.dynamics_cov)

            if self.param_props['update']['dynamics_weights'] and self.param_props['update']['dynamics_input_weights']:
                # do a joint update for A and B
                Mlin = torch.cat((Mz12, Muz2), dim=0)  # from linear terms
                Mquad = iu.block(((Mz1, Muz21.T), (Muz21, Mu1)), dims=(1, 0))  # from quadratic terms

                if self.param_props['shape']['dynamics_input_weights'] == 'diag':
                    if self.param_props['mask']['dynamics_input_weights'] is None:
                        self.param_props['mask']['dynamics_input_weights'] = torch.eye(self.dynamics_dim, self.input_dim, device=self.device, dtype=torch.bool)

                    # make the of which parameters to fit
                    full_block = torch.ones((self.dynamics_dim_full, self.dynamics_dim), device=self.device, dtype=self.dtype)
                    diag_block = torch.tile(self.param_props['mask']['dynamics_input_weights'].T, (self.dynamics_input_lags, 1))
                    mask = torch.cat((full_block, diag_block), dim=0)

                    ABnew = iu.solve_masked(Mquad.T, Mlin[:, :self.dynamics_dim], mask, ridge_penalty=ridge_penalty).T  # new A and B from regression
                else:
                    if self.ridge_lambda == 0:
                        ABnew = torch.linalg.solve(Mquad.T, Mlin[:, :self.dynamics_dim]).T
                    else:
                        ABnew = iu.solve_masked(Mquad.T, Mlin[:, :self.dynamics_dim], ridge_penalty=ridge_penalty).T

                self.dynamics_weights = ABnew[:, :nz]  # new A
                self.dynamics_input_weights = ABnew[:, nz:]

                self.dynamics_weights = torch.cat((self.dynamics_weights, dynamics_pad), dim=0)  # new A
                self.dynamics_input_weights = torch.cat((self.dynamics_input_weights, dynamics_inputs_zeros_pad), dim=0)  # new B

                # check the largest eigenvalue of the dynamics matrix
                max_abs_eig = torch.max(torch.abs(torch.linalg.eigvals(self.dynamics_weights)))
                if max_abs_eig >= 1:
                    warnings.warn('Largest eigenvalue of the dynamics matrix is:' + str(max_abs_eig))

            elif self.param_props['update']['dynamics_weights']:  # update dynamics matrix A only
                if self.ridge_lambda == 0:
                    self.dynamics_weights = torch.linalg.solve(Mz1.T, (Mz12 - Muz21.T @ self.dynamics_input_weights.T)[:, :self.dynamics_dim]).T  # new A
                else:
                    self.dynamics_weights = iu.solve_masked(Mz1.T, (Mz12 - Muz21.T @ self.dynamics_input_weights.T)[:, :self.dynamics_dim], ridge_penalty=ridge_penalty).T  # new A

                self.dynamics_weights = torch.cat((self.dynamics_weights, dynamics_pad), dim=0)  # new A

            elif self.param_props['update']['dynamics_input_weights']:  # update input matrix B only
                if self.param_props['shape']['dynamics_input_weights'] == 'diag':
                    if self.param_props['mask']['dynamics_input_weights'] is None:
                        self.param_props['mask']['dynamics_input_weights'] = torch.eye(self.dynamics_dim, self.input_dim, device=self.device, dtype=torch.bool)

                    # make the of which parameters to fit
                    mask = torch.tile(self.param_props['mask']['dynamics_input_weights'].T, (self.dynamics_input_lags, 1))

                    self.dynamics_input_weights = iu.solve_masked(Mu1.T, (Muz2 - Muz21 @ self.dynamics_weights.T)[:, :self.dynamics_dim], mask).T  # new A and B from regression
                else:
                    self.dynamics_input_weights = torch.linalg.solve(Mu1.T, (Muz2 - Muz21 @ self.dynamics_weights.T)[:, :self.dynamics_dim]).T  # new B

                self.dynamics_input_weights = torch.cat((self.dynamics_input_weights, dynamics_inputs_zeros_pad), dim=0)  # new B

            # Update noise covariance Q
            if self.param_props['update']['dynamics_cov']:
                self.dynamics_cov = (Mz2 + self.dynamics_weights @ Mz1 @ self.dynamics_weights.T + self.dynamics_input_weights @ Mu1 @ self.dynamics_input_weights.T
                                     - self.dynamics_weights @ Mz12 - Mz12.T @ self.dynamics_weights.T
                                     - self.dynamics_input_weights @ Muz2 - Muz2.T @ self.dynamics_input_weights.T
                                     + self.dynamics_weights @ Muz21.T @ self.dynamics_input_weights.T + self.dynamics_input_weights @ Muz21 @ self.dynamics_weights.T) #/ (nt - 1)

                if self.param_props['shape']['dynamics_cov'] == 'diag':
                    self.dynamics_cov = torch.diag(torch.diag(self.dynamics_cov))

                self.dynamics_cov = 0.5 * self.dynamics_cov + 0.5 * self.dynamics_cov.T

            # update obs matrix C & input matrix D
            if self.param_props['update']['emissions_weights'] and self.param_props['update']['emissions_input_weights']:
                # do a joint update to C and D
                Mlin = torch.cat((Mzy, Muy), dim=0)  # from linear terms
                Mquad = iu.block([[Mz, Muz.T], [Muz, Mu2]], dims=(1, 0))  # from quadratic terms
                CDnew = torch.linalg.solve(Mquad.T, Mlin).T  # new A and B from regression
                self.emissions_weights = CDnew[:, :nz]  # new A
                self.emissions_input_weights = CDnew[:, nz:]  # new B
            elif self.param_props['update']['emissions_weights']:  # update C only
                Cnew = torch.linalg.solve(Mz.T, Mzy - Muz.T @ self.emissions_input_weights.T).T  # new A
                self.emissions_weights = Cnew
            elif self.param_props['update']['emissions_input_weights']:  # update D only
                Dnew = torch.linalg.solve(Mu2.T, Muy - Muz @ self.emissions_weights.T).T  # new B
                self.emissions_input_weights = Dnew

            # update obs noise covariance R
            if self.param_props['update']['emissions_cov']:
                self.emissions_cov = (My + self.emissions_weights @ Mz @ self.emissions_weights.T + self.emissions_input_weights @ Mu2 @ self.emissions_input_weights.T
                                      - self.emissions_weights @ Mzy - Mzy.T @ self.emissions_weights.T
                                      - self.emissions_input_weights @ Muy - Muy.T @ self.emissions_input_weights.T
                                      + self.emissions_weights @ Muz.T @ self.emissions_input_weights.T + self.emissions_input_weights @ Muz @ self.emissions_weights.T)

                if self.param_props['shape']['emissions_cov'] == 'diag':
                    self.emissions_cov = torch.diag(torch.diag(self.emissions_cov))

                self.emissions_cov = 0.5 * self.emissions_cov + 0.5 * self.emissions_cov.T

            if not torch.all(self.dynamics_cov == self.dynamics_cov.T):
                warnings.warn('dynamics_cov is not symmetric')

            if not torch.all(self.emissions_cov == self.emissions_cov.T):
                warnings.warn('emissions_cov is not symmetric')

            return log_likelihood, smoothed_means, smoothed_covs
        return None, None, None

    def get_suff_stats(self, emissions, inputs, init_mean, init_cov):
        nt = emissions.shape[0]

        ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses = \
            self.lgssm_smoother(emissions, inputs, init_mean, init_cov)

        # y = torch.where(torch.isnan(emissions), 0, emissions)
        y_nan_loc = torch.isnan(emissions)
        y = torch.where(y_nan_loc, (self.emissions_weights @ smoothed_means.T).T, emissions)

        # =============== Update dynamics parameters ==============
        # Compute sufficient statistics for latents
        if type(smoothed_covs) is tuple:
            covs_summed = smoothed_covs[0][1:, :, :].sum(0) + smoothed_covs[1][:-1, :, :].sum(0) + smoothed_covs[0][-1, :, :] * (nt - 2 * smoothed_covs[0].shape[0])
            crosses_summed = smoothed_crosses[0].sum(0) + smoothed_crosses[1].sum(0) + smoothed_crosses[0][-1, :, :] * (nt - 2 * smoothed_crosses[0].shape[0] - 1)
            first_cov = smoothed_covs[0][0, :, :]
            last_cov = smoothed_covs[1][-1, :, :]
        else:
            covs_summed = smoothed_covs[1:-1, :, :].sum(0)
            crosses_summed = smoothed_crosses[1:-1, :, :].sum(0)
            first_cov = smoothed_covs[0, :, :]
            last_cov = smoothed_covs[-1, :, :]

        Mz1 = covs_summed + first_cov + smoothed_means[:-1, :].T @ smoothed_means[:-1, :]  # E[zz@zz'] for 1 to T-1
        Mz2 = covs_summed + last_cov + smoothed_means[1:, :].T @ smoothed_means[1:, :]  # E[zz@zz'] for 2 to T
        Mz12 = crosses_summed + smoothed_means[:-1, :].T @ smoothed_means[1:, :]  # E[zz_t@zz_{t+1}'] (above-diag)

        # Compute sufficient statistics for inputs x latents
        Mu1 = inputs[1:, :].T @ inputs[1:, :]  # E[uu@uu'] for 2 to T
        Muz2 = inputs[1:, :].T @ smoothed_means[1:, :]  # E[uu@zz'] for 2 to T
        Muz21 = inputs[1:, :].T @ smoothed_means[:-1, :]  # E[uu_t@zz_{t-1} for 2 to T

        # =============== Update observation parameters ==============
        # Compute sufficient statistics
        Mz_emis = last_cov + smoothed_means[-1, :, None] * smoothed_means[-1, None, :]  # re-use Mz1 if possible
        Mu_emis = inputs[0, :, None] * inputs[0, None, :]  # reuse Mu
        Muz_emis = inputs[0, :, None] * smoothed_means[0, None, :]  # reuse Muz
        Muy = inputs.T @ y  # E[uu@yy']

        Mzy = torch.zeros((smoothed_means.shape[1], y.shape[1]), device=self.device, dtype=self.dtype)
        My = torch.zeros((y.shape[1], y.shape[1]), device=self.device, dtype=self.dtype)

        num_time = y.shape[0]
        use_fast_version = self._has_no_scattered_nans(emissions)

        if use_fast_version:
            y_nan_loc_t = y_nan_loc[0, :]
            c_nan = self.emissions_weights[y_nan_loc_t, :]
            r_nan = self.emissions_cov[np.ix_(y_nan_loc_t, y_nan_loc_t)]

            if type(smoothed_covs) is tuple:
                # smoothed covs are only stored til they converge so we have covs for the beginning and end of the data
                converge_size = smoothed_covs[0].shape[0]
                num_middle_covs = num_time - 2 * converge_size

                imputed_variance_my = torch.sum(c_nan @ smoothed_covs[0] @ c_nan.T + r_nan, dim=0)
                imputed_variance_my += torch.sum(c_nan @ smoothed_covs[1] @ c_nan.T + r_nan, dim=0)
                imputed_variance_my += num_middle_covs * (c_nan @ smoothed_covs[0][-1, :, :] @ c_nan.T + r_nan)

                imputed_variance_mzy = torch.sum(smoothed_covs[0] @ c_nan.T, dim=0)
                imputed_variance_mzy += torch.sum(smoothed_covs[1] @ c_nan.T, dim=0)
                imputed_variance_mzy += num_middle_covs * (smoothed_covs[0][-1, :, :] @ c_nan.T)
            else:
                imputed_variance_my = torch.sum(c_nan @ smoothed_covs @ c_nan.T + r_nan, dim=0)
                imputed_variance_mzy = torch.sum(smoothed_covs @ c_nan.T, dim=0)

            # add in the variance from y
            # add in the variance from all the values of y you imputed
            My = y.T @ y
            My[np.ix_(y_nan_loc_t, y_nan_loc_t)] += imputed_variance_my

            # add the covariance between the means and y
            # additionally add the variance from the inferred neurons
            Mzy = smoothed_means.T @ y
            Mzy[:, y_nan_loc_t] += imputed_variance_mzy

        else:
            Mzy = torch.zeros((smoothed_means.shape[1], y.shape[1]), device=self.device, dtype=self.dtype)
            My = torch.zeros((y.shape[1], y.shape[1]), device=self.device, dtype=self.dtype)

            for t in range(num_time):
                y_nan_loc_t = y_nan_loc[t, :]
                c_nan = self.emissions_weights[y_nan_loc_t, :]
                r_nan = self.emissions_cov[np.ix_(y_nan_loc_t, y_nan_loc_t)]

                if type(smoothed_covs) is tuple:
                    # smoothed covs are only stored til they converge so we have covs for the beginning and end of the data
                    converge_size = smoothed_covs[0].shape[0]
                    if t < converge_size:
                        this_cov = smoothed_covs[0][t, :, :]
                    elif num_time - t <= converge_size:
                        ti = converge_size - (num_time - t)
                        this_cov = smoothed_covs[1][ti, :, :]
                    else:
                        this_cov = smoothed_covs[0][-1, :, :]
                else:
                    this_cov = smoothed_covs[t, :, :]

                # add in the variance from y
                My += y[t, :, None] * y[t, :, None].T
                # add in the variance from all the values of y you imputed
                My[np.ix_(y_nan_loc_t, y_nan_loc_t)] += c_nan @ this_cov @ c_nan.T + r_nan

                # add the covariance between the means and y
                Mzy += smoothed_means[t, :, None] * y[t, :, None].T
                # additionally add the variance from the inferred neurons
                Mzy[:, y_nan_loc_t] += this_cov @ c_nan.T

        Mz = Mz1 + Mz_emis
        Mu2 = Mu1 + Mu_emis
        Muz = Muz2 + Muz_emis

        suff_stats = {'Mz1': Mz1,
                      'Mz2': Mz2,
                      'Mz12': Mz12,
                      'Mu1': Mu1,
                      'Muz2': Muz2,
                      'Muz21': Muz21,

                      'Mzy': Mzy,
                      'Muy': Muy,
                      'My': My,
                      'Mz': Mz,
                      'Mu2': Mu2,
                      'Muz': Muz,

                      'nt': nt,
                      }

        return ll, suff_stats, smoothed_means, smoothed_covs

    def _pad_init_for_lags(self):
        self.dynamics_weights_init = self._get_lagged_weights(self.dynamics_weights_init, self.dynamics_lags, fill='eye')
        self.dynamics_input_weights_init = self._get_lagged_weights(self.dynamics_input_weights_init, self.dynamics_lags, fill='zeros')
        self.dynamics_offset_init = self._pad_zeros(self.dynamics_offset_init, self.dynamics_lags, axis=0)
        dci_block = self.dynamics_cov_init
        self.dynamics_cov_init = np.eye(self.dynamics_dim_full) / self.epsilon
        self.dynamics_cov_init[:self.dynamics_dim, :self.dynamics_dim] = dci_block

        self.emissions_weights_init = self._pad_zeros(self.emissions_weights_init, self.dynamics_lags, axis=1)
        self.emissions_input_weights_init = self._pad_zeros(self.emissions_input_weights_init, self.dynamics_input_lags, axis=1)

    def estimate_init_mean(self, emissions):
        # estimate the initial mean of a data set as the mean over time
        init_mean_list = []

        for i in emissions:
            init_mean = torch.nanmean(i, dim=0)
            init_mean[torch.isnan(init_mean)] = torch.nanmean(init_mean)
            # repeat the mean for each delay you have
            init_mean = torch.tile(init_mean, [self.dynamics_lags])
            init_mean_list.append(init_mean)

        return init_mean_list

    def estimate_init_cov(self, emissions):
        init_cov_list = []

        for i in emissions:
            emissions_mean = torch.nanmean(i, dim=0)
            emissions_mean_sub = i - emissions_mean
            emissions_var = torch.nansum(emissions_mean_sub**2, dim=0) / (i.shape[0] - 1)
            emissions_var[emissions_var == 0] = torch.nanmean(emissions_var[emissions_var != 0])
            var_mat = torch.diag(emissions_var)
            var_block = torch.block_diag(*([var_mat] * self.dynamics_lags))
            init_cov_list.append(var_block)

        return init_cov_list

    @staticmethod
    def get_lagged_data(data, lags, add_pad=True):
        num_time, num_neurons = data.shape

        if type(data) is np.ndarray:
            cat_fun = np.concatenate
            zero_fun = np.zeros
        else:
            cat_fun = torch.cat
            zero_fun = torch.zeros

        if add_pad:
            final_time = num_time
            pad = zero_fun((lags - 1, num_neurons))
            data = cat_fun((pad, data), 0)
        else:
            final_time = num_time - lags + 1

        lagged_data = zero_fun((final_time, 0), dtype=data.dtype)

        for tau in reversed(range(lags)):
            if tau == lags-1:
                lagged_data = cat_fun((lagged_data, data[tau:, :]), 1)
            else:
                lagged_data = cat_fun((lagged_data, data[tau:-lags + tau + 1, :]), 1)

        return lagged_data

    @staticmethod
    def _get_lagged_weights(weights, lags_out, fill='eye'):
        if type(weights) is np.ndarray:
            cat_fun = np.concatenate
            split_fun = np.split
            eye_fun = np.eye
            zeros_fun = np.zeros
            split_num = weights.shape[0]
        else:
            cat_fun = torch.cat
            split_fun = torch.split
            eye_fun = torch.eye
            zeros_fun = torch.zeros
            split_num = 1

        num_lags = weights.shape[0]
        lagged_weights = cat_fun(split_fun(weights, split_num, 0), 2)[0, :, :]

        if fill == 'eye':
            fill_mat = eye_fun(lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1])
        elif fill == 'zeros':
            fill_mat = zeros_fun((lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1]))
        else:
            raise Exception('fill value not recognized')

        lagged_weights = cat_fun((lagged_weights, fill_mat), 0)

        return lagged_weights

    @staticmethod
    def _pad_zeros(weights, tau, axis=1):
        if type(weights) is np.ndarray:
            zeros_fun = np.zeros
            cat_fun = np.concatenate
        else:
            zeros_fun = torch.zeros
            cat_fun = torch.cat

        zeros_shape = list(weights.shape)
        zeros_shape[axis] = zeros_shape[axis] * (tau - 1)

        zero_pad = zeros_fun(zeros_shape)

        return cat_fun((weights, zero_pad), axis)

    @staticmethod
    def _has_no_scattered_nans(emissions):
        any_nan_neurons = torch.any(torch.isnan(emissions), dim=0)
        all_nan_neurons = torch.all(torch.isnan(emissions), dim=0)
        return torch.all(any_nan_neurons == all_nan_neurons).numpy()

