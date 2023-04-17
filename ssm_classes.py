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

    def __init__(self, dynamics_dim, emissions_dim, input_dim, param_props=None, dtype=torch.float64, device='cpu', verbose=True):
        self.dynamics_dim = dynamics_dim
        self.emissions_dim = emissions_dim
        self.input_dim = input_dim
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.log_likelihood = None
        self.train_time = None

        self.param_props = {'update': {'dynamics_weights': True,
                                       'dynamics_input_weights': True,
                                       'dynamics_offset': True,
                                       'dynamics_cov': True,
                                       'emissions_weights': True,
                                       'emissions_input_weights': True,
                                       'emissions_offset': True,
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
                            'tau': {'dynamics_weights': 1,
                                    'dynamics_input_weights': 1,
                                    }
                            }

        if param_props is not None:
            for k in self.param_props.keys():
                self.param_props[k].update(param_props[k])

        # initialize dynamics weights
        dynamics_tau = self.param_props['tau']['dynamics_weights']
        self.dynamics_weights_init = 0.9 * np.tile(np.eye(self.dynamics_dim), (dynamics_tau, 1, 1))

        dynamics_input_tau = self.param_props['tau']['dynamics_input_weights']
        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            self.dynamics_input_weights_init = np.zeros((dynamics_input_tau, self.dynamics_dim))
        else:
            self.dynamics_input_weights_init = np.zeros((dynamics_input_tau, self.dynamics_dim, self.input_dim))

        self.dynamics_offset_init = np.zeros(self.dynamics_dim)

        if self.param_props['shape']['dynamics_cov'] == 'diag':
            self.dynamics_cov_init = np.zeros(self.dynamics_dim)
        else:
            self.dynamics_cov_init = np.eye(self.dynamics_dim)

        # initialize emissions weights
        self.emissions_weights_init = np.eye(self.emissions_dim)
        self.emissions_input_weights_init = np.zeros((self.emissions_dim, self.input_dim))
        self.emissions_offset_init = np.zeros(self.emissions_dim)

        if param_props['shape']['emissions_cov'] == 'diag':
            self.emissions_cov_init = np.zeros(self.dynamics_dim)
        else:
            self.emissions_cov_init = np.eye(self.dynamics_dim)

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
        dynamics_tau = self.param_props['tau']['dynamics_weights']
        self.dynamics_weights_init = rng.standard_normal((dynamics_tau, self.dynamics_dim, self.dynamics_dim))
        eig_vals, eig_vects = np.linalg.eig(self.dynamics_weights_init)
        eig_vals = eig_vals / np.max(np.abs(eig_vals)) * max_eig_allowed
        negative_real_eigs = np.real(eig_vals) < 0
        eig_vals[negative_real_eigs] = -eig_vals[negative_real_eigs]
        eig_vals_mat = np.zeros((dynamics_tau, self.dynamics_dim, self.dynamics_dim), dtype=np.cdouble)
        for i in range(dynamics_tau):
            eig_vals_mat[i, :, :] = np.diag(eig_vals[i, :])
        self.dynamics_weights_init = np.real(eig_vects @ np.linalg.solve(np.transpose(eig_vects, (0, 2, 1)), eig_vals_mat))

        dynamics_input_tau = self.param_props['tau']['dynamics_input_weights']
        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            self.dynamics_input_weights_init = init_std * rng.standard_normal((dynamics_input_tau, self.dynamics_dim))
        else:
            self.dynamics_input_weights_init = init_std * rng.standard_normal((dynamics_input_tau, self.dynamics_dim, self.input_dim))

        self.dynamics_offset_init = 0 * rng.standard_normal(self.dynamics_dim)

        if self.param_props['shape']['dynamics_cov'] == 'diag':
            self.dynamics_cov_init = init_std * rng.standard_normal(self.dynamics_dim)
        else:
            self.dynamics_cov_init = init_std * rng.standard_normal((self.dynamics_dim, self.dynamics_dim))
            self.dynamics_cov_init = self.dynamics_cov_init.T @ self.dynamics_cov_init / self.dynamics_dim + np.eye(self.dynamics_dim)

        # randomize emissions weights
        self.emissions_weights_init = init_std * rng.standard_normal((self.emissions_dim, self.dynamics_dim))
        self.emissions_input_weights_init = init_std * rng.standard_normal((self.dynamics_dim, self.input_dim))
        self.emissions_offset_init = 0 * rng.standard_normal(self.dynamics_dim)

        if self.param_props['shape']['emissions_cov'] == 'diag':
            self.emissions_cov_init = init_std * rng.standard_normal(self.emissions_dim)
        else:
            self.emissions_cov_init = init_std * rng.standard_normal((self.emissions_dim, self.emissions_dim))
            self.emissions_cov_init = self.emissions_cov_init.T @ self.emissions_cov_init / self.emissions_dim + np.eye(self.emissions_dim)

        self._set_to_init()

    def _set_to_init(self):
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

        emissions, input = self._stack_data(emissions_list, input_list)

        return emissions, input

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

    def sample(self, num_time=100, init_mean=None, init_cov=None, num_data_sets=None, nan_freq=0.0, rng=np.random.default_rng()):
        dynamics_weights_tau = self.param_props['tau']['dynamics_weights']
        dynamics_input_weights_tau = self.param_props['tau']['dynamics_input_weights']

        # generate a random initial mean and covariance
        if init_mean is None:
            init_mean = rng.standard_normal((num_data_sets, self.dynamics_dim * dynamics_weights_tau))

        if init_cov is None:
            init_cov = rng.standard_normal((num_data_sets, self.dynamics_dim * dynamics_weights_tau, self.dynamics_dim * dynamics_weights_tau))
            init_cov = np.transpose(init_cov, [0, 2, 1]) @ init_cov / (self.dynamics_dim * dynamics_weights_tau)

        latents = np.zeros((num_data_sets, num_time, self.dynamics_dim * dynamics_weights_tau))
        emissions = np.zeros((num_data_sets, num_time, self.emissions_dim))
        inputs = rng.standard_normal((num_data_sets, num_time + dynamics_input_weights_tau - 1, self.input_dim))
        inputs = self._get_lagged_data(inputs, dynamics_input_weights_tau)

        # set up the weights, creating full matricies from diagonals and padding for higher order AR dynamics
        dynamics_weights = self._get_lagged_weights(self.dynamics_weights)

        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            dynamics_input_weights = torch.diag_embed(torch.exp(self.dynamics_input_weights))
        else:
            dynamics_input_weights = self.dynamics_input_weights
        dynamics_input_weights = self._get_lagged_weights(dynamics_input_weights, fill='zeros')

        dynamics_offset = self._pad_zeros(self.dynamics_offset, dynamics_weights_tau, axis=0)

        if self.param_props['shape']['dynamics_cov'] == 'diag':
            dynamics_cov = torch.diag_embed(torch.exp(self.dynamics_cov))
        else:
            dynamics_cov = self.dynamics_cov

        emissions_weights = self._pad_zeros(self.emissions_weights, dynamics_weights_tau, axis=1)

        if self.param_props['shape']['emissions_cov'] == 'diag':
            emissions_cov = torch.diag(torch.exp(self.emissions_cov))
        else:
            emissions_cov = self.emissions_cov

        # get the initial observations
        dynamics_noise = rng.multivariate_normal(np.zeros(self.dynamics_dim), dynamics_cov, size=(num_data_sets, num_time))
        dynamics_noise = self._pad_zeros(dynamics_noise, dynamics_weights_tau, axis=2)
        emissions_noise = rng.multivariate_normal(np.zeros(self.dynamics_dim), emissions_cov, size=(num_data_sets, num_time))
        dynamics_inputs = (dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]
        emissions_inputs = (self.emissions_input_weights @ inputs[:, :, :self.input_dim, None])[:, :, :, 0]
        latent_init = np.zeros((num_data_sets, self.dynamics_dim * dynamics_weights_tau))

        for d in range(num_data_sets):
            latent_init[d, :] = rng.multivariate_normal(init_mean[d, :], init_cov[d, :, :])

        latents[:, 0, :] = utils.batch_Ax(dynamics_weights, latent_init) + \
                           dynamics_inputs[:, 0, :] + \
                           dynamics_offset[None, :] + \
                           dynamics_noise[:, 0, :]

        emissions[:, 0, :] = utils.batch_Ax(emissions_weights, latents[:, 0, :]) + \
                             emissions_inputs[:, 0, :] + \
                             self.emissions_offset[None, :] + \
                             emissions_noise[:, 0, :]

        # loop through time and generate the latents and emissions
        for t in range(1, num_time):
            latents[:, t, :] = (dynamics_weights @ latents[:, t-1, :, None])[:, :, 0] + \
                                dynamics_inputs[:, t, :] + \
                                dynamics_offset[None, :] + \
                                dynamics_noise[:, t, :]

            emissions[:, t, :] = (emissions_weights @ latents[:, t, :, None])[:, :, 0] + \
                                  emissions_inputs[:, t, :] + \
                                  self.emissions_offset[None, :] + \
                                  emissions_noise[:, t, :]

        # crop the data back to the correct dimension size
        latents = latents[:, :, :self.dynamics_dim]
        inputs = inputs[:, :, :self.input_dim]
        emissions = emissions[:, :, :self.emissions_dim]

        # add in nans
        nan_mask = rng.random((num_data_sets, num_time, self.dynamics_dim)) <= nan_freq
        emissions[nan_mask] = np.nan

        latents = [i for i in latents]
        inputs = [i for i in inputs]
        emissions = [i for i in emissions]
        init_mean = [i for i in init_mean]
        init_cov = [i for i in init_cov]

        data_dict = {'latents': latents,
                     'inputs': inputs,
                     'emissions': emissions,
                     'init_mean': init_mean,
                     'init_cov': init_cov,
                     }

        return data_dict

    def fit_em(self, emissions_list, inputs_list, init_mean=None, init_cov=None, num_steps=10):
        emissions, inputs = self.standardize_inputs(emissions_list, inputs_list)

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

        log_likelihood_out = []
        time_out = []

        start = time.time()
        for ep in range(num_steps):
            # ll, init_stats, dynamics_stats, emission_stats = self.e_step(emissions, inputs, init_mean, init_cov)
            # self.m_step(init_stats, dynamics_stats, emission_stats)

            ll = self.em_step_pillow(emissions, inputs, init_mean, init_cov)

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
        filtered_mean = init_mean
        filtered_cov = init_cov

        # for k in self.param_props['shape'].keys():
        #     if k == 'diag':
        #         exec("%s = %d" % (k, torch.diag(torch.exp(self.dynamics_input_weights))))
        #     else:
        #         exec("%s = %d" % (k, self.dynamics_input_weights))
        #
        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            dynamics_input_weights = torch.diag(torch.exp(self.dynamics_input_weights))
        else:
            dynamics_input_weights = self.dynamics_input_weights

        if self.param_props['shape']['dynamics_cov'] == 'diag':
            dynamics_cov = torch.diag(torch.exp(self.dynamics_cov))
        else:
            dynamics_cov = self.dynamics_cov

        if self.param_props['shape']['emissions_cov'] == 'diag':
            emissions_cov = torch.diag(torch.exp(self.emissions_cov))
        else:
            emissions_cov = self.emissions_cov

        dynamics_inputs = (dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]
        emissions_inputs = (self.emissions_input_weights @ inputs[:, :, :, None])[:, :, :, 0]

        filtered_means_list = []
        filtered_covs_list = []

        for t in range(num_timesteps):
            # Shorthand: get parameters and input for time index t
            y = emissions[:, t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)
            R = torch.where(torch.diag_embed(nan_loc), nan_fill, torch.tile(emissions_cov, (num_data_sets, 1, 1)))

            # Predict the next state
            pred_mean = utils.batch_Ax(self.dynamics_weights, filtered_mean) + dynamics_inputs[:, t, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + dynamics_cov

            # Update the log likelihood
            ll_mu = utils.batch_Ax(self.emissions_weights, pred_mean) + emissions_inputs[:, t, :] + self.emissions_offset[None, :]
            ll_cov = self.emissions_weights @ pred_cov @ self.emissions_weights.T + R

            # ll += torch.distributions.multivariate_normal.MultivariateNormal(ll_mu, ll_cov).log_prob(y)
            mean_diff = ll_mu - y
            ll += -1/2 * (emissions.shape[2] * np.log(2*np.pi) + torch.linalg.slogdet(ll_cov)[1] +
                          utils.batch_Ax(mean_diff[:, None, :], torch.linalg.solve(ll_cov, mean_diff))[:, 0])

            # Condition on this emission
            # Compute the Kalman gain
            S = R + self.emissions_weights @ pred_cov @ self.emissions_weights.T
            K = utils.batch_trans(torch.linalg.solve(S, self.emissions_weights @ pred_cov))

            filtered_cov = pred_cov - K @ S @ utils.batch_trans(K)
            filtered_mean = pred_mean + utils.batch_Ax(K, (y - ll_mu))

            filtered_covs_list.append(filtered_cov)
            filtered_means_list.append(filtered_mean)

        filtered_means = torch.permute(torch.stack(filtered_means_list), (1, 0, 2))
        filtered_covs = torch.permute(torch.stack(filtered_covs_list), (1, 0, 2, 3))

        ll = ll.sum() / emissions.numel()

        return ll, filtered_means, filtered_covs

    def lgssm_smoother(self, emissions, inputs, init_mean, init_cov):
        r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states. Technically, this
        implements the Rauch-Tung-Striebel (RTS) smoother.
        Adopted from Dynamax
        """
        num_timesteps = emissions.shape[1]

        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            dynamics_input_weights = torch.diag(torch.exp(self.dynamics_input_weights))
        else:
            dynamics_input_weights = self.dynamics_input_weights

        if self.param_props['shape']['dynamics_cov'] == 'diag':
            dynamics_cov = torch.diag(torch.exp(self.dynamics_cov))
        else:
            dynamics_cov = self.dynamics_cov

        # Run the Kalman filter
        ll, filtered_means, filtered_covs = self.lgssm_filter(emissions, inputs, init_mean, init_cov)

        dynamics_inputs = (dynamics_input_weights @ inputs[:, :, :, None])[:, :, :, 0]

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

            # Compute the smoothed mean and covariance
            pred_mean = utils.batch_Ax(self.dynamics_weights, filtered_mean) + dynamics_inputs[:, t+1, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + dynamics_cov

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            G = utils.batch_trans(torch.linalg.solve(pred_cov, self.dynamics_weights @ filtered_cov))
            smoothed_covs[:, t, :, :] = filtered_cov + G @ (smoothed_cov_next - pred_cov) @ utils.batch_trans(G)
            smoothed_means[:, t, :] = filtered_mean + utils.batch_Ax(G, smoothed_mean_next - pred_mean)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            # TODO: ask why the second expression is not in jonathan's code
            smoothed_crosses[:, t, :, :] = G @ smoothed_cov_next #+ smoothed_means[:, t, :, None] * smoothed_mean_next[:, None, :]

        return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses

    def e_step(self, emissions, inputs, init_mean, init_cov):
        num_data_sets, num_timesteps = emissions.shape[:2]

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

        # Append bias to the input
        inputs = torch.cat((inputs, torch.ones((num_data_sets, num_timesteps, 1))), dim=2)
        up = inputs[:, :-1, :]
        u = inputs
        y = emissions
        y = torch.where(torch.isnan(y), 0, y)

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = smoothed_means[:, 0, :]
        Ex0x0T = smoothed_covs[:, 0, :, :] + Ex0[:, :, None] @ Ex0[:, None, :]
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

    def em_step_pillow(self, emissions, inputs, init_mean, init_cov):
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

        ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses = \
            self.lgssm_smoother(emissions, inputs, init_mean, init_cov)

        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            dynamics_input_weights = torch.diag(torch.exp(self.dynamics_input_weights))
        else:
            dynamics_input_weights = self.dynamics_input_weights

        uu = inputs
        yy = emissions
        yy = torch.where(torch.isnan(yy), 0, yy)
        zzmu = smoothed_means
        zzcov = smoothed_covs
        zzcov_d1 = smoothed_crosses
        num_timesteps = torch.sum(~torch.all(torch.isnan(emissions), dim=2), dim=1)
        data_weights = 1 / num_timesteps

        # Extract sizes
        nz = self.dynamics_dim  # number of latents

        # =============== Update dynamics parameters ==============
        # Compute sufficient statistics for latents
        Mz1 = zzcov[:, :-1, :, :].sum(1) + utils.batch_trans(zzmu[:, :-1, :]) @ zzmu[:, :-1, :]  # E[zz@zz'] for 1 to T-1
        Mz2 = zzcov[:, 1:, :, :].sum(1) + utils.batch_trans(zzmu[:, 1:, :]) @ zzmu[:, 1:, :]  # E[zz@zz'] for 2 to T
        Mz12 = zzcov_d1.sum(1) + utils.batch_trans(zzmu[:, :-1, :]) @ zzmu[:, 1:, :]  # E[zz_t@zz_{t+1}'] (above-diag)

        # Compute sufficient statistics for inputs x latents
        Mu = utils.batch_trans(uu[:, 1:, :]) @ uu[:, 1:, :]  # E[uu@uu'] for 2 to T
        Muz2 = utils.batch_trans(uu[:, 1:, :]) @ zzmu[:, 1:, :]  # E[uu@zz'] for 2 to T
        Muz21 = utils.batch_trans(uu[:, 1:, :]) @ zzmu[:, :-1, :]  # E[uu_t@zz_{t-1} for 2 to T

        # sum the statistics across batches
        Mz1 = (Mz1 * data_weights[:, None, None]).mean(0)
        Mz2 = (Mz2 * data_weights[:, None, None]).mean(0)
        Mz12 = (Mz12 * data_weights[:, None, None]).mean(0)

        Mu = (Mu * data_weights[:, None, None]).mean(0)
        Muz2 = (Muz2 * data_weights[:, None, None]).mean(0)
        Muz21 = (Muz21 * data_weights[:, None, None]).mean(0)

        # update dynamics matrix A & input matrix B
        if self.param_props['update']['dynamics_weights'] and self.param_props['update']['dynamics_input_weights']:
            # do a joint update for A and B
            Mlin = torch.cat((Mz12, Muz2), dim=0)  # from linear terms
            Mquad = utils.block(((Mz1, Muz21.T), (Muz21, Mu)), dims=(1, 0))  # from quadratic terms
            ABnew = torch.linalg.solve(Mquad.T, Mlin).T  # new A and B from regression
            self.dynamics_weights = ABnew[:, :nz]  # new A
            dynamics_input_weights = ABnew[:, nz:]  # new B

        elif self.param_props['update']['dynamics_weights']:  # update dynamics matrix A only
            Anew = torch.linalg.solve(Mz1.T, Mz12 - Muz21.T @ dynamics_input_weights.T).T  # new A
            self.dynamics_weights = Anew

        elif self.param_props['update']['dynamics_input_weights']:  # update input matrix B only
            Bnew = torch.linalg.solve(Mu.T, Muz2 - Muz21 @ self.dynamics_weights.T).T  # new B
            dynamics_input_weights = Bnew

        # TODO: fix this, this is a hack to force diagonal
        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            self.dynamics_input_weights = torch.log(torch.diag(dynamics_input_weights))
            dynamics_input_weights = torch.diag(torch.exp(self.dynamics_input_weights))
        else:
            self.dynamics_input_weights = dynamics_input_weights

        # Update noise covariance Q
        if self.param_props['update']['dynamics_cov']:
            self.dynamics_cov = (Mz2 + self.dynamics_weights @ Mz1 @ self.dynamics_weights.T + dynamics_input_weights @ Mu @ dynamics_input_weights.T
                                - self.dynamics_weights @ Mz12 - Mz12.T @ self.dynamics_weights.T
                                - dynamics_input_weights @ Muz2 - Muz2.T @ dynamics_input_weights.T
                                + self.dynamics_weights @ Muz21.T @ dynamics_input_weights.T + dynamics_input_weights @ Muz21 @ self.dynamics_weights.T) # / ((nt - 1) * yy.shape[0])


        if self.param_props['shape']['dynamics_cov'] == 'diag':
            self.dynamics_cov = torch.log(torch.diag(self.dynamics_cov))


        # =============== Update observation parameters ==============
        # Compute sufficient statistics
        Mz_emis = zzcov[:, -1, :, :] + zzmu[:, -1, :, None] * zzmu[:, -1, None, :]  # re-use Mz1 if possible
        Mu_emis = uu[:, 0, :, None] @ uu[:, 0, None, :]  # reuse Mu
        Muz_emis = uu[:, 0, :, None] @ zzmu[:, 0, None, :]  # reuse Muz
        Mzy = utils.batch_trans(zzmu) @ yy  # E[zz@yy']
        Muy = utils.batch_trans(uu) @ yy  # E[uu@yy']

        Mz_emis = (Mz_emis * data_weights[:, None, None]).mean(0)
        Mu_emis = (Mu_emis * data_weights[:, None, None]).mean(0)
        Muz_emis = (Muz_emis * data_weights[:, None, None]).mean(0)
        Mzy = (Mzy * data_weights[:, None, None]).mean(0)
        Muy = (Muy * data_weights[:, None, None]).mean(0)

        Mz = (Mz1 + Mz_emis)
        Mu = (Mu + Mu_emis)
        Muz = (Muz2 + Muz_emis)

        # update obs matrix C & input matrix D
        if self.param_props['update']['emissions_weights'] and self.param_props['update']['emissions_input_weights']:
            # do a joint update to C and D
            Mlin = torch.cat((Mzy, Muy), dim=0)  # from linear terms
            Mquad = utils.block([[Mz, Muz.T], [Muz, Mu]], dims=(1, 0))  # from quadratic terms
            CDnew = torch.linalg.solve(Mquad.T, Mlin).T  # new A and B from regression
            self.emissions_weights = CDnew[:, :nz]  # new A
            self.emissions_input_weights = CDnew[:, nz:]  # new B
        elif self.param_props['update']['emissions_weights']:  # update C only
            Cnew = torch.linalg.solve(Mz.T, Mzy - Muz.T @ self.emissions_input_weights.T).T  # new A
            self.emissions_weights = Cnew
        elif self.param_props['update']['emissions_input_weights']:  # update D only
            Dnew = torch.linalg.solve(Mu.T, Muy - Muz @ self.emissions_weights.T).T  # new B
            self.emissions_input_weights = Dnew

        # update obs noise covariance R
        if self.param_props['update']['emissions_cov']:
            My = utils.batch_trans(yy) @ yy  # compute suff stat E[yy@yy']
            My = (My * data_weights[:, None, None]).mean(0)  # compute suff stat E[yy@yy']

            self.emissions_cov = (My + self.emissions_weights @ Mz @ self.emissions_weights.T + self.emissions_input_weights @ Mu @ self.emissions_input_weights.T
                                 - self.emissions_weights @ Mzy - Mzy.T @ self.emissions_weights.T
                                 - self.emissions_input_weights @ Muy - Muy.T @ self.emissions_input_weights.T
                                 + self.emissions_weights @ Muz.T @ self.emissions_input_weights.T + self.emissions_input_weights @ Muz @ self.emissions_weights.T)

        if self.param_props['shape']['emissions_cov'] == 'diag':
            self.emissions_cov = torch.log(torch.diag(self.emissions_cov))

        return ll

    def fit_gd(self, emissions_list, input_list, init_mean=None, init_cov=None, learning_rate=1e-2, num_steps=50):
        """ This function will fit the model using gradient descent on the entire data set
        """
        self.dynamics_weights.requires_grad = True
        self.dynamics_input_weights.requires_grad = True
        self.dynamics_cov.requires_grad = True

        self.emissions_weights.requires_grad = True
        self.emissions_input_weights.requires_grad = True
        self.emissions_cov.requires_grad = True

        emissions, inputs = self.standardize_inputs(emissions_list, input_list)

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

        # initialize the optimizer with the parameters we're going to optimize
        # TODO: could change this so update uses the variable name and we just use getattr to append these in a loop
        model_params = []
        if self.param_props['update']['dynamics_weights']:
            self.dynamics_weights.requires_grad = True
            model_params.append(self.dynamics_weights)

        if self.param_props['update']['dynamics_input_weights']:
            self.dynamics_input_weights.requires_grad = True
            model_params.append(self.dynamics_input_weights)

        if self.param_props['update']['dynamics_cov']:
            self.dynamics_cov.requires_grad = True
            model_params.append(self.dynamics_cov)

        if self.param_props['update']['emissions_weights']:
            self.emissions_weights.requires_grad = True
            model_params.append(self.emissions_weights)

        if self.param_props['update']['emissions_input_weights']:
            self.emissions_input_weights.requires_grad = True
            model_params.append(self.emissions_input_weights)

        if self.param_props['update']['emissions_cov']:
            self.emissions_cov.requires_grad = True
            model_params.append(self.emissions_cov)

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
        self.dynamics_input_weights.requires_grad = False
        self.dynamics_cov.requires_grad = False
        self.emissions_weights.requires_grad = False
        self.emissions_input_weights.requires_grad = False
        self.emissions_cov.requires_grad = False

        return

    @staticmethod
    def estimate_init_mean(emissions):
        init_mean = torch.nanmean(emissions, dim=1)

        init_mean[torch.isnan(init_mean)] = torch.tile(torch.nanmean(init_mean, dim=0), (emissions.shape[0], 1))[torch.isnan(init_mean)]

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

    @staticmethod
    def _get_lagged_data(data, lags):
        num_data_sets, num_time, num_neurons = data.shape

        if type(data) is np.ndarray:
            cat_fun = np.concatenate
            lagged_data = np.zeros((num_data_sets, num_time - lags + 1, 0))
        else:
            cat_fun = torch.cat
            lagged_data = torch.zeros((num_data_sets, num_time - lags + 1, 0))

        for tau in reversed(range(lags)):
            if tau == lags-1:
                lagged_data = cat_fun((lagged_data, data[:, tau:, :]), 2)
            else:
                lagged_data = cat_fun((lagged_data, data[:, tau:-lags + tau + 1, :]), 2)

        return lagged_data

    @staticmethod
    def _get_lagged_weights(weights, fill='eye'):
        if type(weights) is np.ndarray:
            cat_fun = np.concatenate
            split_fun = np.split
            eye_fun = np.eye
            zeros_fun = np.zeros
        else:
            cat_fun = torch.cat
            split_fun = torch.split
            eye_fun = torch.eye
            zeros_fun = torch.zeros

        lagged_weights = cat_fun(split_fun(weights, 1, 0), 2)[0, :, :]

        if fill == 'eye':
            fill_mat = eye_fun(lagged_weights.shape[1] - lagged_weights.shape[0], lagged_weights.shape[1])
        elif fill == 'zeros':
            fill_mat = zeros_fun((lagged_weights.shape[1] - lagged_weights.shape[0], lagged_weights.shape[1]))
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

