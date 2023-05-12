import torch
import numpy as np
import pickle
import utilities as utils
from mpi4py import MPI
from scipy.interpolate import interp1d


class Lgssm:
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """

    def __init__(self, dynamics_dim, emissions_dim, input_dim, num_lags=1, param_props=None, dtype=torch.float64,
                 device='cpu', verbose=True, nan_fill=1e8):
        self.num_lags = num_lags
        self.dynamics_dim = dynamics_dim
        self.emissions_dim = emissions_dim
        self.input_dim = input_dim
        self.dynamics_dim_full = self.dynamics_dim * self.num_lags
        self.emissions_dim_full = self.emissions_dim * self.num_lags
        self.input_dim_full = self.input_dim * self.num_lags
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.log_likelihood = None
        self.train_time = None
        self.nan_fill = nan_fill

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
        tau = self.num_lags / 3
        const = (np.exp(3) - 1) * np.exp(1 / tau - 3) / (np.exp(1 / tau) - 1)
        time_decay = np.exp(-np.arange(self.num_lags) / tau) / const
        self.dynamics_weights_init = 0.9 * np.tile(np.eye(self.dynamics_dim), (self.num_lags, 1, 1))
        self.dynamics_weights_init = self.dynamics_weights_init * time_decay[:, None, None]
        self.dynamics_cov_init = np.eye(self.dynamics_dim)
        self.dynamics_input_weights_init = np.zeros((self.num_lags, self.dynamics_dim, self.input_dim))
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

    def randomize_weights(self, max_eig_allowed=0.8, init_std=1.0, rng=np.random.default_rng()):
        # randomize dynamics weights
        tau = self.num_lags / 3
        const = (np.exp(3) - 1) * np.exp(1 / tau - 3) / (np.exp(1 / tau) - 1)
        time_decay = np.exp(-np.arange(self.num_lags) / tau) / const
        self.dynamics_weights_init = rng.standard_normal((self.num_lags, self.dynamics_dim, self.dynamics_dim))
        eig_vals, eig_vects = np.linalg.eig(self.dynamics_weights_init)
        eig_vals = eig_vals / np.max(np.abs(eig_vals)) * max_eig_allowed
        negative_real_eigs = np.real(eig_vals) < 0
        eig_vals[negative_real_eigs] = -eig_vals[negative_real_eigs]
        eig_vals_mat = np.zeros((self.num_lags, self.dynamics_dim, self.dynamics_dim), dtype=np.cdouble)
        for i in range(self.num_lags):
            eig_vals_mat[i, :, :] = np.diag(eig_vals[i, :])
        self.dynamics_weights_init = np.real(eig_vects @ np.transpose(np.linalg.solve(np.transpose(eig_vects, (0, 2, 1)), eig_vals_mat), (0, 2, 1)))
        self.dynamics_weights_init = self.dynamics_weights_init * time_decay[:, None, None]

        if self.param_props['shape']['dynamics_input_weights'] == 'diag':
            dynamics_input_weights_init_diag = init_std * rng.standard_normal((self.num_lags, self.input_dim))
            self.dynamics_input_weights_init = np.zeros((self.num_lags, self.dynamics_dim, self.input_dim))
            for i in range(self.num_lags):
                self.dynamics_input_weights_init[i, :self.input_dim, :] = np.diag(dynamics_input_weights_init_diag[i, :])
        else:
            self.dynamics_input_weights_init = init_std * rng.standard_normal((self.num_lags, self.dynamics_dim, self.input_dim))
        self.dynamics_input_weights_init = self.dynamics_input_weights_init * time_decay[:, None, None]

        self.dynamics_offset_init = np.zeros(self.dynamics_dim)

        if self.param_props['shape']['dynamics_cov'] == 'diag':
            self.dynamics_cov_init = np.diag(np.exp(init_std * rng.standard_normal(self.dynamics_dim)))
        else:
            self.dynamics_cov_init = init_std * rng.standard_normal((self.dynamics_dim, self.dynamics_dim))
            self.dynamics_cov_init = self.dynamics_cov_init.T @ self.dynamics_cov_init / self.dynamics_dim + np.eye(self.dynamics_dim)

        # randomize emissions weights
        self.emissions_weights_init = init_std * rng.standard_normal((self.emissions_dim, self.dynamics_dim))
        self.emissions_input_weights_init = init_std * rng.standard_normal((self.emissions_dim, self.input_dim))
        self.emissions_offset_init = np.zeros(self.emissions_dim)

        if self.param_props['shape']['emissions_cov'] == 'diag':
            self.emissions_cov_init = np.diag(np.exp(init_std * rng.standard_normal(self.dynamics_dim)))
        else:
            self.emissions_cov_init = init_std * rng.standard_normal((self.emissions_dim, self.emissions_dim))
            self.emissions_cov_init = self.emissions_cov_init.T @ self.emissions_cov_init / self.emissions_dim + np.eye(self.emissions_dim)

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

    def sample(self, num_time=100, init_mean=None, init_cov=None, num_data_sets=1, use_sparse_inputs=True,
               scattered_nan_freq=0.0, lost_emission_freq=0.0, rng=np.random.default_rng()):

        latents_list = []
        inputs_list = []
        emissions_list = []
        init_mean_list = []
        init_cov_list = []

        if use_sparse_inputs:
            stim_time_scale = 50
            stims_per_data_set = int(num_time / stim_time_scale)
            num_stims = num_data_sets * stims_per_data_set
            sparse_inputs_init = np.eye(self.input_dim)[rng.choice(self.input_dim, num_stims, replace=True)]

            # upsample to full time
            total_time = num_time * num_data_sets
            sparse_inputs = np.zeros((total_time, self.emissions_dim))
            sparse_inputs[::stim_time_scale, :] = sparse_inputs_init
            sparse_inputs = np.split(sparse_inputs, num_data_sets)

        for d in range(num_data_sets):
            # generate a random initial mean and covariance
            if init_mean is None:
                init_mean = rng.standard_normal(self.dynamics_dim_full)

            if init_cov is None:
                init_cov_block = rng.standard_normal((self.dynamics_dim, self.dynamics_dim))
                init_cov_block = init_cov_block.T @ init_cov_block / self.dynamics_dim
                init_cov = np.eye(self.dynamics_dim_full) / self.nan_fill
                init_cov[:self.dynamics_dim, :self.dynamics_dim] = init_cov_block

            latents = np.zeros((num_time, self.dynamics_dim_full))
            emissions = np.zeros((num_time, self.emissions_dim))

            if use_sparse_inputs:
                inputs = sparse_inputs[d]
                inputs = self._get_lagged_data(inputs, self.num_lags, add_pad=True)
            else:
                inputs = rng.standard_normal((num_time + self.num_lags - 1, self.input_dim))
                inputs = self._get_lagged_data(inputs, self.num_lags, add_pad=False)

            # get the initial observations
            dynamics_noise = rng.multivariate_normal(np.zeros(self.dynamics_dim_full), self.dynamics_cov, size=num_time)
            emissions_noise = rng.multivariate_normal(np.zeros(self.emissions_dim), self.emissions_cov, size=num_time)
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
            for t in range(1, num_time):
                latents[t, :] = (self.dynamics_weights @ latents[t-1, :]) + \
                                 dynamics_inputs[t, :] + \
                                 self.dynamics_offset + \
                                 dynamics_noise[t, :]

                emissions[t, :] = (self.emissions_weights @ latents[t, :]) + \
                                   emissions_inputs[t, :] + \
                                   self.emissions_offset + \
                                   emissions_noise[t, :]

            latents_list.append(latents)
            inputs_list.append(inputs)
            emissions_list.append(emissions)
            init_mean_list.append(init_mean)
            init_cov_list.append(init_cov)

        # add in nans
        scattered_nan_mask = rng.random((num_data_sets, num_time, self.emissions_dim)) <= scattered_nan_freq
        lost_emission_mask = rng.random((num_data_sets, 1, self.emissions_dim)) < lost_emission_freq

        # make sure each data set has at least one measurement and that all emissions were measured at least once
        for d in range(num_data_sets):
            if np.all(lost_emission_mask[d, :, :]):
                lost_emission_mask[d, :, rng.integers(0, self.emissions_dim)] = False

        for n in range(self.emissions_dim):
            if np.all(lost_emission_mask[:, :, n]):
                lost_emission_mask[:, :, n] = False

        nan_mask = scattered_nan_mask | lost_emission_mask

        for d in range(num_data_sets):
            emissions_list[d][nan_mask[d, :, :]] = np.nan

        data_dict = {'latents': latents_list,
                     'inputs': inputs_list,
                     'emissions': emissions_list,
                     'init_mean': init_mean_list,
                     'init_cov': init_cov_list,
                     }

        return data_dict

    def lgssm_filter(self, emissions, inputs, init_mean, init_cov):
        """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
        """
        num_timesteps = emissions.shape[0]

        ll = torch.tensor(0, device=self.device, dtype=self.dtype)
        filtered_mean = init_mean
        filtered_cov = init_cov

        dynamics_inputs = inputs @ self.dynamics_input_weights.T
        emissions_inputs = inputs @ self.emissions_input_weights.T

        filtered_means_list = []
        filtered_covs_list = []

        for t in range(num_timesteps):
            # Shorthand: get parameters and input for time index t
            y = emissions[t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)
            R = torch.where(torch.diag(nan_loc), self.nan_fill, self.emissions_cov)

            # Predict the next state
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

            # Update the log likelihood
            ll_mu = self.emissions_weights @ pred_mean + emissions_inputs[t, :] + self.emissions_offset
            ll_cov = self.emissions_weights @ pred_cov @ self.emissions_weights.T + R

            # ll += torch.distributions.multivariate_normal.MultivariateNormal(ll_mu, ll_cov).log_prob(y)
            mean_diff = y - ll_mu
            ll += -1/2 * (emissions.shape[1] * np.log(2*np.pi) + torch.linalg.slogdet(ll_cov)[1] +
                          torch.dot(mean_diff, torch.linalg.solve(ll_cov, mean_diff)))

            # Condition on this emission
            # Compute the Kalman gain
            K = torch.linalg.solve(ll_cov, self.emissions_weights @ pred_cov).T

            filtered_cov = pred_cov - K @ ll_cov @ K.T
            filtered_mean = pred_mean + K @ mean_diff

            filtered_covs_list.append(filtered_cov)
            filtered_means_list.append(filtered_mean)

        filtered_means = torch.stack(filtered_means_list)
        filtered_covs = torch.stack(filtered_covs_list)

        return ll, filtered_means, filtered_covs

    def lgssm_smoother(self, emissions, inputs, init_mean, init_cov):
        r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states. Technically, this
        implements the Rauch-Tung-Striebel (RTS) smoother.
        Adopted from Dynamax
        """
        num_timesteps = emissions.shape[0]

        # Run the Kalman filter
        ll, filtered_means, filtered_covs = self.lgssm_filter(emissions, inputs, init_mean, init_cov)
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

        return ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses

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
        ll, suff_stats = self.get_suff_stats(emissions, inputs, init_mean, init_cov)

        return ll, suff_stats

    def em_step_pillow(self, emissions_list, inputs_list, init_mean_list, init_cov_list, is_parallel=False):
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

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        nz = self.dynamics_dim_full  # number of latents

        if rank == 0:
            data_out = list(zip(emissions_list, inputs_list, init_mean_list, init_cov_list))
        else:
            data_out = None

        if is_parallel:
            data = comm.scatter(data_out, root=0)

            ll_suff_stats = self.parallel_suff_stats(data)

            ll_suff_stats = comm.gather(ll_suff_stats, root=0)
        else:
            ll_suff_stats = []
            for d in data_out:
                ll_suff_stats.append(self.parallel_suff_stats(d))

        if rank == 0:
            log_likelihood = [i[0] for i in ll_suff_stats]
            log_likelihood = torch.sum(torch.stack(log_likelihood))
            suff_stats = [i[1] for i in ll_suff_stats]

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

            lte_total = torch.sum(torch.stack([i['latent_to_emission'] for i in suff_stats]), dim=0)
            ite_weights = torch.sum(torch.stack([i['input_to_emission'] for i in suff_stats]), dim=0)
            ete_weights = torch.sum(torch.stack([i['emission_to_emission'] for i in suff_stats]), dim=0)

            Mz1 = torch.sum(torch.stack(Mz1_list), dim=0) / (total_time - len(emissions_list))
            Mz2 = torch.sum(torch.stack(Mz2_list), dim=0) / (total_time - len(emissions_list))
            Mz12 = torch.sum(torch.stack(Mz12_list), dim=0) / (total_time - len(emissions_list))
            Mu1 = torch.sum(torch.stack(Mu1_list), dim=0) / (total_time - len(emissions_list))
            Muz2 = torch.sum(torch.stack(Muz2_list), dim=0) / (total_time - len(emissions_list))
            Muz21 = torch.sum(torch.stack(Muz21_list), dim=0) / (total_time - len(emissions_list))

            Mz = torch.sum(torch.stack(Mz_list), dim=0) / total_time
            Mu2 = torch.sum(torch.stack(Mu2_list), dim=0) / total_time
            Muz = torch.sum(torch.stack(Muz_list), dim=0) / total_time

            # Mzy = torch.sum(torch.stack(Mzy_list), dim=0) / lte_total
            # Muy = torch.sum(torch.stack(Muy_list), dim=0) / ite_weights
            # My = torch.sum(torch.stack(My_list), dim=0) / ete_weights
            Mzy = torch.sum(torch.stack(Mzy_list), dim=0) / total_time
            Muy = torch.sum(torch.stack(Muy_list), dim=0) / total_time
            My = torch.sum(torch.stack(My_list), dim=0) / total_time

            # You may not have measured a pair of neurons at the same time across any of the data sets
            # non_zero_non_diag = ete_weights != 0 & ~torch.eye(ete_weights.shape[0], device=self.device, dtype=torch.bool)
            # My[ete_weights == 0] = torch.mean(My[non_zero_non_diag])
            # My[ete_weights == 0] = 0

            # update dynamics matrix A & input matrix B
            # append the trivial parts of the weights from input lags
            dynamics_eye_pad = torch.eye(self.dynamics_dim * (self.num_lags - 1), device=self.device, dtype=self.dtype)
            dynamics_zeros_pad = torch.zeros((self.dynamics_dim * (self.num_lags - 1), self.dynamics_dim), device=self.device, dtype=self.dtype)
            dynamics_pad = torch.cat((dynamics_eye_pad, dynamics_zeros_pad), dim=1)
            dynamics_inputs_zeros_pad = torch.zeros((self.dynamics_dim * (self.num_lags - 1), self.input_dim_full), device=self.device, dtype=self.dtype)

            if self.param_props['update']['dynamics_weights'] and self.param_props['update']['dynamics_input_weights']:
                # do a joint update for A and B
                Mlin = torch.cat((Mz12, Muz2), dim=0)  # from linear terms
                Mquad = utils.block(((Mz1, Muz21.T), (Muz21, Mu1)), dims=(1, 0))  # from quadratic terms

                if self.param_props['shape']['dynamics_input_weights'] == 'diag':
                    if self.param_props['mask']['dynamics_input_weights'] is None:
                        self.param_props['mask']['dynamics_input_weights'] = torch.eye(self.dynamics_dim, self.input_dim, device=self.device, dtype=torch.bool)

                    # make the of which parameters to fit
                    full_block = torch.ones((self.dynamics_dim_full, self.dynamics_dim), device=self.device, dtype=self.dtype)
                    diag_block = torch.tile(self.param_props['mask']['dynamics_input_weights'].T, (self.num_lags, 1))
                    mask = torch.cat((full_block, diag_block), dim=0)

                    ABnew = utils.solve_masked(Mquad.T, Mlin[:, :self.dynamics_dim], mask).T  # new A and B from regression
                else:
                    ABnew = torch.linalg.solve(Mquad.T, Mlin[:, :self.dynamics_dim]).T  # new A and B from regression

                self.dynamics_weights = ABnew[:, :nz]  # new A
                self.dynamics_input_weights = ABnew[:, nz:]

                self.dynamics_weights = torch.cat((self.dynamics_weights, dynamics_pad), dim=0)  # new A
                self.dynamics_input_weights = torch.cat((self.dynamics_input_weights, dynamics_inputs_zeros_pad), dim=0)  # new B

            elif self.param_props['update']['dynamics_weights']:  # update dynamics matrix A only
                self.dynamics_weights = torch.linalg.solve(Mz1.T, (Mz12 - Muz21.T @ self.dynamics_input_weights.T)[:, :self.dynamics_dim]).T  # new A
                self.dynamics_weights = torch.cat((self.dynamics_weights, dynamics_pad), dim=0)  # new A

            elif self.param_props['update']['dynamics_input_weights']:  # update input matrix B only
                if self.param_props['shape']['dynamics_input_weights'] == 'diag':
                    if self.param_props['mask']['dynamics_input_weights'] is None:
                        self.param_props['mask']['dynamics_input_weights'] = torch.eye(self.dynamics_dim, self.input_dim, device=self.device, dtype=torch.bool)

                    # make the of which parameters to fit
                    mask = torch.tile(self.param_props['mask']['dynamics_input_weights'].T, (self.num_lags, 1))

                    self.dynamics_input_weights = utils.solve_masked(Mu1.T, (Muz2 - Muz21 @ self.dynamics_weights.T)[:, :self.dynamics_dim], mask).T  # new A and B from regression
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

                self.dynamics_cov = 0.5 * (self.dynamics_cov + self.dynamics_cov.T)

            # update obs matrix C & input matrix D
            if self.param_props['update']['emissions_weights'] and self.param_props['update']['emissions_input_weights']:
                # do a joint update to C and D
                Mlin = torch.cat((Mzy, Muy), dim=0)  # from linear terms
                Mquad = utils.block([[Mz, Muz.T], [Muz, Mu2]], dims=(1, 0))  # from quadratic terms
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

                self.emissions_cov = 0.5 * (self.emissions_cov + self.emissions_cov.T)

            return log_likelihood

    def get_suff_stats(self, emissions, inputs, init_mean, init_cov):
        nt = emissions.shape[0]

        ll, filtered_means, filtered_covs, smoothed_means, smoothed_covs, smoothed_crosses = \
            self.lgssm_smoother(emissions, inputs, init_mean, init_cov)

        # y = torch.where(torch.isnan(emissions), 0, emissions)
        y = torch.where(torch.isnan(emissions), (self.emissions_weights @ smoothed_means.T).T, emissions)

        # =============== Update dynamics parameters ==============
        # Compute sufficient statistics for latents
        Mz1 = smoothed_covs[:-1, :, :].sum(0) + smoothed_means[:-1, :].T @ smoothed_means[:-1, :]  # E[zz@zz'] for 1 to T-1
        Mz2 = smoothed_covs[1:, :, :].sum(0) + smoothed_means[1:, :].T @ smoothed_means[1:, :]  # E[zz@zz'] for 2 to T
        Mz12 = smoothed_crosses.sum(0) + smoothed_means[:-1, :].T @ smoothed_means[1:, :]  # E[zz_t@zz_{t+1}'] (above-diag)

        # Compute sufficient statistics for inputs x latents
        Mu1 = inputs[1:, :].T @ inputs[1:, :]  # E[uu@uu'] for 2 to T
        Muz2 = inputs[1:, :].T @ smoothed_means[1:, :]  # E[uu@zz'] for 2 to T
        Muz21 = inputs[1:, :].T @ smoothed_means[:-1, :]  # E[uu_t@zz_{t-1} for 2 to T

        # =============== Update observation parameters ==============
        # Compute sufficient statistics
        Mz_emis = smoothed_covs[-1, :, :] + smoothed_means[-1, :, None] * smoothed_means[-1, None, :]  # re-use Mz1 if possible
        Mu_emis = inputs[0, :, None] * inputs[0, None, :]  # reuse Mu
        Muz_emis = inputs[0, :, None] * smoothed_means[0, None, :]  # reuse Muz
        Mzy = smoothed_means.T @ y  # E[zz@yy']
        Muy = inputs.T @ y  # E[uu@yy']
        My = y.T @ y  # compute suff stat E[yy@yy']

        Mz = Mz1 + Mz_emis
        Mu2 = Mu1 + Mu_emis
        Muz = Muz2 + Muz_emis

        # # get number of nan time points
        latent_to_emission = torch.zeros((self.dynamics_dim_full, self.dynamics_dim), device=self.device, dtype=self.dtype)
        input_to_emission = torch.zeros((self.input_dim_full, self.dynamics_dim), device=self.device, dtype=self.dtype)
        emission_to_emission = torch.zeros((self.dynamics_dim, self.dynamics_dim), device=self.device, dtype=self.dtype)
        for j in range(self.dynamics_dim):
            for i in range(self.dynamics_dim_full):
                latent_to_emission[i, j] = torch.sum(~(torch.isnan(smoothed_means[:, i]) | torch.isnan(emissions[:, j])))

            for i in range(self.input_dim_full):
                input_to_emission[i, j] = torch.sum(~(torch.isnan(inputs[:, i]) | torch.isnan(emissions[:, j])))

            for i in range(self.dynamics_dim):
                emission_to_emission[i, j] = torch.sum(~(torch.isnan(emissions[:, i]) | torch.isnan(emissions[:, j])))

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
                      'latent_to_emission': latent_to_emission,
                      'input_to_emission': input_to_emission,
                      'emission_to_emission': emission_to_emission,
                      }

        return ll, suff_stats

    def _pad_init_for_lags(self):
        self.dynamics_weights_init = self._get_lagged_weights(self.dynamics_weights_init, fill='eye')
        self.dynamics_input_weights_init = self._get_lagged_weights(self.dynamics_input_weights_init, fill='zeros')
        self.dynamics_offset_init = self._pad_zeros(self.dynamics_offset_init, self.num_lags, axis=0)
        dci_block = self.dynamics_cov_init
        self.dynamics_cov_init = np.eye(self.dynamics_dim_full) / self.nan_fill
        self.dynamics_cov_init[:self.dynamics_dim, :self.dynamics_dim] = dci_block

        self.emissions_weights_init = self._pad_zeros(self.emissions_weights_init, self.num_lags, axis=1)
        self.emissions_input_weights_init = self._pad_zeros(self.emissions_input_weights_init, self.num_lags, axis=1)

    def estimate_init_mean(self, emissions):
        # estimate the initial mean of a data set as the mean over time
        init_mean_list = []

        for i in emissions:
            init_mean = torch.nanmean(i, dim=0)
            init_mean[torch.isnan(init_mean)] = torch.nanmean(init_mean)
            # repeat the mean for each delay you have
            init_mean = torch.tile(init_mean, [self.num_lags])
            init_mean_list.append(init_mean)

        return init_mean_list

    def estimate_init_cov(self, emissions):
        # estimate the initial covariance of a data set as the data covariance over all time
        init_cov_list = []

        for i in emissions:
            init_cov_block = utils.estimate_cov(i)

            # structure the init cov for lags
            init_cov = torch.eye(self.dynamics_dim_full, device=self.device, dtype=self.dtype) / self.nan_fill
            init_cov[:self.dynamics_dim, :self.dynamics_dim] = init_cov_block

            init_cov_list.append(init_cov)

        return init_cov_list

    @staticmethod
    def _get_lagged_data(data, lags, add_pad=True):
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
    def _get_lagged_weights(weights, fill='eye'):
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
            fill_mat = eye_fun(lagged_weights.shape[0] * (num_lags - 1), lagged_weights.shape[1])
        elif fill == 'zeros':
            fill_mat = zeros_fun((lagged_weights.shape[0] * (num_lags - 1), lagged_weights.shape[1]))
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

