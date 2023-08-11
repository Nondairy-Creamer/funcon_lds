import numpy as np
import pickle
import inference_utilities as iu
import warnings
from scipy.linalg import block_diag


class Lgssm:
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """

    def __init__(self, dynamics_dim, emissions_dim, input_dim, dynamics_lags=1, dynamics_input_lags=1, cell_ids=None,
                 param_props=None, verbose=True, epsilon=1e8, ridge_lambda=0):
        self.dynamics_lags = dynamics_lags
        self.dynamics_input_lags = dynamics_input_lags
        self.dynamics_dim = dynamics_dim
        self.emissions_dim = emissions_dim
        self.input_dim = input_dim
        self.dynamics_dim_full = self.dynamics_dim * self.dynamics_lags
        self.input_dim_full = self.input_dim * self.dynamics_input_lags
        self.verbose = verbose
        self.log_likelihood = None
        self.train_time = None
        self.epsilon = epsilon
        self.sample_rate = 0.5  # default is 2 Hz
        self.ridge_lambda = ridge_lambda
        if cell_ids is None:
            self.cell_ids = [str(i) for i in range(self.dynamics_dim)]
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

    def randomize_weights(self, max_eig_allowed=0.99, rng=np.random.default_rng()):
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
        self.dynamics_weights = self.dynamics_weights_init
        self.dynamics_input_weights = self.dynamics_input_weights_init
        self.dynamics_offset = self.dynamics_offset_init
        self.dynamics_cov = self.dynamics_cov_init

        self.emissions_weights = self.emissions_weights_init
        self.emissions_input_weights = self.emissions_input_weights_init
        self.emissions_offset = self.emissions_offset_init
        self.emissions_cov = self.emissions_cov_init

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

                      'trained': {'dynamics_weights': self.dynamics_weights,
                                  'dynamics_input_weights': self.dynamics_input_weights,
                                  'dynamics_offset': self.dynamics_offset,
                                  'dynamics_cov': self.dynamics_cov,
                                  'emissions_weights': self.emissions_weights,
                                  'emissions_input_weights': self.emissions_input_weights,
                                  'emissions_offset': self.emissions_offset,
                                  'emissions_cov': self.emissions_cov,
                                  },
                      }

        return params_out

    def sample(self, num_time=100, init_mean=None, init_cov=None, num_data_sets=1, input_time_scale=0, inputs_list=None,
               scattered_nan_freq=0.0, lost_emission_freq=0.0, rng=np.random.default_rng(), add_noise=True):
        latents_list = []
        emissions_list = []
        init_mean_list = []
        init_cov_list = []

        if init_mean is None:
            generate_mean = True
        else:
            generate_mean = False

        if init_cov is None:
            generate_cov = True
        else:
            generate_cov = False

        if inputs_list is None:
            generate_inputs = True
        else:
            generate_inputs = False

        if generate_inputs:
            if input_time_scale != 0:
                stims_per_data_set = int(num_time / input_time_scale)
                num_stims = num_data_sets * stims_per_data_set
                sparse_inputs_init = np.eye(self.input_dim)[rng.choice(self.input_dim, num_stims, replace=True)]

                # upsample to full time
                total_time = num_time * num_data_sets
                inputs_list = np.zeros((total_time, self.emissions_dim))
                inputs_list[::input_time_scale, :] = sparse_inputs_init
                inputs_list = np.split(inputs_list, num_data_sets)
            else:
                inputs_list = [rng.standard_normal((num_time, self.input_dim)) for i in range(num_data_sets)]

        inputs_lagged = [self.get_lagged_data(i, self.dynamics_input_lags, add_pad=True) for i in inputs_list]

        for d in range(num_data_sets):
            # generate a random initial mean and covariance
            if generate_mean:
                init_mean = rng.standard_normal(self.dynamics_dim_full)

            if generate_cov:
                init_cov = rng.standard_normal((self.dynamics_dim_full, self.dynamics_dim_full))
                init_cov = init_cov.T @ init_cov

            latents = np.zeros((num_time, self.dynamics_dim_full))
            emissions = np.zeros((num_time, self.emissions_dim))
            inputs = inputs_lagged[d]

            # get the initial observations
            dynamics_noise = add_noise * rng.multivariate_normal(np.zeros(self.dynamics_dim_full), self.dynamics_cov, size=num_time)
            emissions_noise = add_noise * rng.multivariate_normal(np.zeros(self.emissions_dim), self.emissions_cov, size=num_time)
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

            latents_list.append(latents[:, :self.dynamics_dim])
            emissions_list.append(emissions)
            init_mean_list.append(init_mean)
            init_cov_list.append(init_cov)

        # add in nans
        scattered_nans_mask = [rng.random((num_time, self.emissions_dim)) < scattered_nan_freq for i in range(num_data_sets)]
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

    def lgssm_filter(self, emissions, inputs, init_mean, init_cov, memmap_cpu_id=None):
        """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This function can deal with missing data if the data is missing the entire time trace
        """
        num_timesteps = emissions.shape[0]

        if inputs.shape[1] < self.input_dim_full:
            inputs = self.get_lagged_data(inputs, self.dynamics_input_lags)

        ll = 0
        filtered_mean = init_mean.copy()
        filtered_cov = init_cov.copy()

        dynamics_inputs = inputs @ self.dynamics_input_weights.T
        emissions_inputs = inputs @ self.emissions_input_weights.T

        filtered_means = np.zeros((num_timesteps, self.dynamics_dim_full))
        if memmap_cpu_id is None:
            filtered_covs = np.zeros((num_timesteps, self.dynamics_dim_full, self.dynamics_dim_full))
        else:
            file_path = '/tmp/filtered_covs_' + str(memmap_cpu_id) + '.tmp'
            filtered_covs = np.memmap(file_path, dtype='float64', mode='w+',
                                      shape=((num_timesteps, self.dynamics_dim_full, self.dynamics_dim_full)))

        # step through the loop and keep calculating the covariances until they converge
        for t in range(num_timesteps):
            # Shorthand: get parameters and input for time index t
            y = emissions[t, :]

            # locate nans and set covariance at their location to a large number to marginalize over them
            nan_loc = np.isnan(y)
            y = np.where(nan_loc, 0, y)
            R = np.where(np.diag(nan_loc), self.epsilon, self.emissions_cov)

            # Predict the next state
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

            # Update the log likelihood
            ll_mu = self.emissions_weights @ pred_mean + emissions_inputs[t, :] + self.emissions_offset

            ll_cov = self.emissions_weights @ pred_cov @ self.emissions_weights.T + R
            ll_cov_logdet = np.linalg.slogdet(ll_cov)[1]
            ll_cov_inv = np.linalg.inv(ll_cov)

            mean_diff = y - ll_mu
            ll += -1/2 * (emissions.shape[1] * np.log(2*np.pi) + ll_cov_logdet +
                          np.dot(mean_diff, ll_cov_inv @ mean_diff))

            # Condition on this emission
            # Compute the Kalman gain
            K = pred_cov.T @ self.emissions_weights.T @ ll_cov_inv
            filtered_cov = pred_cov - K @ ll_cov @ K.T

            filtered_mean = pred_mean + K @ mean_diff
            filtered_means[t, :] = filtered_mean
            filtered_covs[t, :, :] = filtered_cov

        return ll, filtered_means, filtered_covs

    def lgssm_smoother(self, emissions, inputs, init_mean=None, init_cov=None, memmap_cpu_id=None):
        r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states. Technically, this
        implements the Rauch-Tung-Striebel (RTS) smoother.
        Adopted from Dynamax
        """
        num_timesteps = emissions.shape[0]

        if inputs.shape[1] < self.input_dim_full:
            inputs = self.get_lagged_data(inputs, self.dynamics_input_lags)

        if init_mean is None:
            init_mean = self.estimate_init_mean([emissions])[0]

        if init_cov is None:
            init_mean = self.estimate_init_cov([emissions])[0]

        # first run the kalman forward pass
        ll, filtered_means, filtered_covs = self.lgssm_filter(emissions, inputs, init_mean, init_cov, memmap_cpu_id=memmap_cpu_id)

        dynamics_inputs = inputs @ self.dynamics_input_weights.T

        smoothed_means = filtered_means.copy()
        last_cov = filtered_covs[-1, :, :]
        smoothed_cov_next = last_cov.copy()
        smoothed_covs_sum = np.zeros((self.dynamics_dim_full, self.dynamics_dim_full))
        smoothed_crosses_sum = last_cov.copy()
        my_correction = np.zeros((self.emissions_dim, self.emissions_dim))
        mzy_correction = np.zeros((self.dynamics_dim_full, self.emissions_dim))

        # Run the smoother backward in time
        for t in reversed(range(num_timesteps - 1)):
            # Unpack the input
            filtered_mean = filtered_means[t, :]
            filtered_cov = filtered_covs[t, :, :]
            smoothed_mean_next = smoothed_means[t + 1, :]

            # Compute the smoothed mean and covariance
            pred_mean = self.dynamics_weights @ filtered_mean + dynamics_inputs[t+1, :] + self.dynamics_offset
            pred_cov = self.dynamics_weights @ filtered_cov @ self.dynamics_weights.T + self.dynamics_cov

            # This is like the Kalman gain but in reverse
            # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
            G = np.linalg.solve(pred_cov, self.dynamics_weights @ filtered_cov).T
            smoothed_cov_this = filtered_cov + G @ (smoothed_cov_next - pred_cov) @ G.T
            smoothed_means[t, :] = filtered_mean + G @ (smoothed_mean_next - pred_mean)

            # Compute the smoothed expectation of x_t x_{t+1}^T
            # TODO: ask why the second expression is not in jonathan's code
            smoothed_crosses_sum += G @ smoothed_cov_next #+ smoothed_means[:, t, :, None] * smoothed_mean_next[:, None, :]

            # now calculate the correction for my and mzy
            y_nan_loc_t = np.isnan(emissions[t, :])
            c_nan = self.emissions_weights[y_nan_loc_t, :]
            r_nan = self.emissions_cov[np.ix_(y_nan_loc_t, y_nan_loc_t)]

            if t > 0:
                smoothed_covs_sum += smoothed_cov_this

            # add in the variance from all the values of y you imputed
            my_correction[np.ix_(y_nan_loc_t, y_nan_loc_t)] += c_nan @ smoothed_cov_this @ c_nan.T + r_nan
            mzy_correction[:, y_nan_loc_t] += smoothed_cov_this @ c_nan.T

            smoothed_cov_next = smoothed_cov_this.copy()

        suff_stats = {}
        suff_stats['smoothed_covs_sum'] = smoothed_covs_sum
        suff_stats['smoothed_crosses_sum'] = smoothed_crosses_sum
        suff_stats['first_cov'] = smoothed_cov_this
        suff_stats['last_cov'] = last_cov
        suff_stats['my_correction'] = my_correction
        suff_stats['mzy_correction'] = mzy_correction

        return ll, smoothed_means, suff_stats

    def get_ll(self, emissions, inputs, init_mean, init_cov):
        # get the log-likelihood of the data
        ll = 0

        for d in range(len(emissions)):
            ll += self.lgssm_filter(emissions[d], inputs[d], init_mean[d], init_cov[d])[0]

        return ll

    def parallel_suff_stats(self, data, memmap_cpu_id=None):
        emissions = data[0]
        inputs = data[1]
        init_mean = data[2]
        init_cov = data[3]

        ll, suff_stats, smoothed_means, new_init_covs = self.get_suff_stats(emissions, inputs, init_mean, init_cov,
                                                                            memmap_cpu_id=memmap_cpu_id)

        return ll, suff_stats, smoothed_means, new_init_covs

    def em_step(self, emissions_list, inputs_list, init_mean_list, init_cov_list, cpu_id=0, num_cpus=1,
                memmap_cpu_id=None, max_eig=0.999):
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
            data_out = self.package_data_mpi(emissions_list, inputs_list, init_mean_list, init_cov_list, num_cpus)
        else:
            data_out = None

        data = iu.individual_scatter(data_out, root=0)

        ll_suff_stats_smoothed_means = []
        for d in data:
            ll_suff_stats_smoothed_means.append(self.parallel_suff_stats(d, memmap_cpu_id=memmap_cpu_id))

        ll_suff_stats_smoothed_means = iu.individual_gather(ll_suff_stats_smoothed_means, root=0)

        if cpu_id == 0:
            ll_suff_stats_out = []
            for i in ll_suff_stats_smoothed_means:
                for j in i:
                    ll_suff_stats_out.append(j)

            ll_suff_stats_smoothed_means = ll_suff_stats_out

            log_likelihood = [i[0] for i in ll_suff_stats_smoothed_means]
            log_likelihood = np.sum(np.stack(log_likelihood))
            suff_stats = [i[1] for i in ll_suff_stats_smoothed_means]
            smoothed_means = [i[2] for i in ll_suff_stats_smoothed_means]
            new_init_covs = [i[3] for i in ll_suff_stats_smoothed_means]

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

            Mz1 = np.sum(np.stack(Mz1_list), axis=0) / (total_time - len(emissions_list))
            Mz2 = np.sum(np.stack(Mz2_list), axis=0) / (total_time - len(emissions_list))
            Mz12 = np.sum(np.stack(Mz12_list), axis=0) / (total_time - len(emissions_list))
            Mu1 = np.sum(np.stack(Mu1_list), axis=0) / (total_time - len(emissions_list))
            Muz2 = np.sum(np.stack(Muz2_list), axis=0) / (total_time - len(emissions_list))
            Muz21 = np.sum(np.stack(Muz21_list), axis=0) / (total_time - len(emissions_list))

            Mz = np.sum(np.stack(Mz_list), axis=0) / total_time
            Mu2 = np.sum(np.stack(Mu2_list), axis=0) / total_time
            Muz = np.sum(np.stack(Muz_list), axis=0) / total_time

            Mzy = np.sum(np.stack(Mzy_list), axis=0) / total_time
            Muy = np.sum(np.stack(Muy_list), axis=0) / total_time
            My = np.sum(np.stack(My_list), axis=0) / total_time

            # update dynamics matrix A & input matrix B
            # append the trivial parts of the weights from input lags
            dynamics_eye_pad = np.eye(self.dynamics_dim * (self.dynamics_lags - 1))
            dynamics_zeros_pad = np.zeros((self.dynamics_dim * (self.dynamics_lags - 1), self.dynamics_dim))
            dynamics_pad = np.concatenate((dynamics_eye_pad, dynamics_zeros_pad), axis=1)
            dynamics_inputs_zeros_pad = np.zeros((self.dynamics_dim * (self.dynamics_lags - 1), self.input_dim_full))

            ridge_penalty = self.ridge_lambda * np.diag(self.dynamics_cov)

            if self.param_props['update']['dynamics_weights'] and self.param_props['update']['dynamics_input_weights']:
                # do a joint update for A and B
                Mlin = np.concatenate((Mz12, Muz2), axis=0)  # from linear terms
                Mquad = iu.block(((Mz1, Muz21.T), (Muz21, Mu1)), dims=(1, 0))  # from quadratic terms

                if self.param_props['shape']['dynamics_input_weights'] == 'diag':
                    if self.param_props['mask']['dynamics_input_weights'] is None:
                        self.param_props['mask']['dynamics_input_weights'] = np.eye(self.dynamics_dim, self.input_dim)

                    # make the of which parameters to fit
                    full_block = np.ones((self.dynamics_dim_full, self.dynamics_dim))
                    diag_block = np.tile(self.param_props['mask']['dynamics_input_weights'].T, (self.dynamics_input_lags, 1))
                    mask = np.concatenate((full_block, diag_block), axis=0)

                    ABnew = iu.solve_masked(Mquad.T, Mlin[:, :self.dynamics_dim], mask, ridge_penalty=ridge_penalty).T  # new A and B from regression
                else:
                    if self.ridge_lambda == 0:
                        ABnew = np.linalg.solve(Mquad.T, Mlin[:, :self.dynamics_dim]).T
                    else:
                        ABnew = iu.solve_masked(Mquad.T, Mlin[:, :self.dynamics_dim], ridge_penalty=ridge_penalty).T

                self.dynamics_weights = ABnew[:, :nz]  # new A
                self.dynamics_input_weights = ABnew[:, nz:]

                self.dynamics_weights = np.concatenate((self.dynamics_weights, dynamics_pad), axis=0)  # new A
                self.dynamics_input_weights = np.concatenate((self.dynamics_input_weights, dynamics_inputs_zeros_pad), axis=0)  # new B

                # check the largest eigenvalue of the dynamics matrix
                dyn_eig_vals, dyn_eig_vects = np.linalg.eig(self.dynamics_weights)
                max_abs_eig = np.max(np.abs(dyn_eig_vals))
                if max_abs_eig > max_eig:
                    warnings.warn('Largest eigenvalue of the dynamics matrix is:' + str(max_abs_eig) + ', setting to ' + str(max_eig))

                while max_abs_eig > max_eig:
                    dyn_eig_vals = dyn_eig_vals / max_abs_eig * max_eig
                    self.dynamics_weights = np.real(dyn_eig_vects @ np.linalg.solve(dyn_eig_vects.T, np.diag(dyn_eig_vals)).T)
                    self.dynamics_weights[self.dynamics_dim:, :self.dynamics_dim_full-self.dynamics_dim] = \
                        np.eye(self.dynamics_dim_full-self.dynamics_dim)
                    self.dynamics_weights[self.dynamics_dim:, self.dynamics_dim_full-self.dynamics_dim:] = \
                        np.zeros((self.dynamics_dim_full-self.dynamics_dim, self.dynamics_dim))
                    dyn_eig_vals, dyn_eig_vects = np.linalg.eig(self.dynamics_weights)
                    max_abs_eig = np.max(np.abs(dyn_eig_vals))

            elif self.param_props['update']['dynamics_weights']:  # update dynamics matrix A only
                if self.ridge_lambda == 0:
                    self.dynamics_weights = np.linalg.solve(Mz1.T, (Mz12 - Muz21.T @ self.dynamics_input_weights.T)[:, :self.dynamics_dim]).T  # new A
                else:
                    self.dynamics_weights = iu.solve_masked(Mz1.T, (Mz12 - Muz21.T @ self.dynamics_input_weights.T)[:, :self.dynamics_dim], ridge_penalty=ridge_penalty).T  # new A

                self.dynamics_weights = np.concatenate((self.dynamics_weights, dynamics_pad), axis=0)  # new A

            elif self.param_props['update']['dynamics_input_weights']:  # update input matrix B only
                if self.param_props['shape']['dynamics_input_weights'] == 'diag':
                    if self.param_props['mask']['dynamics_input_weights'] is None:
                        self.param_props['mask']['dynamics_input_weights'] = np.eye(self.dynamics_dim, self.input_dim)

                    # make the of which parameters to fit
                    mask = np.tile(self.param_props['mask']['dynamics_input_weights'].T, (self.dynamics_input_lags, 1))

                    self.dynamics_input_weights = iu.solve_masked(Mu1.T, (Muz2 - Muz21 @ self.dynamics_weights.T)[:, :self.dynamics_dim], mask).T  # new A and B from regression
                else:
                    self.dynamics_input_weights = np.linalg.solve(Mu1.T, (Muz2 - Muz21 @ self.dynamics_weights.T)[:, :self.dynamics_dim]).T  # new B

                self.dynamics_input_weights = np.concatenate((self.dynamics_input_weights, dynamics_inputs_zeros_pad), axis=0)  # new B

            # Update noise covariance Q
            if self.param_props['update']['dynamics_cov']:
                self.dynamics_cov = (Mz2 + self.dynamics_weights @ Mz1 @ self.dynamics_weights.T + self.dynamics_input_weights @ Mu1 @ self.dynamics_input_weights.T
                                     - self.dynamics_weights @ Mz12 - Mz12.T @ self.dynamics_weights.T
                                     - self.dynamics_input_weights @ Muz2 - Muz2.T @ self.dynamics_input_weights.T
                                     + self.dynamics_weights @ Muz21.T @ self.dynamics_input_weights.T + self.dynamics_input_weights @ Muz21 @ self.dynamics_weights.T) #/ (nt - 1)

                if self.param_props['shape']['dynamics_cov'] == 'diag':
                    self.dynamics_cov = np.diag(np.diag(self.dynamics_cov))

                self.dynamics_cov = 0.5 * self.dynamics_cov + 0.5 * self.dynamics_cov.T

            # update obs matrix C & input matrix D
            if self.param_props['update']['emissions_weights'] and self.param_props['update']['emissions_input_weights']:
                # do a joint update to C and D
                Mlin = np.concatenate((Mzy, Muy), axis=0)  # from linear terms
                Mquad = iu.block([[Mz, Muz.T], [Muz, Mu2]], dims=(1, 0))  # from quadratic terms
                CDnew = np.linalg.solve(Mquad.T, Mlin).T  # new A and B from regression
                self.emissions_weights = CDnew[:, :nz]  # new A
                self.emissions_input_weights = CDnew[:, nz:]  # new B
            elif self.param_props['update']['emissions_weights']:  # update C only
                Cnew = np.linalg.solve(Mz.T, Mzy - Muz.T @ self.emissions_input_weights.T).T  # new A
                self.emissions_weights = Cnew
            elif self.param_props['update']['emissions_input_weights']:  # update D only
                Dnew = np.linalg.solve(Mu2.T, Muy - Muz @ self.emissions_weights.T).T  # new B
                self.emissions_input_weights = Dnew

            # update obs noise covariance R
            if self.param_props['update']['emissions_cov']:
                self.emissions_cov = (My + self.emissions_weights @ Mz @ self.emissions_weights.T + self.emissions_input_weights @ Mu2 @ self.emissions_input_weights.T
                                      - self.emissions_weights @ Mzy - Mzy.T @ self.emissions_weights.T
                                      - self.emissions_input_weights @ Muy - Muy.T @ self.emissions_input_weights.T
                                      + self.emissions_weights @ Muz.T @ self.emissions_input_weights.T + self.emissions_input_weights @ Muz @ self.emissions_weights.T)

                if self.param_props['shape']['emissions_cov'] == 'diag':
                    self.emissions_cov = np.diag(np.diag(self.emissions_cov))

                self.emissions_cov = 0.5 * self.emissions_cov + 0.5 * self.emissions_cov.T

            if not np.all(self.dynamics_cov == self.dynamics_cov.T):
                warnings.warn('dynamics_cov is not symmetric')

            if not np.all(self.emissions_cov == self.emissions_cov.T):
                warnings.warn('emissions_cov is not symmetric')

            return log_likelihood, smoothed_means, new_init_covs

        return None, None, None

    def get_suff_stats(self, emissions, inputs, init_mean, init_cov, memmap_cpu_id=None):
        nt = emissions.shape[0]

        ll, smoothed_means, suff_stats = \
            self.lgssm_smoother(emissions, inputs, init_mean, init_cov, memmap_cpu_id=memmap_cpu_id)

        smoothed_covs_sum = suff_stats['smoothed_covs_sum']
        smoothed_crosses_sum = suff_stats['smoothed_crosses_sum']
        first_cov = suff_stats['first_cov']
        last_cov = suff_stats['last_cov']
        my_correction = suff_stats['my_correction']
        mzy_correction = suff_stats['mzy_correction']

        y_nan_loc = np.isnan(emissions)
        y = np.where(y_nan_loc, (self.emissions_weights @ smoothed_means.T).T, emissions)

        # =============== Update dynamics parameters ==============
        # Compute sufficient statistics for latents
        Mz1 = smoothed_covs_sum + first_cov + smoothed_means[:-1, :].T @ smoothed_means[:-1, :]  # E[zz@zz'] for 1 to T-1
        Mz2 = smoothed_covs_sum + last_cov + smoothed_means[1:, :].T @ smoothed_means[1:, :]  # E[zz@zz'] for 2 to T
        Mz12 = smoothed_crosses_sum + smoothed_means[:-1, :].T @ smoothed_means[1:, :]  # E[zz_t@zz_{t+1}'] (above-diag)

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

        My = y.T @ y + my_correction
        Mzy = smoothed_means.T @ y + mzy_correction

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

        return ll, suff_stats, smoothed_means, first_cov

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
            # initialize init_mean to frist nonmean value of the trace
            init_mean = i[0, :]

            # for values that are not nan, find the first non_nan value,
            for nan_loc in np.where(np.isnan(init_mean)):
                first_non_nan_loc = np.where(~np.isnan(i[:, nan_loc]))[0]

                if len(first_non_nan_loc) > 0:
                    init_mean[nan_loc] = i[first_non_nan_loc[0], nan_loc]
                else:
                    init_mean[nan_loc] = np.nan

            # for traces that are all nan, set to the man of all init
            init_mean[np.isnan(init_mean)] = np.nanmean(init_mean)
            init_mean = np.tile(init_mean, [self.dynamics_lags])
            init_mean_list.append(init_mean)

        return init_mean_list

    def estimate_init_cov(self, emissions):
        init_cov_list = []

        for i in emissions:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                emissions_var = np.nanvar(i, axis=0)

            emissions_var[np.isnan(emissions_var)] = np.nanmean(emissions_var[~np.isnan(emissions_var)])
            var_mat = np.diag(emissions_var)
            var_block = block_diag(*([var_mat] * self.dynamics_lags))
            init_cov_list.append(var_block)

        return init_cov_list

    @staticmethod
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
            if tau == lags-1:
                lagged_data = np.concatenate((lagged_data, data[tau:, :]), axis=1)
            else:
                lagged_data = np.concatenate((lagged_data, data[tau:-lags + tau + 1, :]), axis=1)

        return lagged_data

    @staticmethod
    def _get_lagged_weights(weights, lags_out, fill='eye'):
        lagged_weights = np.concatenate(np.split(weights, weights.shape[0], 0), 2)[0, :, :]

        if fill == 'eye':
            fill_mat = np.eye(lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1])
        elif fill == 'zeros':
            fill_mat = np.zeros((lagged_weights.shape[0] * (lags_out - 1), lagged_weights.shape[1]))
        else:
            raise Exception('fill value not recognized')

        lagged_weights = np.concatenate((lagged_weights, fill_mat), 0)

        return lagged_weights

    @staticmethod
    def _pad_zeros(weights, tau, axis=1):
        zeros_shape = list(weights.shape)
        zeros_shape[axis] = zeros_shape[axis] * (tau - 1)

        zero_pad = np.zeros(zeros_shape)

        return np.concatenate((weights, zero_pad), axis)

    @staticmethod
    def _has_no_scattered_nans(emissions):
        any_nan_neurons = np.any(np.isnan(emissions), axis=0)
        all_nan_neurons = np.all(np.isnan(emissions), axis=0)
        return np.all(any_nan_neurons == all_nan_neurons)

    @staticmethod
    def package_data_mpi(emissions_list, inputs_list, init_mean_list, init_cov_list, num_cpus):
        # packages data for sending using MPI
        data_zipped = list(zip(emissions_list, inputs_list, init_mean_list, init_cov_list))
        num_data = len(emissions_list)
        overflow = np.mod(num_data, num_cpus)
        num_data_truncated = num_data - overflow
        # this kind of round about way of distributing the data is to make sure they stay in order
        # when you stack them back up
        chunk_size = [int(num_data_truncated / num_cpus)] * num_cpus

        for i in range(overflow):
            chunk_size[i] += 1

        split_data = []
        pos = 0

        for i in range(len(chunk_size)):
            split_data.append(data_zipped[pos:pos+chunk_size[i]])
            pos += chunk_size[i]

        return split_data


