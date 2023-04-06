import torch
import numpy as np
import pickle
import time
import utilities as util
import inference as infer


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

    def _set_to_init(self):
        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device, dtype=self.dtype)
        self.dynamics_input_weights = torch.tensor(self.dynamics_input_weights_init, device=self.device, dtype=self.dtype)
        self.dynamics_offset = torch.tensor(self.dynamics_offset_init, device=self.device, dtype=self.dtype)
        self.dynamics_cov = torch.tensor(self.dynamics_cov_init, device=self.device, dtype=self.dtype)

        self.emissions_weights = torch.tensor(self.emissions_weights_init, device=self.device, dtype=self.dtype)
        self.emissions_input_weights = torch.tensor(self.emissions_input_weights_init, device=self.device, dtype=self.dtype)
        self.emissions_offset = torch.tensor(self.emissions_offset_init, device=self.device, dtype=self.dtype)
        self.emissions_cov = torch.tensor(self.emissions_cov_init, device=self.device, dtype=self.dtype)

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

    def fit_em(self, emissions_list, input_list, num_steps=10):
        emissions, input = self.standardize_input(emissions_list, input_list)
        init_mean, init_cov = self.estimate_init(emissions)
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
            ll, init_stats, dynamics_stats, emission_stats = infer.e_step(params, emissions, input, init_mean, init_cov)
            infer.m_step(params, init_stats, dynamics_stats, emission_stats)

            log_likelihood_out.append(ll.detach().cpu().numpy())
            time_out.append(time.time() - start)

            if self.verbose:
                print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
                print('log likelihood = ' + str(log_likelihood_out[-1]))
                print('Time elapsed = ' + str(time_out[-1]))

        self.log_likelihood = log_likelihood_out
        self.train_time = time_out

    def standardize_input(self, emissions_list, input_list):
        assert(type(emissions_list) is list)
        assert(type(input_list) is list)

        if emissions_list[0].dtype is not self.dtype:
            emissions_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions_list]
            input_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in input_list]

        emissions, input = self._stack_data(emissions_list, input_list)

        return emissions, input


    @staticmethod
    def estimate_init(emissions):
        init_mean = torch.nanmean(emissions, dim=1)
        init_cov = [util._estimate_cov(i)[None, :, :] for i in emissions]
        init_cov = torch.cat(init_cov, dim=0)

        return init_mean, init_cov

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

