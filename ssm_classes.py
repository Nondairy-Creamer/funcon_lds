import torch
import numpy as np
import pickle
import time


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
        self.loss = None
        self.train_time = None

        self.dynamics_weights_init = 0.9 * np.eye(self.latent_dim)
        self.inputs_weights_log_init = np.zeros(self.latent_dim) - 1
        self.dynamics_cov_log_init = np.zeros(self.latent_dim) - 1
        self.emissions_cov_log_init = np.zeros(self.latent_dim) - 1

        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device,
                                             dtype=self.dtype, requires_grad=True)
        self.inputs_weights_log = torch.tensor(self.inputs_weights_log_init, device=self.device,
                                               dtype=self.dtype, requires_grad=True)
        self.dynamics_cov_log = torch.tensor(self.dynamics_cov_log_init, device=self.device,
                                             dtype=self.dtype, requires_grad=True)
        self.emissions_cov_log = torch.tensor(self.emissions_cov_log_init, device=self.device,
                                              dtype=self.dtype, requires_grad=True)

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

        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device,
                                             dtype=self.dtype, requires_grad=True)
        self.inputs_weights_log = torch.tensor(self.inputs_weights_log_init, device=self.device,
                                               dtype=self.dtype, requires_grad=True)
        self.dynamics_cov_log = torch.tensor(self.dynamics_cov_log_init, device=self.device,
                                             dtype=self.dtype, requires_grad=True)
        self.emissions_cov_log = torch.tensor(self.emissions_cov_log_init, device=self.device,
                                              dtype=self.dtype, requires_grad=True)

    def set_device(self, new_device):
        self.device = new_device
        self.dynamics_weights = self.dynamics_weights.to(new_device)
        self.inputs_weights_log = self.inputs_weights_log.to(new_device)
        self.dynamics_cov_log = self.dynamics_cov_log.to(new_device)
        self.emissions_cov_log = self.emissions_cov_log.to(new_device)

    def fit_gd(self, emissions_list, inputs_list, learning_rate=1e-2, num_steps=50):
        """ This function will fit the model using gradient descent on the entire data set
        """
        assert(type(emissions_list) is list)
        assert(type(inputs_list) is list)

        if emissions_list[0].dtype is not self.dtype:
            emissions_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions_list]
            inputs_list = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in inputs_list]

        emissions, inputs = self.stack_data(emissions_list, inputs_list)

        # initialize the optimizer with the parameters we're going to optimize
        model_params = [self.dynamics_weights, self.inputs_weights_log, self.dynamics_cov_log, self.emissions_cov_log]

        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        loss_out = []
        time_out = []

        init_mean = torch.nanmean(emissions, dim=1)
        init_cov = [self._estimate_cov(i)[None, :, :] for i in emissions]
        init_cov = torch.cat(init_cov, dim=0)

        start = time.time()
        for ep in range(num_steps):
            optimizer.zero_grad()
            loss = self.loss_fn(emissions, inputs, init_mean, init_cov)
            loss.backward()
            loss_out.append(loss.detach().cpu().numpy())
            optimizer.step()

            time_out.append(time.time() - start)

            if self.verbose:
                print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
                print('Loss = ' + str(loss_out[-1]))
                print('Time elapsed = ' + str(time_out[-1]))

        self.loss = loss_out
        self.train_time = time_out

    def fit_batch_sgd(self, emissions, inputs, learning_rate=1e-2, num_steps=50,
                      num_splits=2, batch_size=10, random_seed=None):
        """ This function will fit the model using batch stochastic gradient descent
        """
        assert(type(emissions) is list)
        assert(type(inputs) is list)

        rng = np.random.default_rng(random_seed)

        if emissions[0].dtype is not self.dtype:
            emissions = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions]
            inputs = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in inputs]

        num_epochs = int(np.ceil(num_steps / batch_size))

        # split the dataset to reduce memory consumption
        emissions, inputs = self._split_data(emissions, inputs, num_splits=num_splits)

        # initialize the optimizer with the parameters we're going to optimize
        model_params = [self.dynamics_weights, self.inputs_weights_log, self.dynamics_cov_log, self.emissions_cov_log]

        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        loss_out = []
        time_out = []

        init_mean = [torch.nanmean(i, dim=0) for i in emissions]
        init_cov = [self._estimate_cov(i) for i in emissions]

        start = time.time()
        for ep in range(num_epochs):
            # randomize the batches
            batch_inds = self._get_batches_inds(len(emissions), batch_size=batch_size, generator=rng)

            for bi, b in enumerate(batch_inds):
                emissions_batch = [emissions[i] for i in b]
                inputs_batch = [inputs[i] for i in b]
                init_mean_batch = [init_mean[i] for i in b]
                init_cov_batch = [init_cov[i] for i in b]

                optimizer.zero_grad()
                loss = self.loss_fn(emissions_batch, inputs_batch, init_mean_batch, init_cov_batch)
                loss.backward()
                loss_out.append(loss.detach().cpu().numpy())
                optimizer.step()

                if self.verbose:
                    print('Finished batch ' + str(bi + 1) + '/' + str(len(batch_inds)) + ', Epoch # ' + str(ep + 1))
                    print('Loss = ' + str(loss_out[-1]))
                    print('')

            time_out.append(time.time() - start)

            if self.verbose:
                print('Finished epoch ' + str(ep + 1) + '/' + str(num_epochs))
                print('Time elapsed = ' + str(time_out[-1]))
                print('')

        self.loss = loss_out

    def loss_fn(self, emissions, inputs, init_mean, init_cov, nan_fill=1e8):
        loss = self._lgssm_filter(emissions, inputs, init_mean, init_cov, nan_fill=nan_fill)[0]

        return -torch.sum(loss) / emissions.numel()

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

        for d in range(num_data_sets):
            latents[d, 0, :] = rng.multivariate_normal(dynamics_weights @ init_mean[d, :], init_cov[d, :, :])
            emissions[d, 0, :] = rng.multivariate_normal(latents[d, 0, :], np.diag(emissions_cov))

        for t in range(1, num_time):
            dynamics_noise = rng.multivariate_normal(np.zeros(self.latent_dim), np.diag(dynamics_cov), size=num_data_sets)
            emissions_noise = rng.multivariate_normal(np.zeros(self.latent_dim), np.diag(emissions_cov), size=num_data_sets)

            latents[:, t, :] = (dynamics_weights[None, :, :] @ latents[:, t - 1, :, None])[:, :, 0] + inputs_weights[None, :] * inputs[:, t, :] + dynamics_noise
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

    def _lgssm_filter(self, emissions, inputs, init_mean, init_cov, nan_fill=1e8):
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

            # added by msc to attempt to perform inference over nans
            nan_loc = torch.isnan(y)
            y = torch.where(nan_loc, 0, y)
            R = torch.where(nan_loc, nan_fill, torch.tile(emissions_cov[None, :], (num_data_sets, 1)))

            # Update the log likelihood
            mu = pred_mean
            cov = pred_cov + torch.diag_embed(R)

            y_mean_sub = y - mu
            ll += -torch.linalg.slogdet(cov)[1] - \
                  (y_mean_sub[:, None, :] @ torch.linalg.solve(cov, y_mean_sub)[:, :, None])[:, 0, 0]

            # Condition on this emission
            # Compute the Kalman gain
            K = torch.permute(torch.linalg.solve(cov, pred_cov), (0, 2, 1))

            filtered_cov = pred_cov - K @ cov @ torch.permute(K, (0, 2, 1))
            filtered_mean = (pred_mean[:, :, None] + K @ y_mean_sub[:, :, None])[:, :, 0]

            filtered_covs_list.append(filtered_cov)
            filtered_means_list.append(filtered_mean)

            # Predict the next state
            pred_mean = (dynamics_weights[None, :, :] @ filtered_mean[:, :, None])[:, :, 0] + inputs_weights[None, :] * inputs[:, t, :]
            pred_cov = dynamics_weights[None, :, :] @ filtered_cov @ dynamics_weights.T[None, :, :] + torch.diag_embed(dynamics_cov[None, :])

        filtered_means = torch.stack(filtered_means_list)
        filtered_covs = torch.stack(filtered_covs_list)

        return ll, filtered_means, filtered_covs

    @staticmethod
    def _get_batches_inds(num_data, batch_size, generator):
        num_batches = np.ceil(num_data / batch_size)
        data_inds = np.arange(num_data)
        generator.shuffle(data_inds)
        batch_data_inds = np.array_split(data_inds, num_batches)

        return batch_data_inds

    @staticmethod
    def stack_data(emissions_list, inputs_list):
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
