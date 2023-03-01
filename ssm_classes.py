import torch
import numpy as np
import time


class LgssmSimple:
    """Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
        adapted from Dynamax.
        This verison assumes diagonal input weights, diagonal noise covariance, no offsets, and identity emissions
    """

    def __init__(self, latent_dim, dtype=torch.float64, device='cpu', random_seed=0, verbose=False):
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = device
        self.verbose = verbose

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

        # this is only used for randomizing weights, sampling from the model, and selecting batches
        self._random_generator = np.random.default_rng(random_seed)

    def randomize_weights(self, max_eig_allowed=0.8, init_std=0.1, cov_log_offset=-1):
        self.dynamics_weights_init = self._random_generator.standard_normal((self.latent_dim, self.latent_dim))
        max_eig_in_mat = np.max(np.abs(np.linalg.eigvals(self.dynamics_weights_init)))
        self.dynamics_weights_init = max_eig_allowed * self.dynamics_weights_init / max_eig_in_mat

        self.inputs_weights_log_init = init_std * self._random_generator.standard_normal(self.latent_dim) + cov_log_offset
        self.dynamics_cov_log_init = init_std * self._random_generator.standard_normal(self.latent_dim) + cov_log_offset
        self.emissions_cov_log_init = init_std * self._random_generator.standard_normal(self.latent_dim) + cov_log_offset

        self.dynamics_weights = torch.tensor(self.dynamics_weights_init, device=self.device,
                                             dtype=self.dtype, requires_grad=True)
        self.inputs_weights_log = torch.tensor(self.inputs_weights_log_init, device=self.device,
                                               dtype=self.dtype, requires_grad=True)
        self.dynamics_cov_log = torch.tensor(self.dynamics_cov_log_init, device=self.device,
                                             dtype=self.dtype, requires_grad=True)
        self.emissions_cov_log = torch.tensor(self.emissions_cov_log_init, device=self.device,
                                              dtype=self.dtype, requires_grad=True)

    def fit_gd(self, emissions, inputs, learning_rate=1e-2, num_steps=50):
        """ This function will fit the model using gradient descent on the entire data set
        """
        assert(type(emissions) is list)
        assert(type(inputs) is list)

        if emissions[0].dtype is not self.dtype:
            emissions = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in emissions]
            inputs = [torch.tensor(i, device=self.device, dtype=self.dtype) for i in inputs]

        # initialize the optimizer with the parameters we're going to optimize
        model_params = [self.dynamics_weights, self.inputs_weights_log, self.dynamics_cov_log, self.emissions_cov_log]

        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        loss_out = []

        init_mean = [torch.nanmean(i, dim=0) for i in emissions]
        init_cov = [self._estimate_cov(i) for i in emissions]

        start = time.time()
        for ep in range(num_steps):
            optimizer.zero_grad()
            loss = self.loss_fn(emissions, inputs, init_mean, init_cov)
            loss.backward()
            loss_out.append(loss.detach().cpu().numpy())
            optimizer.step()

            end = time.time()

            if self.verbose:
                print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
                print('Loss = ' + str(loss_out[-1]))
                print('Time elapsed = ' + str(end - start))

        return loss_out

    def fit_batch_sgd(self, emissions, inputs, learning_rate=1e-2, num_steps=50, num_splits=2, batch_size=10):
        """ This function will fit the model using batch stochastic gradient descent
        """
        assert(type(emissions) is list)
        assert(type(inputs) is list)

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

        init_mean = [torch.nanmean(i, dim=0) for i in emissions]
        init_cov = [self._estimate_cov(i) for i in emissions]

        start = time.time()
        for ep in range(num_epochs):
            # randomize the batches
            batch_inds = self._get_batches_inds(len(emissions), batch_size=batch_size)

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

            end = time.time()

            if self.verbose:
                print('Finished epoch ' + str(ep + 1) + '/' + str(num_epochs))
                print('Time elapsed = ' + str(end - start))
                print('')

        return loss_out

    def loss_fn(self, emissions, inputs, init_mean, init_cov, nan_fill=1e8):
        inputs_weights = torch.exp(self.inputs_weights_log)
        dynamics_cov = torch.exp(self.dynamics_cov_log)
        emissions_cov = torch.exp(self.emissions_cov_log)
        loss = torch.tensor(0, dtype=self.dtype, device=self.device)
        num_data_points = [i.numel() for i in emissions]
        num_data_points = np.sum(num_data_points)

        for d in range(len(emissions)):
            loss += self._lgssm_filter(emissions[d], inputs[d], init_mean[d], init_cov[d],
                                       self.dynamics_weights, inputs_weights, dynamics_cov, emissions_cov,
                                       nan_fill=nan_fill)[0]

        return -loss / num_data_points

    def _lgssm_filter(self, emissions, inputs, init_mean, init_cov,
                      dynamics_weights, inputs_weights, dynamics_cov, emissions_cov,
                      nan_fill=1e8):
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
        num_timesteps = len(emissions)

        if inputs is None:
            inputs = torch.zeros((num_timesteps, 0), dtype=self.dtype, device=self.device)

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

            ll += -torch.linalg.slogdet(cov)[1] - torch.dot(y - mu, torch.linalg.solve(cov, y - mu))

            # Condition on this emission
            # Compute the Kalman gain
            S = torch.diag(R) + pred_cov
            K = torch.linalg.solve(S, pred_cov).T
            filtered_cov = pred_cov - K @ S @ K.T
            filtered_mean = pred_mean + K @ (y - pred_mean)

            # Predict the next state
            pred_mean = dynamics_weights @ filtered_mean + inputs_weights * inputs[t]
            pred_cov = dynamics_weights @ filtered_cov @ dynamics_weights.T + torch.diag(dynamics_cov)

            return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

        # Run the Kalman filter
        carry = (0.0, init_mean, init_cov)
        (ll, _, _), (filtered_means, filtered_covs) = self._scan(_step, carry, torch.arange(num_timesteps))

        return ll, filtered_means, filtered_covs

    def sample(self, num_time=100, num_data_sets=1, nan_freq=0.0):
        all_latents = []
        all_emissions = []
        all_inputs = []
        init_mean = []
        init_cov = []

        dynamics_weights_np = self.dynamics_weights.detach().cpu().numpy().copy()
        inputs_weights = np.exp(self.inputs_weights_log.detach().cpu().numpy().copy())
        dynamics_cov = np.exp(self.dynamics_cov_log.detach().cpu().numpy().copy())
        emissions_cov = np.exp(self.emissions_cov_log.detach().cpu().numpy().copy())

        for d in range(num_data_sets):
            # generate a random initial mean and covariance
            init_mean.append(self._random_generator.standard_normal(self.latent_dim))
            init_cov.append(self._random_generator.standard_normal((self.latent_dim, self.latent_dim)))
            init_cov[-1] = init_cov[-1].T @ init_cov[-1] / self.latent_dim

            latents = np.zeros((num_time, self.latent_dim))
            emissions = np.zeros((num_time, self.latent_dim))

            inputs = self._random_generator.standard_normal((num_time, self.latent_dim))
            latents[0, :] = self._random_generator.multivariate_normal(dynamics_weights_np @ init_mean[-1], init_cov[-1])
            emissions[0, :] = self._random_generator.multivariate_normal(latents[0, :], np.diag(emissions_cov))

            for t in range(1, num_time):
                dynamics_noise = self._random_generator.multivariate_normal(np.zeros(self.latent_dim), np.diag(dynamics_cov))
                emissions_noise = self._random_generator.multivariate_normal(np.zeros(self.latent_dim), np.diag(emissions_cov))
                latents[t, :] = dynamics_weights_np @ latents[t - 1, :] + inputs_weights * inputs[t, :] + dynamics_noise
                emissions[t, :] = latents[t, :] + emissions_noise

            # add in nans
            nan_mask = self._random_generator.random((num_time, self.latent_dim)) <= nan_freq
            emissions[nan_mask] = np.nan

            all_emissions.append(emissions)
            all_inputs.append(inputs)
            all_latents.append(latents)

        return all_emissions, all_inputs, all_latents, init_mean, init_cov

    def _get_batches_inds(self, num_data, batch_size):
        num_batches = np.ceil(num_data / batch_size)
        data_inds = np.arange(num_data)
        self._random_generator.shuffle(data_inds)
        batch_data_inds = np.array_split(data_inds, num_batches)

        return batch_data_inds

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

        a_mean_sub = a - torch.nanmean(a, dim=0, keepdims=True)
        # estimate the covariance from the data in a
        cov = nan_matmul(a_mean_sub.T, a_mean_sub) / a.shape[0]

        # some columns will be all 0s due to missing data
        # replace those diagonals with the mean covariance
        cov_diag = torch.diag(cov)
        cov_diag_mean = torch.mean(cov_diag[cov_diag != 0])
        cov_diag = torch.where(cov_diag == 0, cov_diag_mean, cov_diag)

        cov[torch.eye(a.shape[1], dtype=torch.bool)] = cov_diag

        return cov

    @staticmethod
    def _scan(f, init, xs, length=None):
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
