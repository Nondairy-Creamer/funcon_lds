import time
from mpi4py import MPI
from mpi4py.util import pkl5
import numpy as np
import loading_utilities as lu


def block(block_list, dims=(2, 1)):
    layer = []
    for i in block_list:
        layer.append(np.concatenate(i, axis=dims[0]))

    return np.concatenate(layer, axis=dims[1])


def individual_scatter(data, root=0):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    if cpu_id == root:
        item = None

        for i, attr in enumerate(data):
            if i == 0:
                item = attr
            else:
                comm.send(attr, dest=i)

        for i in range(len(data), size):
            comm.send(None, dest=i)
    else:
        item = comm.recv(source=root)

    return item


def individual_gather(data, root=0):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    item = []

    if cpu_id == root:
        for i in range(size):
            if i == root:
                item.append(data)
            else:
                item.append(comm.recv(source=i))

    else:
        comm.send(data, dest=root)

    return item


def solve_masked(A, b, mask=None, ridge_penalty=None):
    # solves the linear equation b=Ax where x has 0's where mask == 0
    x_hat = np.zeros((A.shape[1], b.shape[1]))

    if mask is None:
        mask = np.ones_like(x_hat)

    for i in range(b.shape[1]):
        non_zero_loc = mask[:, i] != 0

        b_i = b[:, i]

        if ridge_penalty is None:
            A_nonzero = A[:, non_zero_loc]
        else:
            r_size = ridge_penalty.shape[0]
            penalty = ridge_penalty[i] * np.eye(r_size)[:, non_zero_loc[:r_size]]
            A_nonzero = A[:, non_zero_loc]
            A_nonzero[:r_size, :penalty.shape[1]] += penalty

        x_hat[non_zero_loc, i] = np.linalg.lstsq(A_nonzero, b_i, rcond=None)[0]

    return x_hat


def fit_em(model, data, init_mean=None, init_cov=None, num_steps=10,
           save_folder='em_test', save_every=20, memmap_cpu_id=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    if cpu_id == 0:
        print('Fitting with EM')

        emissions = data['emissions']
        inputs = data['inputs']

        if len(emissions) < size:
            raise Exception('Number of cpus must be <= number of data sets')

        # lag the inputs if the model has lags
        inputs = [model.get_lagged_data(i, model.dynamics_input_lags) for i in inputs]

        if init_mean is None:
            init_mean = model.estimate_init_mean(emissions)

        if init_cov is None:
            init_cov = model.estimate_init_cov(emissions)

    else:
        emissions = None
        inputs = None
        init_mean = None
        init_cov = None

    log_likelihood_out = []
    time_out = []

    start = time.time()
    for ep in range(num_steps):
        model = comm.bcast(model, root=0)

        ll, smoothed_means, new_init_covs = \
            model.em_step(emissions, inputs, init_mean, init_cov, cpu_id=cpu_id, num_cpus=size, memmap_cpu_id=memmap_cpu_id)

        if cpu_id == 0:
            # set the initial mean and cov to the first smoothed mean / cov
            for i in range(len(smoothed_means)):
                init_mean[i] = smoothed_means[i][0, :]
                if type(new_init_covs[i]) is tuple:
                    init_cov[i] = new_init_covs[i][0]
                else:
                    init_cov[i] = new_init_covs[i]

            log_likelihood_out.append(ll)
            time_out.append(time.time() - start)
            model.log_likelihood = log_likelihood_out
            model.train_time = time_out

            if np.mod(ep + 1, save_every) == 0:
                posterior_train = {'ll': ll,
                                   'posterior': smoothed_means,
                                   'init_mean': init_mean,
                                   'init_cov': init_cov,
                                   }

                lu.save_run(save_folder, model_trained=model, posterior_train=posterior_train)

            if model.verbose:
                print('Finished step', ep + 1, '/', num_steps)
                print('log likelihood =', log_likelihood_out[-1])
                print('Time elapsed =', time_out[-1], 's')
                time_remaining = time_out[-1] / (ep + 1) * (num_steps - ep - 1)
                print('Estimated remaining =', time_remaining, 's')

    if cpu_id == 0:
        return ll, model, init_mean, init_cov
    else:
        return None, None, None, None


def parallel_get_post(model, data_test, init_mean=None, init_cov=None, max_iter=1, converge_res=1e-2, time_lim=100,
                      memmap_cpu_id=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    if cpu_id == 0:
        emissions = data_test['emissions']
        inputs = data_test['inputs']

        if init_mean is None:
            init_mean = model.estimate_init_mean(emissions)

        if init_cov is None:
            init_cov = model.estimate_init_cov(emissions)

        test_data_packaged = model.package_data_mpi(emissions, inputs, init_mean, init_cov, size)
    else:
        test_data_packaged = None

    # get posterior on test data
    model = comm.bcast(model)
    data_test_out = individual_scatter(test_data_packaged)

    if data_test_out is not None:
        ll_smeans = []
        for ii, i in enumerate(data_test_out):
            emissions = i[0][:time_lim, :].copy()
            inputs = i[1][:time_lim, :].copy()
            init_mean = i[2].copy()
            init_cov = i[3].copy()
            converged = False
            iter_num = 1

            while not converged and iter_num <= max_iter:
                ll, smoothed_means, suff_stats = model.lgssm_smoother(emissions, inputs, init_mean, init_cov,
                                                                      memmap_cpu_id=memmap_cpu_id)

                init_mean_new = smoothed_means[0, :].copy()
                init_cov_new = suff_stats['first_cov'].copy()

                init_mean_same = np.max(np.abs(init_mean - init_mean_new)) < converge_res
                init_cov_same = np.max(np.abs(init_cov - init_cov_new)) < converge_res
                if init_mean_same and init_cov_same:
                    converged = True
                else:
                    init_mean = init_mean_new.copy()
                    init_cov = init_cov_new.copy()

                print('cpu_id', cpu_id + 1, '/', size, 'data #', ii + 1, '/', len(data_test_out),
                      'posterior iteration:', iter_num, ', converged:', converged)
                iter_num += 1

            emissions = i[0].copy()
            inputs = i[1].copy()

            ll, posterior, suff_stats = model.lgssm_smoother(emissions, inputs, init_mean, init_cov, memmap_cpu_id)
            post_pred = model.sample(num_time=emissions.shape[0], inputs_list=[inputs], init_mean=init_mean, init_cov=init_cov, add_noise=False)
            post_pred_noise = model.sample(num_time=emissions.shape[0], inputs_list=[inputs], init_mean=init_mean, init_cov=init_cov)

            posterior = posterior[:, :model.dynamics_dim]
            post_pred = post_pred['latents'][0][:, :model.dynamics_dim]
            post_pred_noise = post_pred_noise['latents'][0][:, :model.dynamics_dim]

            ll_smeans.append((ll, posterior, post_pred, post_pred_noise, init_mean, init_cov))
    else:
        ll_smeans = None

    ll_smeans = individual_gather(ll_smeans)
    ll_smeans = [i for i in ll_smeans if i is not None]

    if cpu_id == 0:
        ll_smeans_out = []
        for i in ll_smeans:
            for j in i:
                ll_smeans_out.append(j)

        ll_smeans = ll_smeans_out

        ll = [i[0] for i in ll_smeans]
        ll = np.sum(ll)
        smoothed_means = [i[1] for i in ll_smeans]
        post_pred = [i[2] for i in ll_smeans]
        post_pred_noise = [i[3] for i in ll_smeans]
        init_mean = [i[4] for i in ll_smeans]
        init_cov = [i[5] for i in ll_smeans]

        inference_test = {'ll': ll,
                          'posterior': smoothed_means,
                          'post_pred': post_pred,
                          'post_pred_noise': post_pred_noise,
                          'init_mean': init_mean,
                          'init_cov': init_cov,
                          }

        return inference_test

    return None

