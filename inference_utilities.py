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


def individual_scatter(data, root=0, num_data=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        item = None

        for i, attr in enumerate(data):
            if i == 0:
                item = attr
            else:
                comm.send(attr, dest=i)
    else:
        if num_data is None:
            num_data = size

        if rank < num_data:
            item = comm.recv(source=root)
        else:
            item = None

    return item


def individual_gather(data, root=0, num_data=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    item = []

    if rank == root:
        if num_data is None:
            num_data = size

        for i in range(num_data):
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


def fit_em(model, emissions, inputs, data_test, init_mean=None, init_cov=None, num_steps=10,
           save_folder='em_test', save_every=20, memmap_rank=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print('Fitting with EM')

        if len(emissions) < size:
            raise Exception('Number of cpus must be <= number of data sets')

        # lag the inputs if the model has lags
        inputs = [model.get_lagged_data(i, model.dynamics_input_lags) for i in inputs]

        test_size = len(data_test['emissions'])

        if init_mean is None:
            init_mean = model.estimate_init_mean(emissions)

        if init_cov is None:
            init_cov = model.estimate_init_cov(emissions)

    else:
        emissions = None
        inputs = None
        init_mean = None
        init_cov = None
        test_size = None

    log_likelihood_out = []
    time_out = []
    smoothed_means = None
    ll = None
    init_mean_test = None
    init_cov_test = None
    test_size = comm.bcast(test_size, root=0)

    start = time.time()
    for ep in range(num_steps):
        model = comm.bcast(model, root=0)

        ll, smoothed_means, new_init_covs = \
            model.em_step(emissions, inputs, init_mean, init_cov, cpu_id=rank, num_cpus=size, memmap_rank=memmap_rank)

        if rank == 0:
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

            if model.verbose:
                print('Finished step', ep + 1, '/', num_steps)
                print('log likelihood =', log_likelihood_out[-1])
                print('Time elapsed =', time_out[-1], 's')
                time_remaining = time_out[-1] / (ep + 1) * (num_steps - ep - 1)
                print('Estimated remaining =', time_remaining, 's')

        if np.mod(ep + 1, save_every) == 0:
            if rank == 0:
                print('saving intermediate posterior of the test data')

            inference_test = parallel_get_post(model, data_test,
                                               init_mean=init_mean_test, init_cov=init_cov_test)

            if rank == 0:
                init_mean_test = inference_test['init_mean']
                init_cov_test = inference_test['init_cov']

                inference_train = {'ll': ll,
                                   'posterior': smoothed_means,
                                   'init_mean': init_mean,
                                   'init_cov': init_cov,
                                   }

                lu.save_run(save_folder, model, inference_train=inference_train, inference_test=inference_test)

    return ll, model, smoothed_means, init_mean, init_cov


def parallel_get_post(model, data_test, init_mean=None, init_cov=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        emissions = data_test['emissions']
        inputs = data_test['inputs']
        # get the cpu_ids for running the model on the test data
        data_test_size = np.min((size, len(emissions)))

        if init_mean is None:
            init_mean = model.estimate_init_mean(emissions)

        if init_cov is None:
            init_cov = model.estimate_init_cov(emissions)

        test_data_packaged = model.package_data_mpi(emissions, inputs, init_mean, init_cov, data_test_size)
    else:
        data_test_size = None
        test_data_packaged = None

    # get posterior on test data
    data_test_size = comm.bcast(data_test_size)
    model = comm.bcast(model)
    data_test_out = individual_scatter(test_data_packaged, num_data=data_test_size)

    if data_test_out is not None:
        ll_smeans = []
        for ii, i in enumerate(data_test_out):
            emissions = i[0]
            inputs = i[1]
            init_mean = i[2]
            init_cov = i[3]
            converged = False
            iter_num = 1

            while not converged:
                ll, smoothed_means, suff_stats = model.lgssm_smoother(emissions, inputs, init_mean, init_cov)

                init_mean_new = smoothed_means[0, :]
                init_cov_new = suff_stats['first_cov']

                init_mean_same = np.max(np.abs(init_mean - init_mean_new)) < 1 / model.epsilon
                init_cov_same = np.max(np.abs(init_cov - init_cov_new)) < 1 / model.epsilon
                if init_mean_same and init_cov_same:
                    converged = True
                else:
                    init_mean = init_mean_new
                    init_cov = init_cov_new

                print('rank', rank + 1, '/', size, 'data #', ii + 1, '/', len(data_test_out),
                      'posterior iteration:', iter_num, ', converged:', converged)
                iter_num += 1

            post_pred = model.sample(num_time=emissions.shape[0], inputs_list=[inputs], init_mean=init_mean, init_cov=init_cov, add_noise=False)
            post_pred_noise = model.sample(num_time=emissions.shape[0], inputs_list=[inputs], init_mean=init_mean, init_cov=init_cov)

            ll_smeans.append((ll, smoothed_means, post_pred, post_pred_noise, init_mean, init_cov))

        ll_smeans = individual_gather(ll_smeans, num_data=data_test_size)

    if rank == 0:
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

