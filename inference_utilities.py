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
    rank = comm.Get_rank()

    if rank == root:
        item = None

        for i, attr in enumerate(data):
            if i == 0:
                item = attr
            else:
                comm.send(attr, dest=i)
    else:
        item = comm.recv(source=root)

    return item


def individual_gather(data, root=0):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    item = []

    if rank == root:
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


def fit_em(model, emissions, inputs, init_mean=None, init_cov=None, num_steps=10,
           save_folder='em_test', save_every=10):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(emissions) < size:
        raise Exception('More CPUs requested than number of data sets. Number of cpus must be <= number of data sets')

    if rank == 0:
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
    smoothed_means = None

    print('Fitting with EM')
    start = time.time()
    for ep in range(num_steps):
        model = comm.bcast(model, root=0)

        ll, smoothed_means, smoothed_covs = \
            model.em_step(emissions, inputs, init_mean, init_cov, cpu_id=rank, num_cpus=size)

        if rank == 0:
            # set the initial mean and cov to the first smoothed mean / cov
            for i in range(len(smoothed_means)):
                init_mean[i] = smoothed_means[i][0, :]
                if type(smoothed_covs[i]) is tuple:
                    init_cov[i] = smoothed_covs[i][0][0, :, :]
                else:
                    init_cov[i] = smoothed_covs[i][0, :, :]

            log_likelihood_out.append(ll)
            time_out.append(time.time() - start)
            model.log_likelihood = log_likelihood_out
            model.train_time = time_out

            if np.mod(ep, save_every-1) == 0:
                initial_conditions = {'init_mean': init_mean, 'init_cov': init_cov}
                lu.save_run(save_folder, model, posterior=smoothed_means, initial_conditions=initial_conditions)

            if model.verbose:
                print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
                print('log likelihood = ' + str(log_likelihood_out[-1]))
                print('Time elapsed = ' + str(time_out[-1]))

    return model, smoothed_means, init_mean, init_cov


