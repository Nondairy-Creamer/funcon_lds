import torch
import time
from mpi4py import MPI
import numpy as np
import loading_utilities as lu


def block(block_list, dims=(2, 1)):
    layer = []
    for i in block_list:
        layer.append(torch.cat(i, dim=dims[0]))

    return torch.cat(layer, dim=dims[1])


def individual_scatter(data, root=0):
    comm = MPI.COMM_WORLD
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
    comm = MPI.COMM_WORLD
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


def solve_masked(A, b, mask):
    # solves the linear equation b=Ax where x has 0's where mask == 0

    dtype = A.dtype
    device = A.device
    x_hat = torch.zeros((A.shape[1], b.shape[1]), device=device, dtype=dtype)

    for i in range(b.shape[1]):
        non_zero_loc = mask[:, i] != 0

        b_i = b[:, i]
        A_nonzero = A[:, non_zero_loc]

        x_hat[non_zero_loc, i] = torch.linalg.lstsq(A_nonzero, b_i, rcond=None)[0]

    return x_hat


def fit_em(model, emissions_list, inputs_list, init_mean=None, init_cov=None, num_steps=10,
           save_folder='trained_models', save_every=10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        emissions, inputs = model.standardize_inputs(emissions_list, inputs_list)

        # lag the inputs if the model has lags
        inputs = [model.get_lagged_data(i, model.dynamics_input_lags) for i in inputs]

        if init_mean is None:
            init_mean = model.estimate_init_mean(emissions)
        else:
            init_mean = [torch.tensor(i, device=model.device, dtype=model.dtype) for i in init_mean]

        if init_cov is None:
            init_cov = model.estimate_init_cov(emissions)
        else:
            init_cov = [torch.tensor(i, device=model.device, dtype=model.dtype) for i in init_cov]
    else:
        emissions = None
        inputs = None

    log_likelihood_out = []
    time_out = []
    smoothed_means = None

    start = time.time()
    for ep in range(num_steps):
        ll, smoothed_means = model.em_step(emissions, inputs, init_mean, init_cov, cpu_id=rank, num_cpus=size)

        if ll is None:
            continue

        log_likelihood_out.append(ll.detach().cpu().numpy())
        time_out.append(time.time() - start)
        model.log_likelihood = log_likelihood_out
        model.train_time = time_out

        if np.mod(ep, save_every) == 0:
            lu.save_run(save_folder, model_trained=model, posterior=smoothed_means)

        if model.verbose:
            print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
            print('log likelihood = ' + str(log_likelihood_out[-1]))
            print('Time elapsed = ' + str(time_out[-1]))

    return model, smoothed_means


