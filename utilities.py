import torch
import time
from mpi4py import MPI
import os
import numpy as np


def save_model_slurm(model, save_folder):
    if 'SLURM_JOB_ID' in os.environ:
        slurm_tag = '_' + os.environ['SLURM_JOB_ID']
    else:
        slurm_tag = ''

    true_model_save_path = save_folder + '/model' + slurm_tag + '_true.pkl'
    trained_model_save_path = save_folder + '/model' + slurm_tag + '_trained.pkl'

    # if there is an old "true" model delete it because it doesn't correspond to this trained model
    if os.path.exists(true_model_save_path):
        os.remove(true_model_save_path)

    model.save(path=trained_model_save_path)


def block(block_list, dims=(2, 1)):
    layer = []
    for i in block_list:
        layer.append(torch.cat(i, dim=dims[0]))

    return torch.cat(layer, dim=dims[1])


def estimate_cov(a):
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


def fit_em(model, emissions_list, inputs_list, init_mean=None, init_cov=None, num_steps=10, is_parallel=False,
           save_folder='trained_models', save_every=10):
    comm = MPI.COMM_WORLD

    if emissions_list is not None:
        emissions, inputs = model.standardize_inputs(emissions_list, inputs_list)

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

    start = time.time()
    for ep in range(num_steps):
        if is_parallel:
            model = comm.bcast(model, root=0)

        ll = model.em_step(emissions, inputs, init_mean, init_cov, is_parallel=is_parallel)

        if ll is None:
            continue

        if 'SLURM_JOB_ID' in os.environ:
            slurm_tag = '_' + os.environ['SLURM_JOB_ID']
        else:
            slurm_tag = ''

        log_likelihood_out.append(ll.detach().cpu().numpy())
        time_out.append(time.time() - start)
        model.log_likelihood = log_likelihood_out
        model.train_time = time_out

        if np.mod(ep, save_every) == 0:
            trained_model_save_path = save_folder + '/model' + slurm_tag + '_trained.pkl'
            model.save(path=trained_model_save_path)

        if model.verbose:
            print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
            print('log likelihood = ' + str(log_likelihood_out[-1]))
            print('Time elapsed = ' + str(time_out[-1]))

    return model

