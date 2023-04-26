import torch
import time
from mpi4py import MPI


def batch_Ax(A, x):
    return (A @ x[:, :, None])[:, :, 0]


def block(block_list, dims=(2, 1)):
    layer = []
    for i in block_list:
        layer.append(torch.cat(i, dim=dims[0]))

    return torch.cat(layer, dim=dims[1])


def batch_trans(batch_matrix):
    return torch.permute(batch_matrix, (0, 2, 1))


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


def fit_em(model, emissions_list, inputs_list, init_mean=None, init_cov=None, num_steps=10, is_parallel=False):
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

        ll = model.em_step_pillow(emissions, inputs, init_mean, init_cov, is_parallel=is_parallel)

        if ll is None:
            continue

        log_likelihood_out.append(ll.detach().cpu().numpy())
        time_out.append(time.time() - start)

        if model.verbose:
            print('Finished step ' + str(ep + 1) + '/' + str(num_steps))
            print('log likelihood = ' + str(log_likelihood_out[-1]))
            print('Time elapsed = ' + str(time_out[-1]))

    model.log_likelihood = log_likelihood_out
    model.train_time = time_out

    return model

