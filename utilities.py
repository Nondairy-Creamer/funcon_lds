import torch


def block(block_list, dims=(2, 1)):
    layer = []
    for i in block_list:
        layer.append(torch.cat(i, dim=dims[0]))

    return torch.cat(layer, dim=dims[1])


def batch_trans(batch_matrix):
    return torch.permute(batch_matrix, (0, 2, 1))


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