import numpy as np
import scipy


def nan_r2(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot

    return r2


def nan_corr(y_true, y_hat, alpha=0.05, mean_sub=True):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    if mean_sub:
        y_true = y_true - np.mean(y_true)
        y_hat = y_hat - np.mean(y_hat)

    y_true_std = np.std(y_true, ddof=1)
    y_hat_std = np.std(y_hat, ddof=1)

    corr = (np.mean(y_true * y_hat) / y_true_std / y_hat_std)

    # now estimate the confidence intervals for the correlation
    n = y_true.shape[0]
    z_a = scipy.stats.norm.ppf(1 - alpha / 2)
    z_r = np.log((1 + corr) / (1 - corr)) / 2
    l = z_r - (z_a / np.sqrt(n - 3))
    u = z_r + (z_a / np.sqrt(n - 3))
    ci_l = (np.exp(2 * l) - 1) / (np.exp(2 * l) + 1)
    ci_u = (np.exp(2 * u) - 1) / (np.exp(2 * u) + 1)
    ci = [np.abs(ci_l - corr), ci_u - corr]

    return corr, ci


def accuracy(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    return np.mean(y_true == y_hat)


def precision(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    true_positives = np.sum((y_true == 1) & (y_hat == 1))
    false_positives = np.sum((y_true == 0) & (y_hat == 1))

    return true_positives / (true_positives + false_positives)


def recall(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    true_positives = np.sum((y_true == 1) & (y_hat == 1))
    false_negatives = np.sum((y_true == 1) & (y_hat == 0))

    return true_positives / (true_positives + false_negatives)


def f_measure(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    p = precision(y_true, y_hat)
    r = recall(y_true, y_hat)

    return (2 * p * r) / (p + r)


def mutual_info(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    p_y_true = np.array([1 - np.mean(y_true), np.mean(y_true)])
    p_y_hat = np.array([1 - np.mean(y_hat), np.mean(y_hat)])

    p_joint = np.zeros((2, 2))
    p_joint[0, 0] = np.mean((y_true == 0) & (y_hat == 0))
    p_joint[1, 0] = np.mean((y_true == 1) & (y_hat == 0))
    p_joint[0, 1] = np.mean((y_true == 0) & (y_hat == 1))
    p_joint[1, 1] = np.mean((y_true == 1) & (y_hat == 1))

    p_outer = p_y_true[:, None] * p_y_hat[None, :]

    mi = 0
    for i in range(2):
        for j in range(2):
            if p_joint[i, j] != 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / p_outer[i, j])

    return mi


def metric_ci(metric, y_true, y_hat, alpha=0.05, n_boot=1000, rng=np.random.default_rng()):
    y_true = y_true.astype(float)
    y_hat = y_hat.astype(float)

    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    nan_loc = np.isnan(y_true) | np.isnan(y_hat)
    y_true = y_true[~nan_loc]
    y_hat = y_hat[~nan_loc]

    mi = metric(y_true, y_hat)
    booted_mi = np.zeros(n_boot)
    mi_ci = np.zeros(2)

    for n in range(n_boot):
        sample_inds = rng.integers(0, high=y_true.shape[0], size=y_true.shape[0])
        y_true_resampled = y_true[sample_inds]
        y_hat_resampled = y_hat[sample_inds]
        booted_mi[n] = metric(y_true_resampled, y_hat_resampled)

    mi_ci[0] = np.percentile(booted_mi, alpha * 100)
    mi_ci[1] = np.percentile(booted_mi, (1 - alpha) * 100)

    mi_ci = np.abs(mi_ci - mi)

    return mi, mi_ci


def metric_null(metric, y_true, n_sample=1000, rng=np.random.default_rng()):
    y_true = y_true.reshape(-1)

    nan_loc = np.isnan(y_true)
    y_true = y_true[~nan_loc]

    py = np.mean(y_true)
    sampled_mi = np.zeros(n_sample)

    for n in range(n_sample):
        random_example = rng.uniform(0, 1, size=y_true.shape) < py
        sampled_mi[n] = metric(y_true, random_example)

    return np.mean(sampled_mi)