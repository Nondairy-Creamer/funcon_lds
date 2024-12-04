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


def two_sample_bootstrap_paired(x, y, func=np.mean, n_boot=1000, rng=np.random.default_rng(0)):
    # get rid of nans
    x = x.reshape(-1)
    y = y.reshape(-1)
    nan_loc = np.isnan(x) | np.isnan(y)
    x = x[~nan_loc]
    y = y[~nan_loc]

    n_data = x.shape[0]

    stat = np.zeros(n_boot)
    for n in range(n_boot):
        sample_inds = rng.integers(0, high=n_data, size=n_data)
        x_sampled = x[sample_inds]
        y_sampled = y[sample_inds]

        stat[n] = func(x_sampled) - func(y_sampled)

    if np.mean(stat < 0):
        stat *= -1

    p = np.mean(stat < 0) * 2

    return p


def two_sample_boostrap_corr_p(target, data_1, data_2, alpha=0.05, n_boot=1000, rng=np.random.default_rng()):
    booted_metric = np.zeros(n_boot)

    # get rid of nans
    target = target.reshape(-1).astype(float)
    data_1 = data_1.reshape(-1).astype(float)
    data_2 = data_2.reshape(-1).astype(float)

    nan_loc = np.isnan(target) | np.isnan(data_1) | np.isnan(data_2)
    target = target[~nan_loc]
    data_1 = data_1[~nan_loc]
    data_2 = data_2[~nan_loc]
    n_data = target.shape[0]

    for n in range(n_boot):
        sample_inds = rng.integers(0, high=n_data, size=n_data)
        target_resampled = target[sample_inds]
        data_1_resampled = data_1[sample_inds]
        data_2_resampled = data_2[sample_inds]

        data_1_corr = nan_corr(target_resampled, data_1_resampled)[0]
        data_2_corr = nan_corr(target_resampled, data_2_resampled)[0]

        booted_metric[n] = data_1_corr - data_2_corr

    data_mean = nan_corr(target, data_1)[0] - nan_corr(target, data_2)[0]

    ci = [np.percentile(booted_metric, alpha / 2 * 100),
          np.percentile(booted_metric, (1 - alpha / 2) * 100)]
    ci = np.abs(np.array(ci) - data_mean)

    if np.median(booted_metric) < 0:
        booted_metric *= -1

    p = 2 * np.mean(booted_metric <= 0)

    return p, data_mean, ci


def two_sample_corr_p(target, data_1, data_2, alpha=0.05):
    # get rid of nans
    target = target.reshape(-1).astype(float).copy()
    data_1 = data_1.reshape(-1).astype(float).copy()
    data_2 = data_2.reshape(-1).astype(float).copy()

    nan_loc = np.isnan(target) | np.isnan(data_1) | np.isnan(data_2)
    target = target[~nan_loc]
    data_1 = data_1[~nan_loc]
    data_2 = data_2[~nan_loc]
    n = target.shape[0]

    s = np.corrcoef(np.concatenate((target[None, :], data_1[None, :], data_2[None, :]), axis=0))
    s_det = np.linalg.det(s)
    numerator = (n - 1) * (1 + s[1, 2])
    denom_1 = 2 * ((n - 1) / (n - 3)) * s_det
    denom_2 = 1/4 * (s[0, 1] + s[0, 2])**2 * (1 - s[1, 2])**3
    t = (s[0, 1] - s[0, 2]) * np.sqrt(numerator / (denom_1 + denom_2))
    p = scipy.stats.t.cdf(t, n - 3) * 2

    return p


def nan_corr(y_true, y_hat, alpha=0.05, mean_sub=True):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    if mean_sub:
        y_true = y_true - np.mean(y_true)
        y_hat = y_hat - np.mean(y_hat)

    y_true_std = np.std(y_true, ddof=0)
    y_hat_std = np.std(y_hat, ddof=0)

    corr = np.mean(y_true * y_hat) / y_true_std / y_hat_std

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

    true_positives = np.sum((y_true == 1) & (y_hat == 1))
    true_negatives = np.sum((y_true == 0) & (y_hat == 0))
    total = y_true.shape[0]

    return (true_positives + true_negatives) / total


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

    mi_ci[0] = np.percentile(booted_mi, alpha / 2 * 100)
    mi_ci[1] = np.percentile(booted_mi, (1 - alpha / 2) * 100)

    mi_ci = np.abs(mi_ci - mi)

    return mi, mi_ci


def metric_null(metric, y_true, n_sample=1000, rng=np.random.default_rng()):
    y_true = y_true.reshape(-1)

    nan_loc = np.isnan(y_true)
    y_true = y_true[~nan_loc]

    py = np.mean(y_true)
    if py > 0.5:
        py = 1
    else:
        py = 0
    sampled_mi = np.zeros(n_sample)

    for n in range(n_sample):
        random_example = rng.uniform(0, 1, size=y_true.shape) < py
        sampled_mi[n] = metric(y_true, random_example)

    return np.mean(sampled_mi)

