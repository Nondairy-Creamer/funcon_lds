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


def individual_gather_sum(data, root=0):
    # as you gather inputs, rather than storing them sum them together
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    def combine_packet(packet):
        combined_packet = list(packet[0])
        combined_packet[2] = [combined_packet[2]]
        combined_packet[3] = [combined_packet[3]]

        for ii, i in enumerate(packet[1:]):
            combined_packet[0] += i[0]

            for k in i[1].keys():
                combined_packet[1][k] += i[1][k]

            combined_packet[2].append(i[2])
            combined_packet[3].append(i[3])

        return combined_packet

    if cpu_id == root:
        cpu_list = [i for i in range(size) if i != root]

        data_gathered = combine_packet(data)

        for cl in cpu_list:
            data_received = comm.recv(source=cl)

            data_received = combine_packet(data_received)

            data_gathered[0] += data_received[0]

            for k in data_received[1].keys():
                data_gathered[1][k] += data_received[1][k]

            for i in data_received[2]:
                data_gathered[2].append(i)

            for i in data_received[3]:
                data_gathered[3].append(i)

    else:
        comm.send(data, dest=root)
        data_gathered = None

    return data_gathered


def solve_masked(A, b, mask=None, ridge_penalty=None):
    # solves the linear equation b=Ax where x has 0's where mask == 0
    x_hat = np.zeros((A.shape[1], b.shape[1]))

    if mask is None:
        mask = np.ones_like(x_hat)

    for i in range(b.shape[1]):
        non_zero_loc = mask[:, i] != 0

        if ridge_penalty is not None:
            r_size = ridge_penalty.shape[0]
            penalty = ridge_penalty[i] * np.eye(r_size)
            A[:r_size, :r_size] = A[:r_size, :r_size] + penalty

        b_i = b[non_zero_loc, i]
        A_nonzero = A[np.ix_(non_zero_loc, non_zero_loc)]

        x_hat[non_zero_loc, i] = np.linalg.solve(A_nonzero, b_i)

    return x_hat


def fit_em(model, data, init_mean=None, init_cov=None, num_steps=10,
           save_folder='em_test', save_every=10, memmap_cpu_id=None, starting_step=0):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    if cpu_id == 0:
        print('Fitting with EM')

        emissions = data['emissions']
        inputs = data['inputs']

        if len(emissions) < size:
            raise Exception('Number of cpus must be <= number of data sets')

        if init_mean is None:
            init_mean = model.estimate_init_mean(emissions)

        if init_cov is None:
            init_cov = model.estimate_init_cov(emissions)

        starting_log_likelihood = model.log_likelihood
        starting_time = model.train_time

    else:
        emissions = None
        inputs = None
        init_mean = None
        init_cov = None

    log_likelihood_out = []
    time_out = []

    start = time.time()
    for ep in range(starting_step, starting_step + num_steps):
        model = comm.bcast(model, root=0)

        ll, smoothed_means, new_init_covs = \
            model.em_step(emissions, inputs, init_mean, init_cov, cpu_id=cpu_id, num_cpus=size, memmap_cpu_id=memmap_cpu_id)

        if cpu_id == 0:
            # set the initial mean and cov to the first smoothed mean / cov
            for i in range(len(smoothed_means)):
                init_mean[i] = smoothed_means[i][0, :]
                init_cov[i] = new_init_covs[i] / 2 + new_init_covs[i].T / 2

            log_likelihood_out.append(ll)
            time_out.append(time.time() - start)
            if starting_step > 0:
                model.log_likelihood = np.concatenate((starting_log_likelihood, log_likelihood_out))
                model.train_time = np.concatenate((starting_time, time_out))
            else:
                model.log_likelihood = log_likelihood_out
                model.train_time = time_out

            if np.mod(ep + 1, save_every) == 0:
                smoothed_means = [i[:, :model.dynamics_dim] for i in smoothed_means]

                posterior_train = {'ll': ll,
                                   'posterior': smoothed_means,
                                   'init_mean': init_mean,
                                   'init_cov': init_cov,
                                   }

                lu.save_run(save_folder, model_trained=model, ep=ep+1, posterior_train=posterior_train)

            if model.verbose:
                print('Finished step', ep + 1, '/', starting_step + num_steps)
                print('log likelihood =', log_likelihood_out[-1])
                print('Time elapsed =', time_out[-1], 's')
                time_remaining = time_out[-1] / (ep - starting_step + 1) * (num_steps - (ep - starting_step) - 1)
                print('Estimated remaining =', time_remaining, 's')

    if cpu_id == 0:
        return ll, model, init_mean, init_cov
    else:
        return None, None, None, None


def parallel_get_post(model, data, init_mean=None, init_cov=None, max_iter=1, converge_res=1e-2, time_lim=300,
                      memmap_cpu_id=None, infer_missing=False):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    if cpu_id == 0:
        emissions = data['emissions']
        inputs = data['inputs']

        if init_mean is None:
            init_mean = model.estimate_init_mean(emissions)

        if init_cov is None:
            init_cov = model.estimate_init_cov(emissions)

        test_data_packaged = model.package_data_mpi(emissions, inputs, init_mean, init_cov, size)
    else:
        test_data_packaged = None

    # get posterior on test data
    model = comm.bcast(model)
    data_out = individual_scatter(test_data_packaged)

    if data_out is not None:
        ll_smeans = []
        for ii, i in enumerate(data_out):
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
                init_cov_new = init_cov_new / 2 + init_cov_new.T / 2

                init_mean_same = np.max(np.abs(init_mean - init_mean_new)) < converge_res
                init_cov_same = np.max(np.abs(init_cov - init_cov_new)) < converge_res
                if init_mean_same and init_cov_same:
                    converged = True
                else:
                    init_mean = init_mean_new.copy()
                    init_cov = init_cov_new.copy()

                print('cpu_id', cpu_id + 1, '/', size, 'data #', ii + 1, '/', len(data_out),
                      'posterior iteration:', iter_num, ', converged:', converged)
                iter_num += 1

            emissions = i[0].copy()
            inputs = i[1].copy()

            ll, posterior = model.lgssm_smoother(emissions, inputs, init_mean, init_cov, memmap_cpu_id)[:2]
            model_sampled = model.sample(num_time=emissions.shape[0], inputs_list=[inputs], init_mean=[init_mean], init_cov=[init_cov], add_noise=False)
            model_sampled_noise = model.sample(num_time=emissions.shape[0], inputs_list=[inputs], init_mean=[init_mean], init_cov=[init_cov])

            posterior = posterior[:, :model.dynamics_dim]
            model_sampled = model_sampled['latents'][0][:, :model.dynamics_dim]
            model_sampled_noise = model_sampled_noise['latents'][0][:, :model.dynamics_dim]

            posterior_missing = None
            ll_missing = []
            if infer_missing:
                print('inferring missing neurons')

                posterior_missing = np.zeros_like(emissions)
                for n in range(emissions.shape[1]):
                    print('inferring neuron ' + str(n + 1) + '/' + str(emissions.shape[1]))

                    if np.any(~np.isnan(emissions[:, n])):
                        emissions_missing = emissions.copy()
                        emissions_missing[:, n] = np.nan
                        ll_missing_this, posterior_recon = model.lgssm_smoother(emissions_missing, inputs, init_mean, init_cov, memmap_cpu_id)[:2]

                        ll_missing.append(ll_missing_this)
                        posterior_missing[:, n] = posterior_recon[:, n]
                    else:
                        ll_missing.append(ll.copy())
                        posterior_missing[:, n] = posterior[:, n]

            ll_smeans.append((ll, posterior, model_sampled, model_sampled_noise, init_mean, init_cov, posterior_missing, ll_missing))
    else:
        ll_smeans = None

    ll_smeans = individual_gather(ll_smeans)
    # this is a hack to force blocking so some processes don't end before others
    blocking_scatter = individual_scatter(ll_smeans)

    if cpu_id == 0:
        ll_smeans = [i for i in ll_smeans if i is not None]

        ll_smeans_out = []
        for i in ll_smeans:
            for j in i:
                ll_smeans_out.append(j)

        ll_smeans = ll_smeans_out

        ll = [i[0] for i in ll_smeans]
        ll = np.sum(ll)
        smoothed_means = [i[1] for i in ll_smeans]
        model_sampled = [i[2] for i in ll_smeans]
        model_sampled_noise = [i[3] for i in ll_smeans]
        init_mean = [i[4] for i in ll_smeans]
        init_cov = [i[5] for i in ll_smeans]
        posterior_missing = [i[6] for i in ll_smeans]
        ll_missing = [i[7] for i in ll_smeans]

        inference_test = {'ll': ll,
                          'posterior': smoothed_means,
                          'model_sampled': model_sampled,
                          'model_sampled_noise': model_sampled_noise,
                          'init_mean': init_mean,
                          'init_cov': init_cov,
                          'cell_ids': model.cell_ids,
                          'posterior_missing': posterior_missing,
                          'll_missing': ll_missing,
                          }

        return inference_test

    return None


def parallel_get_ll(model, data):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()
    size = comm.Get_size()

    if cpu_id == 0:
        emissions = data['emissions']
        inputs = data['inputs']
        init_mean = data['init_mean']
        init_cov = data['init_cov']

        test_data_packaged = model.package_data_mpi(emissions, inputs, init_mean, init_cov, size)
    else:
        test_data_packaged = None

    # get posterior on test data
    model = comm.bcast(model)
    data_out = individual_scatter(test_data_packaged)

    emissions_this = [i[0] for i in data_out]
    inputs_this = [i[1] for i in data_out]
    init_mean_this = [i[2] for i in data_out]
    init_cov_this = [i[3] for i in data_out]

    ll = model.get_ll(emissions_this, inputs_this,
                      init_mean_this, init_cov_this)

    ll = individual_gather(ll)
    # this is a hack to force blocking so some processes don't end before others
    blocking_scatter = individual_scatter(ll)

    if cpu_id == 0:
        ll = [i for i in ll if i is not None]

        return np.sum(ll)

    return None


def is_pd(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

