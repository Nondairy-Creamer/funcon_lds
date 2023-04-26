import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_iter = 5


def some_fun(data):
    return np.sum(data**2)


if rank == 0:
    data_size = 3
    rng = np.random.default_rng(0)

    data = [rng.standard_normal(data_size), rng.standard_normal(data_size), rng.standard_normal(data_size)]
    data_ind = comm.scatter(data, root=0)

else:
    data = None

# em step loop
for s in range(num_iter):
    # for each data set
    data_ind = comm.scatter(data, root=0)
    stats = some_fun(data_ind)
    stats = comm.gather(stats, root=0)

    if rank == 0:
        data = [i+1 for i in data]
        # print('rank:', rank, 'data:', data)
        print(np.sum(stats))


