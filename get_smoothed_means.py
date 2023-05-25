import sys
from pathlib import Path
import pickle
import loading_utilities as lu
import inference_utilities as iu
import os
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
cpu_id = comm.Get_rank()
num_cpus = comm.Get_size()

if len(sys.argv) == 2:
    model_folder = Path(sys.argv[1])
else:
    # search for the most recently generated model
    max_mtime = 0
    max_file = ''
    max_dir = ''
    search_dir = 'trained_models'

    for dirname, subdirs, files in os.walk(search_dir):
        for fname in files:
            full_path = os.path.join(dirname, fname)
            mtime = os.stat(full_path).st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
                max_dir = dirname
                max_file = fname

    model_folder = Path(max_dir)

model_path = model_folder / 'model_trained.pkl'
model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()

if cpu_id == 0:
    # load in the model and the data
    data_path = model_folder / 'data.pkl'

    data_file = open(data_path, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    emissions_list = data['emissions']
    inputs_list = data['inputs']
    cell_ids = data['cell_ids']

    # convert the data to tensors
    emissions, inputs = model.standardize_inputs(emissions_list, inputs_list)

    # if not provided with an initial mean / cov calculate them
    init_mean = model.estimate_init_mean(emissions)
    init_cov = model.estimate_init_cov(emissions)

    # package the data to be sent out to each of the children
    # if there is more data than chilren, create groups of data to be sent to each
    data_out = list(zip(emissions, inputs, init_mean, init_cov))
    num_data = len(emissions_list)
    chunk_size = int(np.ceil(num_data / num_cpus))
    # split data out into a list of inputs
    data_out = [data_out[i:i + chunk_size] for i in range(0, num_data, chunk_size)]
else:
    data_out = None

print('Got to cpu:', cpu_id)
data = iu.individual_scatter(data_out, root=0)
smoothed_means = []

for d in data:
    this_emission = d[0]
    this_input = d[1]
    this_init_mean = d[2]
    this_init_cov = d[3]
    smoothed_means.append(
        model.lgssm_smoother(this_emission, this_input, init_mean=this_init_mean, init_cov=this_init_cov)[3])

print('Finished cpu:', cpu_id)

smoothed_means = iu.individual_gather(smoothed_means, root=0)

if cpu_id == 0:
    print('Gathered')

    smoothed_means_out = []
    for i in smoothed_means:
        for j in i:
            smoothed_means_out.append(j)

    smoothed_means = smoothed_means_out

    lu.save_run(model_folder.parent, smoothed_means=smoothed_means)
    print('saved')
