import analysis_utilities as au
from pathlib import Path
import numpy as np
import metrics as met
from matplotlib import pyplot as plt
import wormneuroatlas as wa
import pickle

connectome_folder = Path('/home/mcreamer/Documents/data_sets/connectomes/')

saved_run_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models')
data_folder = Path('exp_DL1_IL45_N80_R0_synap_nf10/20240422_152517')
data_test_file = open(saved_run_folder / data_folder / 'data_test.pkl', 'rb')
data_test = pickle.load(data_test_file)
data_test_file.close()

cell_ids = data_test['cell_ids']
watlas = wa.NeuroAtlas()
cell_ids = list(watlas.neuron_ids)
cell_ids[cell_ids.index('AWCON')] = 'AWCR'
cell_ids[cell_ids.index('AWCOFF')] = 'AWCL'
cell_ids.remove('VD9')
cell_ids.remove('AWCL')
cell_ids.remove('AWCR')


# code to process the files from witvliet into matricies
def process_witvliet(file_name):
    # get the number of columns in the witvliet data
    with open(connectome_folder / file_name) as f:
        header = f.readline()
        num_cols = len(header.strip().split(","))

    row_cell_ids = header.strip().split(",")[1:]

    row_cell_ids = list(np.loadtxt(connectome_folder / file_name, delimiter=',', max_rows=1, dtype=str))
    num_cols = len(row_cell_ids)
    row_cell_ids = row_cell_ids[1:]
    col_cell_ids = list(np.loadtxt(connectome_folder / file_name, delimiter=',', usecols=0, dtype=str))
    col_cell_ids = col_cell_ids[2:]

    connectome_in = np.loadtxt(connectome_folder / file_name, delimiter=",", skiprows=2, usecols=range(1, num_cols))

    connectome = np.zeros((len(row_cell_ids), len(row_cell_ids)))
    for rci_index, rci in enumerate(row_cell_ids):
        cci_index = col_cell_ids.index(rci)
        connectome[rci_index, :] = connectome_in[cci_index, :]

    # get rid of some neurons
    neurons_to_remove = ['AWCL', 'AWCR']

    for n in neurons_to_remove:
        neuron_index = row_cell_ids.index(n)
        connectome = np.delete(connectome, neuron_index, axis=0)
        connectome = np.delete(connectome, neuron_index, axis=1)
        row_cell_ids.pop(neuron_index)

    return connectome, row_cell_ids


def process_nemanode(file_name, cell_ids=None, delimiter='\t'):
    pre_cell_ids = list(np.loadtxt(connectome_folder / file_name, delimiter=delimiter, usecols=0, dtype=str))[1:]
    post_cell_ids = list(np.loadtxt(connectome_folder / file_name, delimiter=delimiter, usecols=1, dtype=str))[1:]
    synapse_type = list(np.loadtxt(connectome_folder / file_name, delimiter=delimiter, usecols=2, dtype=str))[1:]
    synapses = np.loadtxt(connectome_folder / file_name, skiprows=1, delimiter=delimiter, usecols=3)
    num_connections = len(pre_cell_ids)

    connectome = np.zeros((len(cell_ids), len(cell_ids)))
    for i in range(num_connections):
        if pre_cell_ids[i] in cell_ids and post_cell_ids[i] in cell_ids:
            # if synapse_type[i] == 'electrical':
            #     continue

            pre_index = cell_ids.index(pre_cell_ids[i])
            post_index = cell_ids.index(post_cell_ids[i])
            connectome[post_index, pre_index] += synapses[i]

            # if electrical, symmetrize
            if synapse_type[i] == 'electrical':
                connectome[pre_index, post_index] += synapses[i]

    return connectome


# code to process durbin into matricies
def process_durbin(file_name, cell_ids=None):
    cell_ids_1 = list(np.loadtxt(connectome_folder / file_name, delimiter=',', usecols=0, dtype=str))
    cell_ids_2 = list(np.loadtxt(connectome_folder / file_name, delimiter=',', usecols=1, dtype=str))
    synapse_type = list(np.loadtxt(connectome_folder / file_name, delimiter=',', usecols=2, dtype=str))
    dataset = list(np.loadtxt(connectome_folder / file_name, delimiter=',', usecols=3, dtype=str))

    synapses = np.loadtxt(connectome_folder / file_name, skiprows=0, delimiter=',', usecols=4)
    num_connections = len(cell_ids_1)

    connectome = {'whole': np.zeros((len(cell_ids), len(cell_ids))),
                  'l4': np.zeros((len(cell_ids), len(cell_ids))),
                  }
    for i in range(num_connections):
        if cell_ids_1[i] in cell_ids and cell_ids_2[i] in cell_ids:
            dataset_tag = 'whole' if dataset[i] == 'N2U' else 'l4'
            index_1 = cell_ids.index(cell_ids_1[i])
            index_2 = cell_ids.index(cell_ids_2[i])

            if 'Send' in synapse_type[i]:
                connectome[dataset_tag][index_2, index_1] += synapses[i]
            elif 'Receive' in synapse_type[i]:
                connectome[dataset_tag][index_1, index_2] += synapses[i]
            else:
                connectome[dataset_tag][index_2, index_1] += synapses[i]
                connectome[dataset_tag][index_1, index_2] += synapses[i]

    return connectome['whole'], connectome['l4']


# read in the files
# witvliet_7_2, cell_ids = process_witvliet('witvliet_7.csv')
# witvliet_8_2 = process_witvliet('witvliet_8.csv')[0]
# white_l4 = process_nemanode('aconnectome_white_1986_L4.csv', cell_ids)
# white_adult = process_nemanode('aconnectome_white_1986_A.csv', cell_ids)
# white_whole = process_nemanode('aconnectome_white_1986_whole.csv', cell_ids)
# durbin_whole, durbin_l4 = process_durbin('durbin_revised.csv', cell_ids)

witvliet_7 = process_nemanode('nemanode/witvliet_2020_7.csv', cell_ids)
witvliet_8 = process_nemanode('nemanode/witvliet_2020_8.csv', cell_ids)
white_whole = process_nemanode('nemanode/white_1986_whole.csv', cell_ids)
white_l4 = process_nemanode('nemanode/white_1986_jsh.csv', cell_ids)

# get the anatomy used in the paper
anatomy_in_paper = au.load_anatomical_data(cell_ids)
# my_connectome = anatomy_in_paper['chem_conn'] + anatomy_in_paper['gap_conn']

# get anatomy straight from neuroatlas
# frandi_chem, frandi_gap = watlas.get_aconnectome_from_file(chem_th=0, gap_th=0, exclude_white=False, average=False)
frandi_chem, frandi_gap, frandi_pep = au.get_anatomical_data(cell_ids)
my_connectome = frandi_chem + frandi_gap

reconstructed_connectome = witvliet_7 + 0*witvliet_8 + 0*white_whole + 0*white_l4
# reconstructed_connectome = 0*witvliet_7 + 0*witvliet_8 + durbin_whole + 0*durbin_l4

# my_connectome[my_connectome > 1] = 1
# reconstructed_connectome[reconstructed_connectome > 1] = 1

my_connectome_synapses = my_connectome[my_connectome > 0]
reconstructed_connectome_synapses = reconstructed_connectome[reconstructed_connectome > 0]

similarity = met.nan_corr(my_connectome, reconstructed_connectome)[0]
print(similarity)


# plt.figure()
# plt.scatter(my_connectome_synapses, reconstructed_connectome_synapses)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(my_connectome, interpolation='nearest')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_connectome, interpolation='nearest')
plt.colorbar()

plt.figure()
plt.imshow(my_connectome - reconstructed_connectome, interpolation='nearest')
plt.colorbar()

plt.show()
a=1