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


def process_nemanode(file_name, cell_ids, included_types=['chem', 'gap'], delimiter='\t'):
    pre_cell_ids = list(np.loadtxt(connectome_folder / file_name, delimiter=delimiter, usecols=0, dtype=str))[1:]
    post_cell_ids = list(np.loadtxt(connectome_folder / file_name, delimiter=delimiter, usecols=1, dtype=str))[1:]
    synapse_type = list(np.loadtxt(connectome_folder / file_name, delimiter=delimiter, usecols=2, dtype=str))[1:]
    synapses = np.loadtxt(connectome_folder / file_name, skiprows=1, delimiter=delimiter, usecols=3)
    num_connections = len(pre_cell_ids)

    connectome = np.zeros((len(cell_ids), len(cell_ids)))
    for i in range(num_connections):
        # check if both cells are in the list of cell ids
        if pre_cell_ids[i] in cell_ids and post_cell_ids[i] in cell_ids:
            pre_index = cell_ids.index(pre_cell_ids[i])
            post_index = cell_ids.index(post_cell_ids[i])

            if synapse_type[i] == 'electrical':
                if 'gap' in included_types:
                    # add the post to pre electrical synapse
                    connectome[post_index, pre_index] += synapses[i]

                    # if the neurons are different add in
                    if pre_index != post_index:
                        connectome[pre_index, post_index] += synapses[i]
            else:
                if 'chem' in included_types:
                    # add in the pre to post chemical synapse
                    connectome[post_index, pre_index] += synapses[i]

    return connectome

binarize_connectome = True
included_types = ['gap']
# cell_ids = data_test['cell_ids']
watlas = wa.NeuroAtlas()
cell_ids = list(watlas.neuron_ids)
cell_ids[cell_ids.index('AWCON')] = 'AWCR'
cell_ids[cell_ids.index('AWCOFF')] = 'AWCL'
cell_ids.remove('VD9')
cell_ids.remove('AWCL')
cell_ids.remove('AWCR')

witvliet_7 = process_nemanode('nemanode/witvliet_2020_7.csv', cell_ids, included_types=included_types)
witvliet_8 = process_nemanode('nemanode/witvliet_2020_8.csv', cell_ids, included_types=included_types)
white_whole = process_nemanode('nemanode/white_1986_whole.csv', cell_ids, included_types=included_types)
white_l4 = process_nemanode('nemanode/white_1986_jsh.csv', cell_ids, included_types=included_types)

# get the anatomy used in the paper
# anatomy_in_paper = au.load_anatomical_data(cell_ids)
# my_connectome = anatomy_in_paper['chem_conn'] + anatomy_in_paper['gap_conn']

# get anatomy straight from neuroatlas
# frandi_chem, frandi_gap = watlas.get_aconnectome_from_file(chem_th=0, gap_th=0, exclude_white=False, average=False)
frandi_chem, frandi_gap, frandi_pep = au.get_anatomical_data(cell_ids)
my_connectome = np.zeros_like(frandi_chem)

for s in included_types:
    if s == 'chem':
        my_connectome += frandi_chem
    if s == 'gap':
        my_connectome += frandi_gap

reconstructed_connectome = witvliet_7 + 0*witvliet_8 + 0*white_whole + 0*white_l4

for i in [False, True]:
    if i:
        my_connectome[my_connectome > 1] = 1
        reconstructed_connectome[reconstructed_connectome > 1] = 1


    print(np.sum((np.abs(my_connectome - reconstructed_connectome))[np.triu_indices(my_connectome.shape[0])]))

similarity = met.nan_corr(my_connectome, reconstructed_connectome)[0]
print(similarity)

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