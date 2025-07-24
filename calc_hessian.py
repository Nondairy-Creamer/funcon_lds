from pathlib import Path
import analysis_utilities as au
import lgssm_utilities as lgssmu
import pickle


data_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/exp_DL1_IL45_N80_R0_synap_nf10/20240422_152517/')
# data_folder = Path('/home/mcreamer/Documents/python/funcon_lds/trained_models/syn_test/20250717_183110/')

model_file = open(data_folder / 'models' / 'model_trained.pkl', 'rb')
model = pickle.load(model_file)
model_file.close()

model = au.normalize_model(model)[0]

# load posterior
post_file = open(data_folder / 'posterior_train.pkl', 'rb')
posterior_dict = pickle.load(post_file)
post_file.close()

# load data
data_train_file = open(data_folder / 'data_train.pkl', 'rb')
data_train = pickle.load(data_train_file)
data_train_file.close()

hess_dict = {
    'emissions': data_train['emissions'],
    'inputs': data_train['inputs'],
    'emissions_offset': posterior_dict['emissions_offset'],
    'init_mean': posterior_dict['init_mean'],
    'init_cov': posterior_dict['init_cov'],
    }

lgssmu.approximate_hessian_diagonal(model, hess_dict)
