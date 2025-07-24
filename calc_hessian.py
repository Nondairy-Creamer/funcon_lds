from pathlib import Path
import analysis_utilities as au
import lgssm_utilities as lgssmu
import pickle
import torch


dtype = torch.float32
device = 'cpu'

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
    'emissions': [torch.tensor(i, dtype=dtype, device=device) for i in data_train['emissions']],
    'inputs': [torch.tensor(i, dtype=dtype, device=device) for i in data_train['inputs']],
    'emissions_offset': [torch.tensor(i, dtype=dtype, device=device) for i in posterior_dict['emissions_offset']],
    'init_mean': [torch.tensor(i, dtype=dtype, device=device) for i in posterior_dict['init_mean']],
    'init_cov': [torch.tensor(i, dtype=dtype, device=device) for i in posterior_dict['init_cov']],
    }

model.dynamics_weights = torch.tensor(model.dynamics_weights, dtype=dtype, device=device, requires_grad=True)
model.dynamics_input_weights = torch.tensor(model.dynamics_input_weights, dtype=dtype, device=device)
model.dynamics_cov = torch.tensor(model.dynamics_cov, dtype=dtype, device=device)
model.emissions_weights = torch.tensor(model.emissions_weights, dtype=dtype, device=device)
model.emissions_input_weights = torch.tensor(model.emissions_input_weights, dtype=dtype, device=device)
model.emissions_cov = torch.tensor(model.emissions_cov, dtype=dtype, device=device)

# lgssmu.approximate_hessian_diagonal(model, hess_dict)
lgssmu.calc_hessian_torch(model, hess_dict)
