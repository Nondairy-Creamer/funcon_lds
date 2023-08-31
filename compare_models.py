from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt

data_sets_location = Path('/home/mcreamer/Documents/trained_models/no_ridge_full_emissions/')
sorting_param = 'dynamics_lags'
data = []

for p in sorted(data_sets_location.rglob('data_train.pkl')):
    data_folder = p.parent

    # get the last trained file
    model_full_path = p.parent / 'models' / 'model_trained_-1.pkl'
    if not model_full_path.exists():
        all_model_paths = sorted(model_full_path.parent.glob('model_trained*'))
        model_path_number = [str(i.name)[14:-4] for i in all_model_paths]
        current_model_ind = model_path_number.index('')
        all_model_paths.pop(current_model_ind)
        model_path_number.pop(current_model_ind)
        model_path_number = [int(i) for i in model_path_number]
        model_index = np.argmax(model_path_number)
        model_full_path = all_model_paths[model_index]

    data.append({})

    data[-1]['path'] = data_folder

    # First get the final model
    model_full_file = open(model_full_path, 'rb')
    data[-1]['model_full'] = pickle.load(model_full_file)
    model_full_file.close()

    # now get the model at its highest likelihood
    model_path = data_folder / 'models' / 'model_trained.pkl'
    model_file = open(model_path, 'rb')
    data[-1]['model'] = pickle.load(model_file)
    model_file.close()

    post_train_path = data_folder / 'posterior_train.pkl'
    post_train_file = open(post_train_path, 'rb')
    data[-1]['post_train'] = pickle.load(post_train_file)
    post_train_file.close()

    post_test_path = data_folder / 'posterior_test.pkl'
    post_test_file = open(post_test_path, 'rb')
    data[-1]['post_test'] = pickle.load(post_test_file)
    post_test_file.close()

    data_train_path = data_folder / 'data_train.pkl'
    data_train_file = open(data_train_path, 'rb')
    data[-1]['data_train'] = pickle.load(data_train_file)
    data_train_file.close()

    data_test_path = data_folder / 'data_test.pkl'
    data_test_file = open(data_test_path, 'rb')
    data[-1]['data_test'] = pickle.load(data_test_file)
    data_test_file.close()

    data[-1]['sorting_id'] = getattr(data[-1]['model'], sorting_param)

    # normalize the log likelihood by the number of non nan values in the data for visualization
    num_data_points_train = np.sum([np.sum(~np.isnan(i)) for i in data[-1]['data_train']['emissions']])
    num_data_points_test = np.sum([np.sum(~np.isnan(i)) for i in data[-1]['data_test']['emissions']])
    data[-1]['model'].log_likelihood = data[-1]['model'].log_likelihood / num_data_points_train
    data[-1]['model_full'].log_likelihood = data[-1]['model_full'].log_likelihood / num_data_points_train
    data[-1]['post_train']['ll'] = data[-1]['post_train']['ll'] / num_data_points_train
    data[-1]['post_test']['ll'] = data[-1]['post_test']['ll'] / num_data_points_test

sorting_list = [i['sorting_id'] for i in data]
data = [data[i] for i in np.argsort(sorting_list)]

for d in data:
    plt.figure()
    plt.plot(d['model_full'].log_likelihood)
    plt.plot(d['model'].log_likelihood)
    plt.title(sorting_param + ': ' + str(d['sorting_id']))
    plt.xlabel('Iterations of EM')
    plt.ylabel('mean log likelihood')

train_ll = [i['post_train']['ll'] for i in data]
test_ll = [i['post_test']['ll'] for i in data]

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(sorting_list, train_ll, label='train')
plt.xlabel(sorting_param)
plt.ylabel('mean log likelihood')
plt.subplot(1, 2, 2)
plt.plot(sorting_list, test_ll, label='test')
plt.xlabel(sorting_param)
plt.tight_layout()

plt.show()

