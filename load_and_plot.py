import sys
import plotting
import pickle
import preprocessing as pp

model_true = None
has_ground_truth = True
if len(sys.argv) == 2:
    has_ground_truth = sys.argv[1]

if has_ground_truth:
    run_params = pp.get_params(param_name='params_synth')
else:
    run_params = pp.get_params(param_name='params')

model_trained_path = run_params['model_save_folder'] + '/model_trained.pkl'
model_trained_file = open(model_trained_path, 'rb')
model_trained = pickle.load(model_trained_file)
model_trained_file.close()

if has_ground_truth:
    dtype = model_trained.dtype
    device = model_trained.device

    model_true_path = run_params['model_save_folder'] + '/model_true.pkl'
    model_true_file = open(model_true_path, 'rb')
    model_true = pickle.load(model_true_file)
    model_true_file.close()

plotting.plot_model_params(model_trained, model_synth_true=model_true)

