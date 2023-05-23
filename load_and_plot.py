import sys
import plotting
import pickle
import os
from pathlib import Path


model_true = None
has_ground_truth = False

if len(sys.argv) == 2:
    model_path = Path(sys.argv[1])
    model_folder = model_path.parent

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

model_trained_path = model_folder / 'model_trained.pkl'
model_trained_file = open(model_trained_path, 'rb')
model_trained = pickle.load(model_trained_file)
model_trained_file.close()

# check if a true model exists
model_true_path = model_folder / 'model_true.pkl'

if model_true_path.exists():
    dtype = model_trained.dtype
    device = model_trained.device

    model_true_file = open(model_true_path, 'rb')
    model_true = pickle.load(model_true_file)
    model_true_file.close()

plotting.plot_model_params(model_trained, model_true=model_true)

