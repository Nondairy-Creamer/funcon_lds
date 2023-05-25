import sys
from pathlib import Path
import pickle
import loading_utilities as lu
import inference_utilities as iu
import os


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

# load in the model and the data
model_path = model_folder / 'model_trained.pkl'
data_path = model_folder / 'data.pkl'

model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()

data_file = open(data_path, 'rb')
data = pickle.load(data_file)
data_file.close()

emissions = data['emissions']
inputs = data['inputs']
cell_ids = data['cell_ids']

smoothed_means = iu.parallel_smoother(model, emissions, inputs)

lu.save_run(model_folder.parent, smoothed_means=smoothed_means)

