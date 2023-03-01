import sys
import plotting
import pickle

model_path = sys.argv[1]

model_file = open(model_path, 'rb')
model_trained = pickle.load(model_file)

plotting.trained_on_real(model_trained)
