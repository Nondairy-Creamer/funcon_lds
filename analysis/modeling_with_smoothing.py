from matplotlib import pyplot as plt
import pickle
import numpy as np
from pathlib import Path

# the goal fo this function is to determine how accurate a connectome constrained model is in the presence of
# significant extrasynaptic signaling

file_path = Path('/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/nature_review_appeal/smoothed_mismatch_data.pkl')

save_file = open(file_path, 'rb')
data = pickle.load(save_file)
save_file.close()

mask_array = [0, 10, 20, 30, 40]

num_repeat = 10
true_corr = np.array(data['true_corr']).reshape((num_repeat, len(mask_array))).T
uncon_corr = np.array(data['uncon_corr']).reshape((num_repeat, len(mask_array))).T
mismatch_corr = np.array(data['mismatch_corr']).reshape((num_repeat, len(mask_array))).T
corr_to_true = np.array(data['mis_to_true_weights_corr']).reshape((num_repeat, len(mask_array))).T

plt.figure()
# plt.plot(mask_array, true_corr)
# plt.plot(mask_array, uncon_corr)
plt.plot(mask_array, mismatch_corr)
plt.ylim([0, 1])
plt.xlabel('filter tau (time points)')
plt.ylabel('correlation between measured and model STAMs')
plt.title('model prediction of STAMs')

true_mean = np.mean(true_corr, axis=1)
uncon_mean = np.mean(uncon_corr, axis=1)
mismatch_mean = np.mean(mismatch_corr, axis=1)

true_sem = np.std(true_corr, axis=1) / np.sqrt(num_repeat)
uncon_sem = np.std(uncon_corr, axis=1) / np.sqrt(num_repeat)
mismatch_sem = np.std(mismatch_corr, axis=1) / np.sqrt(num_repeat)

plt.figure()
# plt.errorbar(mask_array, np.mean(true_corr, axis=1), yerr=true_sem, label='true model')
# plt.errorbar(mask_array, np.mean(uncon_corr, axis=1), yerr=uncon_sem, linestyle='--', label='fully-connected')
plt.errorbar(mask_array, np.mean(mismatch_corr, axis=1), yerr=mismatch_sem, label='connectome-constrained model')
plt.ylim([0, 1])
plt.xlabel('filter tau (time points)')
plt.ylabel('correlation between measured and model STAMs')
plt.title('model prediction of STAMs')
plt.legend()
plt.savefig(file_path.parent / 'smoothed_mismatch.pdf')

plt.figure()
plt.errorbar(mask_array, np.mean(corr_to_true, axis=1), yerr=mismatch_sem)
plt.ylim([0, 1.2])
plt.xlabel('filter tau (time points)')
plt.ylabel('correlation between measured and model weights')
# plt.title('model prediction of STAMs')
plt.savefig(file_path.parent / 'smoothed_mismatch_weights.pdf')

plt.show()



