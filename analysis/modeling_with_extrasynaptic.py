from matplotlib import pyplot as plt
import pickle
import numpy as np
from pathlib import Path

# the goal fo this function is to determine how accurate a connectome constrained model is in the presence of
# significant extrasynaptic signaling

file_path = Path('/home/mcreamer/Documents/google_drive/leifer_pillow_lab/papers/2023_lds/figures/drafts_subpannels/nature_review_appeal/mismatch_data.pkl')

save_file = open(file_path, 'rb')
data = pickle.load(save_file)
save_file.close()

mask_array = [0.1, 0.125, 0.15, 0.175, 0.2, 0.4]
mask_array = (np.array(mask_array) - 0.1) / np.array(mask_array) * 100

num_repeat = 10
true_corr = np.array(data['true_corr']).reshape((num_repeat, len(mask_array))).T
uncon_corr = np.array(data['uncon_corr']).reshape((num_repeat, len(mask_array))).T
mismatch_corr = np.array(data['mismatch_corr']).reshape((num_repeat, len(mask_array))).T

plt.figure()
# plt.plot(mask_array, true_corr)
# plt.plot(mask_array, uncon_corr)
plt.plot(mask_array, mismatch_corr)
plt.ylim([0, 1])
plt.xlabel('percent of connections that are extrasynaptic')
plt.ylabel('correlation')
plt.title('model prediction of STAMs')

true_mean = np.mean(true_corr, axis=1)
uncon_mean = np.mean(uncon_corr, axis=1)
mismatch_mean = np.mean(mismatch_corr, axis=1)

true_sem = np.std(true_corr, axis=1) / np.sqrt(num_repeat)
uncon_sem = np.std(uncon_corr, axis=1) / np.sqrt(num_repeat)
mismatch_sem = np.std(mismatch_corr, axis=1) / np.sqrt(num_repeat)

plt.figure()
plt.errorbar(mask_array, np.mean(true_corr, axis=1), yerr=true_sem, label='true model')
plt.errorbar(mask_array, np.mean(uncon_corr, axis=1), yerr=uncon_sem, linestyle='--', label='unconstrained')
plt.errorbar(mask_array, np.mean(mismatch_corr, axis=1), yerr=mismatch_sem, label='connectome-constrained')
plt.ylim([0, 1])
plt.xlabel('percent of connections that are extrasynaptic')
plt.ylabel('correlation')
plt.title('model prediction of STAMs')
plt.legend()
plt.savefig(file_path.parent / 'mismatch.pdf')

plt.show()



