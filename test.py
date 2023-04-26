import numpy as np
from matplotlib import pyplot as plt
import time
import utilities as utils


seed = None
rng = np.random.default_rng(seed)

num_dim = 120
A = rng.standard_normal((num_dim, num_dim))
B = rng.standard_normal((num_dim, num_dim))
C = rng.standard_normal((num_dim, num_dim))

x = rng.standard_normal((num_dim, num_dim))
d_diag = rng.standard_normal(num_dim)
d = np.diag(d_diag)

a_c_ct_b = np.block([[A, C], [C.T, B]])
x_d = np.block([[x], [d]])

y_z = a_c_ct_b @ x_d + rng.standard_normal((num_dim * 2, num_dim))
y = y_z[:num_dim, :]
z = y_z[num_dim:, :]

start = time.time()
x_d_hat = np.linalg.solve(a_c_ct_b, y_z)
x_d_hat[num_dim:, :] = np.diag(np.diag(x_d_hat[num_dim:, :]))
print('full method took:', time.time() - start, 's')

start = time.time()
x_d_hat_prime = utils.solve_half_diag_np(a_c_ct_b, y_z)
print('diag_method took:', time.time() - start, 's')

start = time.time()
x = np.linalg.solve(A, y)
d_diag_hat = np.zeros_like(d_diag)
# attempt at diagonal fitting
for i in range(num_dim):
    d_diag_hat[i] = np.linalg.lstsq(B[:, i, None], z[:, i], rcond=None)[0]

x_d_hat_split = np.block([[x], [np.diag(d_diag_hat)]])
print('half_method took:', time.time() - start, 's')

plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(x_d)
plt.subplot(1, 4, 2)
plt.imshow(x_d_hat)
plt.subplot(1, 4, 3)
plt.imshow(x_d_hat_prime)
plt.subplot(1, 4, 4)
plt.imshow(x_d_hat_split)

plt.show()





