import os
import scipy
import random as rand
from scipy import io as sio
import numpy as np
import matplotlib.pyplot as plt

import tuning_curve
import brain

# A FAIRE : plotter le signal au cours du temps, le signal pour les différents q_MAP, sigma2_q

# Define the seed to reproduce results from random processes
rand.seed(1);

# Import Matlab structure containing data and defining inputs
data_mat = sio.loadmat('data/ideal_observer_toy_example.mat', struct_as_record=False)
out = data_mat['out']
out_HMM = out[0,0].HMM

# The full distribution P(1|2) (50 values to represent it and 600 trials)
p1g2_dist = out_HMM[0,0].p1g2_dist

n_val = np.size(p1g2_dist, 0)
n_trial = np.size(p1g2_dist, 1)

# The MAP of this distribution
p1g2_map = np.zeros(n_trial)
x = np.linspace(0, 1, n_val)

for k in range(0,n_trial):
    map_idx = np.argmax(p1g2_dist[:, k])
    p1g2_map[k] = x[map_idx]

# The standard deviation of this distribution
p1g2_sd = out_HMM[0, 0].p1g2_sd
p1g2_sd = p1g2_sd[0, :]

# ## Test to check the loading has been correctly performed
#
# # Select a random trial
# trial = rand.randrange(n_trial)
# y = p1g2_dist[:, trial]
#
# #Find the closest point from the linear spacing at the MAP
# k = 0
# while (x[k]<p1g2_map[trial]):
#     k+=1
#
# plt.plot(x, y)    # Full distribution
# plt.plot([p1g2_map[trial]], [y[k]], marker='o', markersize=3, color="red")    # Single point at the MAP
# plt.text(p1g2_map[trial], y[k], 'MAP', fontsize=12, color='red')
# plt.show()


## Define the parameters of interest

# Tuning curve type
tc_type = 'gaussian'

# Number of tuning curves
N = 8

# Standard deviation of tuning curves' parameter (for gaussian : std. For sigmoid : divise number in the exponential)
t = 0.2

# Bound considered for the tuning curves. For Gaussian tuning curve, the first tuning curve is centered around the lower bound,
# whereas the last tuning curve is centered around the upper bound

# Probability here
lower_bound = 0
upper_bound = 1

# Creates the tuning_curve object
tc = tuning_curve.tuning_curve(tc_type, N, t, lower_bound, upper_bound)

# Test plotting the tuning curves
x = np.linspace(lower_bound, upper_bound)

fig = plt.figure(1)
for i in range(0, N):
    plt.plot(x, tc.f(x, i))

plt.xlabel('Preferred probability')
plt.ylabel('Tuning curve')
plt.show()

### Simulate signal from the data without noise
q_map = p1g2_map
sigma2_q = np.power(p1g2_sd, 2)

sigma2_map = np.var(q_map)    # Variance of the signal of q_map's
sigma2_sigma = np.var(sigma2_q)    # Variance of the signal of sigma2_q's

# Properties of the brain to be simulated
coding_scheme = 'rate'
n_voxel = 100
n_population = 2

simulated_brain = brain.brain(coding_scheme, n_voxel, n_population)

# Computes the signal for each voxel
signal = np.zeros(n_voxel)
for voxel in range(n_voxel):
    signal[voxel] = simulated_brain.activity(voxel, q_map[0], sigma2_q[0], sigma2_map, sigma2_sigma)

x = np.linspace(0, n_voxel-1, n_voxel)

fig = plt.figure(2)
plt.plot(x, signal)
plt.xlabel('Voxel index')
plt.ylabel('Signal intensity')
plt.show()

# x = np.linspace(0, n_trial-1, n_trial)
#
# fig = plt.figure(3)
# plt.plot(x, q_map)
# plt.plot(x, sigma2_q)
# # plt.xlabel('Voxel index')
# # plt.ylabel('Signal intensity')
# plt.show()
