import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import distrib
import tuning_curve
import voxel

# Define the seed to reproduce results from random processes
rand.seed(1);

# Import Matlab structure containing data and defining inputs
data_mat = sio.loadmat('data/ideal_observer_toy_example.mat', struct_as_record=False)
out = data_mat['out']
out_HMM = out[0,0].HMM

# The full distribution P(1|2) (resolution of 50 values and 600 trials)
p1g2_dist = out_HMM[0,0].p1g2_dist

n_val = np.size(p1g2_dist, 0)
n_trial = np.size(p1g2_dist, 1)

# The MAP of this distribution
p1g2_map = np.zeros(n_trial)
x = np.linspace(0, 1, n_val)

for k in range(0, n_trial):
    map_idx = np.argmax(p1g2_dist[:, k])
    p1g2_map[k] = x[map_idx]

# The standard deviation of this distribution
p1g2_sd = out_HMM[0, 0].p1g2_sd
p1g2_sd = p1g2_sd[0, :]

# PLOTTER STANDARD DEVIATION ! ! Pour tester

# ## Test to check the loading and MAP/std calculations have been correctly performed
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

### Simulate signal from the data without noise

# # Data of interest
# q_map = p1g2_map
# sigma_q = p1g2_sd   # We will consider the std only

# Generate data from beta distributions samples from means and std
n_moment = 100    # Number of generated moment (i.e. number of generated mean and number of generated sd)
q_mean = np.linspace(0.1, 0.9, n_moment)
sigma_q = np.linspace(np.mean(p1g2_sd)-np.std(p1g2_sd), np.mean(p1g2_sd)+np.std(p1g2_sd), n_moment)

# Creation of a list of simulated distributions
simulated_distrib = [[None for i in range(n_moment)] for j in range(n_moment)]

for k_mean in range(n_moment):
    for k_sigma in range(n_moment):
        simulated_distrib[k_mean][k_sigma] = distrib.distrib(q_mean[k_mean], sigma_q[k_sigma])

# # Plots the distribution
# fig = plt.figure()
# k_mean = 3
# k_sigma = 4
# y = simulated_distrib[k_mean][k_sigma].beta(x)
# plt.plot(x, y)    # Full distribution
# plt.show()


# We find the variance of the data in order to scale equally activity from mean and activity from uncertainty
q_mean_sd = np.std(q_mean)    # Variance of the signal of q_mean's
sigma_q_sd = np.std(sigma_q)    # Variance of the signal of sigma_q's

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mean = 0
tc_upper_bound_mean = 1
tc_lower_bound_sigma = 0
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_sigma = np.max(sigma_q)+2*sigma_q_sd

# Resolution of the continuous plots
plot_resolution = n_moment    # we consider the same resolution as for DPC

# we define the x-axis for varying mean
x_mean = np.linspace(np.min(q_mean), np.max(q_mean), plot_resolution)
# we define the x-axis for varying uncertainty
x_sigma = np.linspace(np.min(sigma_q), np.max(sigma_q), plot_resolution)


### 1) Rate coding simulation


# Properties of the "rate voxel" to be simulated
coding_scheme = 'rate'
n_population = 2    # one for the mean, one for the std
n_subpopulation = 1    # by definition of rate coding


# Creation of the "rate voxel"
rate_voxel = voxel.voxel(coding_scheme, n_population, n_subpopulation)

# Computes the signal
rate_signal = np.zeros([n_moment, n_moment])
for k_mean in range(n_moment):    # For each mean
    for k_sigma in range(n_moment):    # For each sd
        rate_signal[k_mean, k_sigma] = rate_voxel.activity(simulated_distrib[k_mean][k_sigma], q_mean_sd, sigma_q_sd)

### PPC simulation

# TC related to the mean

tc_type_mean = 'gaussian'    # Tuning curve type
N_mean = 10    # Number of tuning curves
t_mean = 0.08    # The best value from the previous "sum" analysis

# Creates the tuning_curve object
tc_mean = tuning_curve.tuning_curve(tc_type_mean, N_mean, t_mean, tc_lower_bound_mean, tc_upper_bound_mean)

# TC related to the uncertainty

tc_type_sigma = 'gaussian'    # Tuning curve type
N_sigma = 10    # Number of tuning curves
t_sigma = 0.02    # The best value from the previous "sum" analysis

# Creates the tuning_curve object
tc_sigma = tuning_curve.tuning_curve(tc_type_sigma, N_sigma, t_sigma, tc_lower_bound_sigma, tc_upper_bound_sigma)

# Properties of the voxel to be simulated
coding_scheme = 'ppc'
n_population = 2    # one mean and one std
n_subpopulation = tc_mean.N    # here the mean and the std's tuning curves shall have the same N

ppc_voxel = voxel.voxel(coding_scheme, n_population, n_subpopulation, [tc_mean, tc_sigma])

# Computes the signal
ppc_signal = np.zeros([n_moment, n_moment])
for k_mean in range(n_moment):
    for k_sigma in range(n_moment):
        ppc_signal[k_mean, k_sigma] = ppc_voxel.activity(simulated_distrib[k_mean][k_sigma])

### DPC simulation

# Properties of the voxel to be simulated
coding_scheme = 'dpc'
n_population = 1    # population of tuning curves to be used for projection
n_subpopulation = tc_mean.N

dpc_voxel = voxel.voxel(coding_scheme, n_population, n_subpopulation, [tc_mean])    # we only input the tuning curve of the mean

# Computes the signal
dpc_signal = np.zeros([n_moment, n_moment])
for k_mean in range(n_moment):
    for k_sigma in range(n_moment):
        dpc_signal[k_mean, k_sigma] = dpc_voxel.activity(simulated_distrib[k_mean][k_sigma])

### PLOTS

# Plot the signal across voxels

# The mean and sd we want
k_mean = 1
k_sigma = 1

# Plot the signal for different voxel types and different distributions
k_jump = 1    # To skip some curves

fig = plt.figure()
plt.subplot(321)
for k_sigma in range(0,n_moment,k_jump):
    plt.plot(x_mean, rate_signal[:, k_sigma], label='sigma_q='+str(round(sigma_q[k_sigma],2)))
plt.xlabel('Inferred probability')
plt.ylabel('Signal intensity')
plt.title('Rate coding signal')
plt.legend()

plt.subplot(322)
for k_mean in range(0,n_moment,k_jump):
    plt.plot(x_sigma, rate_signal[k_mean, :], label='q_mean='+str(round(q_mean[k_mean],2)))
plt.xlabel('Inferred standard deviation')
plt.ylabel('Signal intensity')
plt.title('Rate coding signal')
plt.legend()

plt.subplot(323)
for k_sigma in range(0,n_moment,k_jump):
    plt.plot(x_mean, ppc_signal[:, k_sigma], label='sigma_q='+str(round(sigma_q[k_sigma],2)))
plt.xlabel('Inferred probability')
plt.ylabel('Signal intensity')
plt.title('PPC signal')
plt.legend()

plt.subplot(324)
for k_mean in range(0,n_moment,k_jump):
    plt.plot(x_sigma, ppc_signal[k_mean, :], label='q_mean='+str(round(q_mean[k_mean],2)))
plt.xlabel('Inferred standard deviation')
plt.ylabel('Signal intensity')
plt.title('PPC signal')
plt.legend()

plt.subplot(325)
for k_sigma in range(0,n_moment,k_jump):
    plt.plot(x_mean, dpc_signal[:, k_sigma], label='sigma_q='+str(round(sigma_q[k_sigma],2)))
plt.xlabel('Inferred probability')
plt.ylabel('Signal intensity')
plt.title('DPC signal')
plt.legend()

plt.subplot(326)
for k_mean in range(0,n_moment,k_jump):
    plt.plot(x_sigma, dpc_signal[voxel, k_mean, :], label='q_mean='+str(round(q_mean[k_mean],2)))
plt.xlabel('Inferred standard deviation')
plt.ylabel('Signal intensity')
plt.title('DPC signal')
plt.legend()
plt.show()


#############

# # Plot the optimal tuning curves
#
# fig = plt.figure()
# x = np.linspace(tc_lower_bound_mean,tc_upper_bound_mean,1000)
# plt.subplot(211)
# for i in range(0, N_mean):
#     plt.plot(x, tc_mean.f(x, i))
#
# plt.xlabel('Preferred probability')
# plt.ylabel('Tuning curve value')
# plt.title('Optimal tuning curves for encoding the mean (N='+str(tc_mean.N)+')')
#
# plt.subplot(212)
# x = np.linspace(tc_lower_bound_sigma,tc_upper_bound_sigma,1000)
# for i in range(0, N_sigma):
#     plt.plot(x, tc_sigma.f(x, i))
#
# plt.xlabel('Preferred standard deviation')
# plt.ylabel('Tuning curve value')
# plt.title('Optimal tuning curves for encoding the uncertainty (N='+str(tc_sigma.N)+')')
# plt.show()
#
#
# # Find the optimal tuning curves' standard deviation for fixed N : we consider the lowest std such that the sum over the TC is constant
#
# tested_t = [0.01,0.03,0.05,0.08,0.1,0.2,0.5]    # The different std
# x = np.linspace(tc_lower_bound_mean, tc_upper_bound_mean, 1000)
# tc_sum = np.zeros([len(tested_t), 1000])    # Initialization of the different TC sum's
#
# for k_t in range(len(tested_t)):
#     tc = tuning_curve.tuning_curve(tc_type_mean, N_mean, tested_t[k_t], tc_lower_bound_mean, tc_upper_bound_mean)
#     for i in range(N_mean):
#         tc_sum[k_t, :] += tc.f(x, i)
#
# fig = plt.figure()
# plt.subplot(211)
# for k_t in range(len(tested_t)):
#     plt.plot(x, tc_sum[k_t, :], label='t='+str(tested_t[k_t]))
#
# plt.xlabel('Probability')
# plt.ylabel('Sum over the tuning curves')
# plt.legend()
#
# tested_t = [0.005,0.01,0.02,0.03,0.05,0.08]    # The different std
# x = np.linspace(tc_lower_bound_sigma, tc_upper_bound_sigma, 1000)
# tc_sum = np.zeros([len(tested_t), 1000])
#
# for k_t in range(len(tested_t)):
#     tc = tuning_curve.tuning_curve(tc_type_sigma, N_sigma, tested_t[k_t], tc_lower_bound_sigma, tc_upper_bound_sigma)
#     for i in range(N_sigma):
#         tc_sum[k_t, :] += tc.f(x, i)
#
# plt.subplot(212)
# for k_t in range(len(tested_t)):
#     plt.plot(x, tc_sum[k_t, :], label='t='+str(tested_t[k_t]))
#
# plt.xlabel('Probability')
# plt.ylabel('Sum over the tuning curves')
# plt.legend()
# plt.show()