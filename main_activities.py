import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
import numpy as np
#import decimal
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import neural_proba
from neural_proba import distrib
from neural_proba import tuning_curve
from neural_proba import voxel
from neural_proba import experiment
from neural_proba import fmri


# Define the seed to reproduce results from random processes
rand.seed(5);

# In order to plot the full range of distribution with 2D grid of mean and sd (True)
# or the distributions of the sequence only (False)
plot_all_range = True

#########################################

# Import Matlab structure containing data and defining inputs
#data_mat = sio.loadmat('data/ideal_observer_1.mat', struct_as_record=False)

[p1g2_dist, p1g2_mu, p1g2_sd] = neural_proba.import_distrib_param(1, 1, 380, 'HMM')

p1g2_dist = p1g2_dist[0]
p1g2_mu = p1g2_mu[0]
p1g2_sd = p1g2_sd[0]

### Generate equivalent beta distribution

# DISTRIBUTION SIMULATED IN ORDER TO PLOT THE SHAPE OF THE SIGNAL

# Generate data from beta distributions samples from means and std
n_mu = 30    # Number of generated means
n_sigma = 30    # Number of generated standard deviations
mu = np.linspace(0.4, 0.6, n_mu)
sigma = np.linspace(0.076, 0.35, n_sigma)    # We don't go to the max to have bell shaped beta

# Creation of a list of simulated distributions
simulated_distrib = [[None for j in range(n_sigma)] for i in range(n_mu)]

for k_mu in range(n_mu):
    for k_sigma in range(n_sigma):
        simulated_distrib[k_mu][k_sigma] = distrib(mu[k_mu], sigma[k_sigma])

# # Plots the distribution
# fig = plt.figure()
# k_mu = -1
# k_sigma = -1
# x = np.linspace(0, 1, 100)
# y = simulated_distrib[k_mu][k_sigma].beta(x)
# plt.plot(x, y)    # Full distribution
# plt.show()

# Resolution of the continuous plots
plot_resolution = n_mu    # we consider the same resolution as for DPC

# we define the x-axis for varying mean
x_mu = np.linspace(np.min(mu), np.max(mu), plot_resolution)
# we define the x-axis for varying uncertainty
x_sigma = np.linspace(np.min(sigma), np.max(sigma), plot_resolution)

# # We remove the infinite values
# inf_indices = [0, 1, 205, 206, 207, 208, 299]
# p1g2_mu = np.delete(p1g2_mu, inf_indices)
# p1g2_sd = np.delete(p1g2_sd, inf_indices)
# p1g2_dist = np.delete(p1g2_dist, inf_indices, 1)

#
# # Plots the distribution
# fig = plt.figure()
# x = np.linspace(0, 1, 50)
# y = p1g2_dist[:, 299]
# plt.plot(x, y)    # Full distribution
# plt.show()


# We find the variance of the data in order to scale equally activity from mean and activity from uncertainty
mu_sd = 0.2    # Variance of the signal of mu's
sigma_sd = 0.043    # Variance of the signal of sigma's

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mu = 0
tc_upper_bound_mu = 1
tc_lower_bound_sigma = 0.04
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_sigma = 0.35

### 1) Rate coding simulation

# Properties of the "rate voxel" to be simulated
coding_scheme = 'rate'
population_fraction = [0.5, 0.5]    # one for the mean, one for the std

# Creation of the "rate voxel"
rate_voxel = voxel(coding_scheme, population_fraction)

# Computes the signal
rate_activity = rate_voxel.generate_activity(simulated_distrib, mu_sd, sigma_sd)

### 2) PPC simulation

# Properties of the voxel to be simulated
coding_scheme = 'ppc'
population_fraction = [0.5, 0.5]    # Population fraction (one mean, one std)

# TC related to the mean
tc_type_mu = 'gaussian'    # Tuning curve type
N_mu = 10    # Number of tuning curves
t_mu = 0.05    # The best value from the previous "sum" analysis
# Creates the tuning_curve object
tc_mu = tuning_curve(tc_type_mu, N_mu, t_mu, tc_lower_bound_mu, tc_upper_bound_mu)

# TC related to the uncertainty
tc_type_sigma = 'gaussian'    # Tuning curve type
N_sigma = 10    # Number of tuning curves
t_sigma = 2e-2    # The best value from the previous "sum" analysis
# Creates the tuning_curve object
tc_sigma = tuning_curve(tc_type_sigma, N_sigma, t_sigma, tc_lower_bound_sigma, tc_upper_bound_sigma)

# Creation of the "ppc voxel"
ppc_voxel = voxel(coding_scheme, population_fraction, [tc_mu, tc_sigma])

# Computes the signal
ppc_activity = ppc_voxel.generate_activity(simulated_distrib)

### 3) DPC simulation

# Properties of the voxel to be simulated
coding_scheme = 'dpc'
population_fraction = [1]    # One population : related to tuning curves of the mean

# Creation of the "dpc voxel"
dpc_voxel = voxel(coding_scheme, population_fraction, [tc_mu])    # we only input the tuning curve of the mean

# We impose the same neuron fraction for DPC and PPC, for comparison
dpc_voxel.subpopulation_fraction = np.expand_dims(ppc_voxel.subpopulation_fraction[0, :], axis=0)
dpc_voxel.weights = np.expand_dims(ppc_voxel.weights[0, :], axis=0)

# Computes the signal
dpc_activity = dpc_voxel.generate_activity(simulated_distrib, use_high_integration_resolution=False)

### PLOTS

# Plot the signal for different voxel types and different distributions
k_jump = 6    # To skip some curves

#### BEGIN INDIVIDUAL ACTIVITIES IPLOTS ###

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_sigma in range(0,n_sigma,k_jump):
#     ax.plot(x_mu, rate_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
# ax.set_xlabel('Probability', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_ylabel('Neural activity', fontsize=15)
# ax.set_title('Rate',fontsize=20)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/rate_mu.png', bbox_inches='tight')
#
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_mu in range(0,n_mu,k_jump):
#     ax.plot(x_sigma, rate_activity[k_mu, :], label='mu='+str(round(mu[k_mu],2)))
# ax.set_xlabel('Uncertainty', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_ylabel('Neural activity', fontsize=15)
# ax.set_title('Rate',fontsize=20)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/rate_sigma.png', bbox_inches='tight')
#
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_sigma in range(0,n_sigma,k_jump):
#     ax.plot(x_mu, ppc_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
# ax.set_xlabel('Probability', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_ylabel('Neural activity',fontsize=15)
# ax.set_title('PPC',fontsize=20)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/ppc_mu.png', bbox_inches='tight')
#
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_mu in range(0,n_mu,k_jump):
#     ax.plot(x_sigma, ppc_activity[k_mu, :], label='mu='+str(round(mu[k_mu],2)))
# ax.set_xlabel('Uncertainty', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_ylabel('Neural activity',fontsize=15)
# ax.set_title('PPC',fontsize=20)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/ppc_sigma.png', bbox_inches='tight')
#
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_sigma in range(0,n_sigma,k_jump):
#     ax.plot(x_mu, dpc_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
# ax.set_ylabel('Neural activity')
# ax.set_xlabel('Probability', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_title('DPC',fontsize=20)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/dpc_mu.png', bbox_inches='tight')
#
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_mu in range(0,n_mu,k_jump):
#     ax.plot(x_sigma, dpc_activity[k_mu, :], label='mu='+str(round(mu[k_mu],2)))
# ax.set_xlabel('Uncertainty', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_ylabel('Neural activity',fontsize=15)
# ax.set_title('DPC',fontsize=20)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/dpc_sigma.png', bbox_inches='tight')
#
# plt.show()

# width = np.mean(mu)/(2*ppc_voxel.tuning_curve[0].N)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.linspace(0, 1, ppc_voxel.tuning_curve[0].N)
# ax.bar(x, ppc_voxel.subpopulation_fraction[0], width=width)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# ax.set_xlabel('Probability', fontsize=15)
# ax.set_ylabel('Neural fraction', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_title('Mixture in this voxel', fontsize=20)
# plt.savefig('output/figures/mixture_mu.png', bbox_inches='tight')
#
# plt.show()
#
# width = np.mean(mu)/(2*ppc_voxel.tuning_curve[1].N)*(tc_upper_bound_sigma-tc_lower_bound_sigma)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.linspace(tc_lower_bound_sigma, tc_upper_bound_sigma, ppc_voxel.tuning_curve[1].N)
# ax.bar(x, ppc_voxel.subpopulation_fraction[1], width=width)
# x0,x1 = ax.get_xlim()
# y0,y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# ax.set_ylabel('Neural fraction', fontsize=15)
# ax.set_xlabel('Uncertainty', fontsize=15)
# ax.tick_params(labelsize=15)
# ax.set_title('Mixture in this voxel', fontsize=20)
# plt.savefig('output/figures/mixture_sigma.png', bbox_inches='tight')
#
# plt.show()

### END INDIVIDUAL PLOTS ###

### BEGINNING OF VULGARIZATION PLOTS ###

high = 0.7
low = 0.1
width = 0.5
low_low = np.array([low, low])
low_high = np.array([low, high])
high_low = np.array([high, low])
high_high = np.array([high, high])

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(0, 1, 2, endpoint=True)
ax.bar(x, low_low, width=width, color=['black', 'grey'])
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.set_ylabel('Neural activity', fontsize=15)
ax.set_title('Mixture in this voxel', fontsize=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.savefig('output/figures/low_low.png', bbox_inches='tight')

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'Probability neuron'
labels[2] = ''
labels[3] = 'Uncertainty neuron'

ax.set_xticklabels(labels)
label = ax.yaxis.get_major_ticks()[2].label
label.set_fontsize(15)
plt.xticks(rotation=45)
ax.get_yaxis().set_ticks([])

plt.show()

a=1

### BEGIN ALL ACTIVITIES ON SAME FIGURES ###

# fig = plt.figure()
# # Big subplots for common axes
#
# ax_left = fig.add_subplot(121)
# # Turn off axis lines and ticks of the big subplot
# ax_left.spines['top'].set_color('none')
# ax_left.spines['bottom'].set_color('none')
# ax_left.spines['left'].set_color('none')
# ax_left.spines['right'].set_color('none')
# ax_left.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
# ax_left.set_xlabel('Inferred probability')
#
# ax_right = fig.add_subplot(122)
# # Turn off axis lines and ticks of the big subplot
# ax_right.spines['top'].set_color('none')
# ax_right.spines['bottom'].set_color('none')
# ax_right.spines['left'].set_color('none')
# ax_right.spines['right'].set_color('none')
# ax_right.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
# ax_right.set_xlabel('Inferred standard deviation')
#
#
# ax_rate_mu = fig.add_subplot(421)
# for k_sigma in range(0,n_sigma,k_jump):
#     ax_rate_mu.plot(x_mu, rate_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
# ax_rate_mu.set_ylabel('Signal intensity')
# ax_rate_mu.set_title('Rate coding signal')
# ax_rate_mu.legend()
# ax_rate_mu.get_xaxis().set_ticks([])
#
# ax_rate_sd = fig.add_subplot(422)
# for k_mu in range(0,n_mu,k_jump):
#     ax_rate_sd.plot(x_sigma, rate_activity[k_mu, :], label='mu='+str(round(mu[k_mu],2)))
# ax_rate_sd.set_ylabel('Signal intensity')
# ax_rate_sd.set_title('Rate coding signal')
# ax_rate_sd.legend()
# ax_rate_sd.get_xaxis().set_ticks([])
#
#
# ax_ppc_mu = fig.add_subplot(423)
# for k_sigma in range(0,n_sigma,k_jump):
#     ax_ppc_mu.plot(x_mu, ppc_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
# ax_ppc_mu.set_ylabel('Signal intensity')
# ax_ppc_mu.set_title('PPC signal')
# ax_ppc_mu.legend()
# ax_ppc_mu.get_xaxis().set_ticks([])
#
# ax_ppc_sd = fig.add_subplot(424)
# for k_mu in range(0, n_mu, k_jump):
#     ax_ppc_sd.plot(x_sigma, ppc_activity[k_mu, :], label='mu='+str(round(mu[k_mu],2)))
# ax_ppc_sd.set_ylabel('Signal intensity')
# ax_ppc_sd.set_title('PPC signal')
# ax_ppc_sd.legend()
# ax_ppc_sd.get_xaxis().set_ticks([])
#
#
# ax_dpc_mu = fig.add_subplot(425)
# for k_sigma in range(0,n_sigma,k_jump):
#     ax_dpc_mu.plot(x_mu, dpc_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
# ax_dpc_mu.set_ylabel('Signal intensity')
# ax_dpc_mu.set_title('DPC signal')
# ax_dpc_mu.legend()
# ax_dpc_mu.get_xaxis().set_ticks([])
#
#
# ax_dpc_sd = fig.add_subplot(426)
# for k_mu in range(0,n_mu,k_jump):
#     ax_dpc_sd.plot(x_sigma, dpc_activity[k_mu, :], label='mu='+str(round(mu[k_mu],2)))
# ax_dpc_sd.set_ylabel('Signal intensity')
# ax_dpc_sd.set_title('DPC signal')
# ax_dpc_sd.legend()
# ax_dpc_sd.get_xaxis().set_ticks([])
#
# width = np.mean(mu)/(2*ppc_voxel.tuning_curve[1].N)
#
# ax_frac_mu = fig.add_subplot(427)
# x = np.linspace(0, 1, ppc_voxel.tuning_curve[0].N)
# ax_frac_mu.bar(x, ppc_voxel.subpopulation_fraction[0], width=width)
# ax_frac_mu.set_ylabel('Fraction')
# ax_frac_mu.set_title('Neural fraction')
#
# width = np.mean(sigma)/(2*ppc_voxel.tuning_curve[1].N)
#
# ax_frac_sd = fig.add_subplot(428)
# x = np.linspace(tc_lower_bound_sigma, tc_upper_bound_sigma, ppc_voxel.tuning_curve[1].N)
# ax_frac_sd.bar(x, ppc_voxel.subpopulation_fraction[1], width=width)
# ax_frac_sd.set_ylabel('Fraction')
# ax_frac_sd.set_title('Neural fraction')
#
#
# ax_rate_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_rate_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])
# ax_ppc_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_ppc_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])
# ax_dpc_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_dpc_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])
# ax_frac_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_frac_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])
#
# plt.show()


# #Just plot the activity
# if not plot_all_range:
#     fig = plt.figure()
#     x = np.linspace(0, 380, 380)
#     plt.plot(x, dpc_activity)
#     plt.show()


#############

# Plot the optimal tuning curves

# fig = plt.figure()
# x = np.linspace(tc_lower_bound_mu,tc_upper_bound_mu,1000)
# plt.subplot(211)
# for i in range(0, N_mu):
#     plt.plot(x, tc_mu.f(x, i))
#
# plt.xlabel('Preferred probability')
# plt.ylabel('Tuning curve value')
# plt.title('Optimal tuning curves for encoding the mean (N='+str(tc_mu.N)+')')
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
# tested_t = [0.01,0.03,0.05,0.08,0.1]    # The different std
# x = np.linspace(tc_lower_bound_mu, tc_upper_bound_mu, 1000)
# tc_sum = np.zeros([len(tested_t), 1000])    # Initialization of the different TC sum's
#
# for k_t in range(len(tested_t)):
#     tc = tuning_curve(tc_type_mu, N_mu, tested_t[k_t], tc_lower_bound_mu, tc_upper_bound_mu)
#     for i in range(N_mu):
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
# tested_t = [5e-3, 1e-2, 2e-2, 3e-2, 5e-2]    # The different std
# x = np.linspace(tc_lower_bound_sigma, tc_upper_bound_sigma, 1000)
# tc_sum = np.zeros([len(tested_t), 1000])
#
# for k_t in range(len(tested_t)):
#     tc = tuning_curve(tc_type_sigma, N_sigma, tested_t[k_t], tc_lower_bound_sigma, tc_upper_bound_sigma)
#     for i in range(N_sigma):
#         tc_sum[k_t, :] += tc.f(x, i)
#
# plt.subplot(212)
# for k_t in range(len(tested_t)):
#     plt.plot(x, tc_sum[k_t, :], label='t='+str(tested_t[k_t]))
#
# plt.xlabel('Standard deviation')
# plt.ylabel('Sum over the tuning curves')
# plt.legend()
# plt.show()

##################
