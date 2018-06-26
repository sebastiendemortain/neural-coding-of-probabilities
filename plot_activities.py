import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
from scipy import integrate

import numpy as np
#import decimal
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

import utils

import neural_proba
from neural_proba import distrib
from neural_proba import tuning_curve
from neural_proba import voxel
from neural_proba import experiment
from neural_proba import fmri


# Define the seed to reproduce results from random processes
rand.seed(8)
#rand.seed(9)
# In order to plot the full range of distribution with 2D grid of mean and sd (True)
# or the distributions of the sequence only (False)
plot_all_range = True

#########################################

# Import Matlab structure containing data and defining inputs
#data_mat = sio.loadmat('data/ideal_observer_1.mat', struct_as_record=False)

# [p1_dist, p1_mu, p1_sd] = neural_proba.import_distrib_param(1, 1, 380, 'HMM')
[p1_dist, p1_mu, p1_sd] = neural_proba.import_distrib_param(1000, 4, 380, 'transition')

p1_dist = p1_dist[0]
all_mu = np.concatenate(p1_mu).ravel()    # All values of mu (for percentiles)
p1_mu = p1_mu[0]

all_conf = -np.log(np.concatenate(p1_sd).ravel())    # All values of conf (for percentiles)
p1_sd = p1_sd[0]


# To get the percentiles

# fig = plt.figure()
# plt.hist(all_conf, bins=100)
# plt.title("Confidence histogram")
# plt.xlabel("Confidence")
# plt.ylabel("Frequency")
# plt.show()

mu_percentiles = np.percentile(all_mu, np.linspace(0, 100, 100))

# fig = plt.figure()
# plt.hist(all_mu, bins=100)
# plt.title("Mu histogram")
# plt.xlabel("Mu")
# plt.ylabel("Frequency")
# plt.show()

conf_percentiles = np.percentile(all_conf, np.linspace(0, 100, 100))

# fig = plt.figure()
# plt.hist(conf_percentiles, bins=100)
# plt.title("Confidence histogram")
# plt.xlabel("Confidence")
# plt.ylabel("Frequency")
# plt.show()

### Generate equivalent beta distribution



# DISTRIBUTION SIMULATED IN ORDER TO PLOT THE SHAPE OF THE SIGNAL

# Generate data from beta distributions samples from means and std
n_mu = 10    # Number of generated means
n_conf = 10    # Number of generated standard deviations

min_mu = 0.25
max_mu = 0.8

min_conf = 1.8
max_conf = 2.6

mu = np.linspace(min_mu, max_mu, n_mu)
conf = np.linspace(min_conf, max_conf, n_conf)    # We don't go to the max to have bell shaped beta

# Creation of a list of simulated distributions
simulated_distrib = [[None for j in range(n_conf)] for i in range(n_mu)]

for k_mu in range(n_mu):
    for k_conf in range(n_conf):
        simulated_distrib[k_mu][k_conf] = distrib(mu[k_mu], np.exp(-conf[k_conf]))

# # Plots the distribution
# fig = plt.figure()
# k_mu = -1
# k_conf = -1
# x = np.linspace(0, 1, 100)
# y = simulated_distrib[k_mu][k_conf].beta(x)
# plt.plot(x, y)    # Full distribution
# plt.show()

# Resolution of the continuous plots
plot_resolution = n_mu    # we consider the same resolution as for DPC

# we define the x-axis for varying mean
x_mu = np.linspace(np.min(mu), np.max(mu), plot_resolution)
# we define the x-axis for varying uncertainty
x_conf = np.linspace(np.min(conf), np.max(conf), plot_resolution)

# # We remove the infinite values
# inf_indices = [0, 1, 205, 206, 207, 208, 299]
# p1_mu = np.delete(p1_mu, inf_indices)
# p1_sd = np.delete(p1_sd, inf_indices)
# p1_dist = np.delete(p1_dist, inf_indices, 1)

#
# # Plots the distribution
# fig = plt.figure()
# x = np.linspace(0, 1, 50)
# y = p1_dist[:, 299]
# plt.plot(x, y)    # Full distribution
# plt.show()


# We find the variance of the data in order to scale equally activity from mean and activity from uncertainty
mu_sd = 0.219    # Variance of the signal of mu's
conf_sd = 0.284    # Variance of the signal of conf's

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mu = 0
tc_upper_bound_mu = 1
tc_lower_bound_conf = 1.1
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_conf = 2.6

# Optimal t for each N
k_N = 4
N_array = np.array([2, 4, 6, 8, 10, 14, 20])

t_mu_gaussian_array = np.array([0.15, 0.1, 7e-2, 5e-2, 4e-2, 3e-2, 2e-2])
t_conf_gaussian_array = np.array([0.25, 0.15, 0.10, 8e-2, 6e-2, 4e-2, 3e-2])

t_mu_sigmoid_array = np.sqrt(2*np.pi)/4*t_mu_gaussian_array
t_conf_sigmoid_array = np.sqrt(2*np.pi)/4*t_conf_gaussian_array

# fig = plt.figure()
# plt.plot(N_array, t_mu_sigmoid_array, label='mu sigmoid')
# plt.plot(N_array, t_conf_sigmoid_array, label='conf sigmoid')
# plt.plot(N_array, t_mu_gaussian_array, label='mu gaussian')
# plt.plot(N_array, t_conf_gaussian_array, label='conf gaussian')
#
# plt.legend()
#
# plt.show()


k_scheme = 1
scheme_array = ['gaussian', 'sigmoid']

if scheme_array[k_scheme]=='gaussian':
    t_mu_array = t_mu_gaussian_array
    t_conf_array = t_conf_gaussian_array
elif scheme_array[k_scheme]=='sigmoid':
    t_mu_array = t_mu_sigmoid_array
    t_conf_array = t_conf_sigmoid_array


###
# w = neural_proba.get_subpopulation_fraction(1, 10, subpopulation_sparsity_exp = 1)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.linspace(0, 1, 10)
# ax.bar(x, w[0], width = 0.03)
# ax.set_title('Mixture in this voxel', fontsize=20)
# plt.savefig('output/figures/mixture_random.png', bbox_inches='tight')
#
# plt.show()

### 1) Rate coding simulation

# Properties of the "rate voxel" to be simulated
coding_scheme = 'rate'
population_fraction = np.array([0.5, 0.5])    # one for the mean, one for the std
subpopulation_fraction = np.ones((2, 1))

# Creation of the "rate voxel"
rate_voxel = voxel(coding_scheme, population_fraction, subpopulation_fraction)

# Computes the signal
rate_activity = rate_voxel.generate_activity(simulated_distrib, mu_sd, conf_sd)

### 2) PPC simulation

# Properties of the voxel to be simulated
coding_scheme = 'ppc'
population_fraction = np.array([0.5, 0.5])    # Population fraction (one mean, one std)
# TC related to the mean
tc_type_mu = scheme_array[k_scheme]    # Tuning curve type
N_mu = N_array[k_N]    # Number of tuning curves
t_mu = 0.2*t_mu_array[k_N]    # The best value from the previous "sum" analysis
# Creates the tuning_curve object
tc_mu = tuning_curve(tc_type_mu, N_mu, t_mu, tc_lower_bound_mu, tc_upper_bound_mu)

# TC related to the uncertainty
tc_type_conf = scheme_array[k_scheme]    # Tuning curve type
N_conf = N_array[k_N]    # Number of tuning curves
t_conf = t_conf_array[k_N]    # The best value from the previous "sum" analysis
# Creates the tuning_curve object
tc_conf = tuning_curve(tc_type_conf, N_conf, t_conf, tc_lower_bound_conf, tc_upper_bound_conf)

# Subpopulation fraction random creation (we assume N_mu=N_conf)
subpopulation_fraction = neural_proba.get_subpopulation_fraction(len(population_fraction),
                                                                 N_mu)

# Creation of the "ppc voxel"
ppc_voxel = voxel(coding_scheme, population_fraction, subpopulation_fraction, [tc_mu, tc_conf])

# Computes the signal
ppc_activity = ppc_voxel.generate_activity(simulated_distrib)

### 3) DPC simulation

# Properties of the voxel to be simulated
coding_scheme = 'dpc'
population_fraction = [1]    # One population : related to tuning curves of the mean

# We impose the same neuron fraction for DPC and PPC, for comparison
subpopulation_fraction = np.expand_dims(ppc_voxel.subpopulation_fraction[0, :], axis=0)

# Creation of the "dpc voxel" with identical subpopulation for mu
dpc_voxel = voxel(coding_scheme, population_fraction, subpopulation_fraction, [tc_mu])
dpc_voxel.weights = np.expand_dims(ppc_voxel.weights[0, :], axis=0)

# Computes the signal
dpc_activity = dpc_voxel.generate_activity(simulated_distrib, use_high_integration_resolution=False)

### PLOTS

# Plot the signal for different voxel types and different distributions
k_jump = 6    # To skip some curves
ax_fontsize = 15
xtick_fontsize = 15
title_fontsize = 20

#### BEGIN INDIVIDUAL ACTIVITIES IPLOTS ###
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_conf in range(0,n_conf,k_jump):
#     ax.plot(x_mu, rate_activity[:, k_conf], label='confidence='+str(round(conf[k_conf],1)))
# utils.plot_detail(fig, ax, '$\mu$', 'Global firing activity (Hz)', xtick_fontsize, xfontstyle='italic')
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# #ax.get_xaxis().set_ticks([])
# #
# plt.savefig('output/figures/rate_mu.pdf', bbox_inches='tight')
# #
# plt.show()
# #
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_conf in range(0,n_conf,k_jump):
#     ax.plot(x_mu, ppc_activity[:, k_conf], label='confidence='+str(round(conf[k_conf],1)))
# utils.plot_detail(fig, ax, '$\mu$', 'Global firing activity (Hz)', xtick_fontsize, xfontstyle='italic')
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/ppc_mu.pdf', bbox_inches='tight')
#
# plt.show()
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_conf in range(0,n_conf,k_jump):
#     ax.plot(x_mu, dpc_activity[:, k_conf], label='confidence='+str(round(conf[k_conf],1)))
# utils.plot_detail(fig, ax, '$\mu$', 'Global firing activity (Hz)', xtick_fontsize, xfontstyle='italic')
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/dpc_mu.pdf', bbox_inches='tight')
#
# plt.show()
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_mu in range(0,n_mu,k_jump):
#     ax.plot(x_conf, rate_activity[k_mu, :], label='$\mu$='+str(round(mu[k_mu],2)))
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# utils.plot_detail(fig, ax, 'Confidence', 'Global firing activity (Hz)', xtick_fontsize)
#
# plt.savefig('output/figures/rate_conf.pdf', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_mu in range(0,n_mu,k_jump):
#     ax.plot(x_conf, ppc_activity[k_mu, :], label='$\mu$='+str(round(mu[k_mu],2)))
# utils.plot_detail(fig, ax, 'Confidence', 'Global firing activity (Hz)', xtick_fontsize)
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/ppc_conf.pdf', bbox_inches='tight')
#
# plt.show()
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k_mu in range(0,n_mu,k_jump):
#     ax.plot(x_conf, dpc_activity[k_mu, :], label='$\mu$='+str(round(mu[k_mu],2)))
# utils.plot_detail(fig, ax, 'Confidence', 'Global firing activity (Hz)', xtick_fontsize)
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# # ax_rate_mu.get_xaxis().set_ticks([])
# plt.savefig('output/figures/dpc_conf.pdf', bbox_inches='tight')
#
# plt.show()
#
# width = np.mean(mu)/(2*ppc_voxel.tuning_curve[0].N)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.linspace(1/ppc_voxel.tuning_curve[0].N, 1-1/ppc_voxel.tuning_curve[0].N, ppc_voxel.tuning_curve[0].N,
#                 endpoint=True)
# ax.bar(x, ppc_voxel.subpopulation_fraction[0], width=width)
# utils.plot_detail(fig, ax, 'Probability tuning curve mean', 'Neural fraction', xtick_fontsize)
#
# # x = np.linspace(2/ppc_voxel.tuning_curve[0].N, 1-2/ppc_voxel.tuning_curve[0].N, ppc_voxel.tuning_curve[0].N/2,
# #                 endpoint=True)
# # ax.bar(x, ppc_voxel.subpopulation_fraction[0][:int(ppc_voxel.tuning_curve[0].N/2)], width=width, label='Increasing sigmoid', alpha=0.5)
# # ax.bar(x, ppc_voxel.subpopulation_fraction[0][int(ppc_voxel.tuning_curve[0].N/2):], width=width, label='Decreasing sigmoid', alpha=0.5)
# # utils.plot_detail(fig, ax, 'Probability tuning curve inflection point', 'Neural fraction', xtick_fontsize)
# ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8),  shadow=False, ncol=1)
# ax.tick_params(labelsize=xtick_fontsize)
# ax.set_title('Mixture in this voxel', fontsize=20)
# plt.savefig('output/figures/mixture_mu.pdf', bbox_inches='tight')
#
# plt.show()
#
#
# width = np.mean(mu)/(2*ppc_voxel.tuning_curve[1].N)*(tc_upper_bound_conf-tc_lower_bound_conf)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.linspace(tc_lower_bound_conf+1/ppc_voxel.tuning_curve[1].N, tc_upper_bound_conf-1/ppc_voxel.tuning_curve[1].N, ppc_voxel.tuning_curve[1].N,
#                 endpoint=True)
# ax.bar(x, ppc_voxel.subpopulation_fraction[1], width=width)
# utils.plot_detail(fig, ax, 'Confidence tuning curve mean', 'Neural fraction', xtick_fontsize)
#
# # x = np.linspace(tc_lower_bound_conf+2/ppc_voxel.tuning_curve[1].N, tc_upper_bound_conf-2/ppc_voxel.tuning_curve[1].N, ppc_voxel.tuning_curve[1].N/2,
# #                 endpoint=True)
# # ax.bar(x, ppc_voxel.subpopulation_fraction[1][:int(ppc_voxel.tuning_curve[1].N/2)], width=width, label='Increasing sigmoid', alpha=0.5)
# # ax.bar(x, ppc_voxel.subpopulation_fraction[1][int(ppc_voxel.tuning_curve[1].N/2):], width=width, label='Decreasing sigmoid', alpha=0.5)
# # utils.plot_detail(fig, ax, 'Confidence tuning curve inflection point', 'Neural fraction', xtick_fontsize)
# ax.set_title('Mixture in this voxel', fontsize=20)
# ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8),  shadow=False, ncol=1)
# plt.savefig('output/figures/mixture_conf.pdf', bbox_inches='tight')
#
# plt.show()

## END INDIVIDUAL PLOTS ###

# ### BEGINNING OF RATE VULGARIZATION PLOTS ###
#
# mu = 0.6
# conf = 1.9/2.6
# acti = np.array([mu, conf])
# width = 0.5
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = ['Probability neuron', 'Confidence neuron']
# ax.bar(x, acti, width=width, color = ['red', 'blue'])
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, '', 'Firing rate (Hz)', xtick_fontsize)
# ax.plot(np.linspace(-0.3, 1.3, 10), np.ones(10), color='black',
#                        linestyle='--')
# plt.xticks(rotation=45, fontsize=xtick_fontsize)
# ax.get_yaxis().set_ticks([])
#
# plt.savefig('output/figures/rate_example_mu_conf.pdf', bbox_inches='tight')
#
# plt.show()
# a=1

# high = 0.9
# low = 0.1
# width = 0.5
# low_low = np.array([low, low])
# low_high = np.array([low, high])
# high_low = np.array([high, low])
# high_high = np.array([high, high])
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = ['Probability neuron', 'Uncertainty neuron']
# ax.bar(x, low_low, width=width, color=['red', 'blue'])
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, '', 'Firing rate (Hz)', xtick_fontsize)
#
# plt.xticks(rotation=45, fontsize=xtick_fontsize)
# ax.get_yaxis().set_ticks([])
#
# plt.savefig('output/figures/low_low.png', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = ['Probability neuron', 'Uncertainty neuron']
# ax.bar(x, high_low, width=width, color=['red', 'blue'])
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, '', 'Firing rate (Hz)', xtick_fontsize)
#
# plt.xticks(rotation=45, fontsize=xtick_fontsize)
# ax.get_yaxis().set_ticks([])
#
# plt.savefig('output/figures/high_low.png', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = ['Probability neuron', 'Uncertainty neuron']
# ax.bar(x, high_high, width=width, color=['red', 'blue'])
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, '', 'Firing rate (Hz)', xtick_fontsize)
# plt.xticks(rotation=45, fontsize=xtick_fontsize)
# ax.get_yaxis().set_ticks([])
#
# plt.savefig('output/figures/high_high.png', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = ['Probability neuron', 'Uncertainty neuron']
# ax.bar(x, low_high, width=width, color=['red', 'blue'])
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, '', 'Firing rate (Hz)', xtick_fontsize)
#
# plt.xticks(rotation=45, fontsize=xtick_fontsize)
# ax.get_yaxis().set_ticks([])
#
# plt.savefig('output/figures/low_high.png', bbox_inches='tight')
#
# plt.show()

# ## END OF RATE VULGARIZATION PLOT
#
# ## BEGINNING OF PPC VULGARIZATION PLOT

# mu_point = 0.6
# tc_mu_point = np.zeros(10)
# conf_point = 1.9
# tc_conf_point = np.zeros(10)
#
# color_point = [None for i in range(10)]
# dash_line = [None for i in range(10)]
#
#
# fig = plt.figure()
# x = np.linspace(tc_lower_bound_mu, tc_upper_bound_mu,1000)
# ax = plt.subplot(111)
# for i in range(0, N_mu):
#     p = ax.plot(x, tc_mu.f(x, i), label='Neuron '+str(i+1))
#     tc_mu_point[i] = tc_mu.f(mu_point, i)
#     color_point[i] = p[0].get_color()
#     dash_line[i] = ax.plot(mu_point*np.ones(10), np.linspace(0, tc_mu_point[i], 10), color=color_point[i],
#                            linestyle='--')
# ax.scatter([mu_point*np.ones(10)], [np.reshape(tc_mu_point, (10,))], color=color_point)
# ax.set_ylim(0, 1.05)
#
# utils.plot_detail(fig, ax, '$\mu$', 'Firing rate (Hz)', xtick_fontsize, aspect=0.5)
#
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# ax.get_yaxis().set_ticks([])
#
# #plt.title('Optimal tuning curves for encoding the mean (N='+str(tc_mu.N)+')')
# plt.savefig('output/figures/tc_mu.pdf', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure()
#
# ax = plt.subplot(111)
# x = np.linspace(tc_lower_bound_conf,tc_upper_bound_conf, 1000)
# for i in range(0, N_conf):
#     plt.plot(x, tc_conf.f(x, i), label='Neuron '+str(i+1))
#     tc_conf_point[i] = tc_conf.f(conf_point, i)
#     dash_line[i] = ax.plot(conf_point*np.ones(10), np.linspace(0, tc_conf_point[i], 10), color=color_point[i],
#                            linestyle='--')
# ax.scatter([conf_point*np.ones(10)], [np.reshape(tc_conf_point, (10,))], color=color_point)
#
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, '$-\log(\sigma)$', 'Firing rate (Hz)', xtick_fontsize, aspect=0.5)
#
# # plt.title('Optimal tuning curves for encoding the uncertainty (N='+str(tc_conf.N)+')')
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
# plt.savefig('output/figures/tc_conf.pdf', bbox_inches='tight')
#
# plt.show()
#
#
# width = 0.3
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = [str(i+1) for i in range(10)]
# ax.bar(x, tc_mu_point, width=width, color=color_point)
# ax.set_ylim(0, 1.05)
#
# utils.plot_detail(fig, ax, 'Probability neuron index', 'Firing rate (Hz)', xtick_fontsize)
#
# plt.savefig('output/figures/color_mu.pdf', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = [str(i+1) for i in range(10)]
# ax.bar(x, tc_conf_point, width=width, color=color_point)
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, 'Confidence neuron index', 'Firing rate (Hz)', xtick_fontsize)
#
# plt.savefig('output/figures/color_conf.pdf', bbox_inches='tight')
#
# plt.show()

#
# ### END OF PPC VULGARIZATION PLOT
#
# ### BEGINNING OF DPC VULGARIZATION PLOT
# k_mu = 20
# k_conf = 1
#
# #dist = simulated_distrib[k_mu, k_conf]
# x = np.linspace(tc_lower_bound_mu, tc_upper_bound_mu,1000)
# mu1=0.2
# mu2=0.6
# conf1=0.08
# conf2=0.1
# pi1=0.0
# pi2=1
# beta = pi1*np.exp(-0.5*(x-mu1)**2/(conf1**2))+pi2*np.exp(-0.5*(x-mu2)**2/(conf2**2))
#
#
# # Integration param
# delta_x = 1/(1000-1)
#
# # Projection
# proj = np.zeros(10)
# # Transparency coefficient
# alpha = np.zeros(10)
#
# color_point = [None for i in range(10)]
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(x, beta, color='black')
# ax.set_ylim(0, 1.1)
# utils.plot_detail(fig, ax, 'p(H)', 'Probability density', xtick_fontsize, xfontstyle='italic', aspect=0.5)
# #plt.title('Optimal tuning curves for encoding the mean (N='+str(tc_mu.N)+')')
# plt.savefig('output/figures/dist_alone.pdf', bbox_inches='tight')
#
# plt.show()
#
# # fig = plt.figure()
# # ax = plt.subplot(111)
# # ax.plot(x, beta, color='black')
# # ax.set_ylim(0, 1.1)
# # utils.plot_detail(fig, ax, 'p(H)', 'Probability density', xtick_fontsize, xfontstyle='italic', aspect=0.5)
# # plt.annotate(s='', xy=(1,1), xytext=(0,0), arrowprops=dict(arrowstyle='<->'))
# # #plt.title('Optimal tuning curves for encoding the mean (N='+str(tc_mu.N)+')')
# # plt.savefig('output/figures/dist_alone.pdf', bbox_inches='tight')
# #
# # plt.show()
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(x, beta, color='black')
#
# for i in range(0, N_mu):
#     p = ax.plot(x, tc_mu.f(x, i), label='Neuron '+str(i+1))
#     #tc_mu_point[i] = tc_mu.f(mu_point, i)
#     color_point[i] = p[0].get_color()
#     #dash_line[i] = ax.plot(mu_point*np.ones(10), np.linspace(0, tc_mu_point[i], 10), color=color_point[i],
#     #                       linestyle='--')
#     proj[i] = np.dot(beta, tc_mu.f(x, i))*delta_x
# ax.set_ylim(0, 1.1)
# utils.plot_detail(fig, ax, 'p(H)', 'Firing rate (Hz)', xtick_fontsize, xfontstyle='italic', aspect=0.5)
#
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),  shadow=False, ncol=1)
#
# #plt.title('Optimal tuning curves for encoding the mean (N='+str(tc_mu.N)+')')
# plt.savefig('output/figures/dist_with_tuning_curves.pdf', bbox_inches='tight')
#
# for i in range(0, N_mu):
#     alpha[i] = 2*proj[i]/np.sum(proj)
#     ax.fill_between(x, tc_mu.f(x, i), color=color_point[i], alpha=alpha[i])
#
# #plt.title('Optimal tuning curves for encoding the mean (N='+str(tc_mu.N)+')')
# plt.savefig('output/figures/dist_with_filled_tuning_curves.pdf', bbox_inches='tight')
#
# plt.show()
#
# width = 0.3
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = [str(i+1) for i in range(10)]
# ax.bar(x, alpha, width=width, color=color_point)
# ax.set_ylim(0, 1.05)
# utils.plot_detail(fig, ax, 'Neuron index', 'Firing rate (Hz)', xtick_fontsize)
#
# plt.savefig('output/figures/color_dpc.pdf', bbox_inches='tight')
#
# plt.show()
### END OF DPC VULGARIZATION PLOT


# ### BEGIN ALL ACTIVITIES ON SAME FIGURES ###
#
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
# ax_right.set_xlabel('Inferred confidence')
#
#
# ax_rate_mu = fig.add_subplot(421)
# for k_conf in range(0,n_conf,k_jump):
#     ax_rate_mu.plot(x_mu, rate_activity[:, k_conf], label='uncertainty(s.d.)='+str(round(conf[k_conf],2)))
# ax_rate_mu.set_ylabel('Signal intensity')
# ax_rate_mu.set_title('Rate coding signal')
# ax_rate_mu.legend()
# ax_rate_mu.get_xaxis().set_ticks([])
#
# ax_rate_sd = fig.add_subplot(422)
# for k_mu in range(0,n_mu,k_jump):
#     ax_rate_sd.plot(x_conf, rate_activity[k_mu, :], label='p(head)='+str(round(mu[k_mu],2)))
# ax_rate_sd.set_ylabel('Signal intensity')
# ax_rate_sd.set_title('Rate coding signal')
# ax_rate_sd.legend()
# ax_rate_sd.get_xaxis().set_ticks([])
#
#
# ax_ppc_mu = fig.add_subplot(423)
# for k_conf in range(0,n_conf,k_jump):
#     ax_ppc_mu.plot(x_mu, ppc_activity[:, k_conf], label='uncertainty(s.d.)='+str(round(conf[k_conf],2)))
# ax_ppc_mu.set_ylabel('Signal intensity')
# ax_ppc_mu.set_title('PPC signal')
# ax_ppc_mu.legend()
# ax_ppc_mu.get_xaxis().set_ticks([])
#
# ax_ppc_sd = fig.add_subplot(424)
# for k_mu in range(0, n_mu, k_jump):
#     ax_ppc_sd.plot(x_conf, ppc_activity[k_mu, :], label='p(head)='+str(round(mu[k_mu],2)))
# ax_ppc_sd.set_ylabel('Signal intensity')
# ax_ppc_sd.set_title('PPC signal')
# ax_ppc_sd.legend()
# ax_ppc_sd.get_xaxis().set_ticks([])
#
#
# ax_dpc_mu = fig.add_subplot(425)
# for k_conf in range(0,n_conf,k_jump):
#     ax_dpc_mu.plot(x_mu, dpc_activity[:, k_conf], label='uncertainty(s.d.)='+str(round(conf[k_conf],2)))
# ax_dpc_mu.set_ylabel('Signal intensity')
# ax_dpc_mu.set_title('DPC signal')
# ax_dpc_mu.legend()
# ax_dpc_mu.get_xaxis().set_ticks([])
#
#
# ax_dpc_sd = fig.add_subplot(426)
# for k_mu in range(0,n_mu,k_jump):
#     ax_dpc_sd.plot(x_conf, dpc_activity[k_mu, :], label='p(head)='+str(round(mu[k_mu],2)))
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
# width = np.mean(conf)/(2*ppc_voxel.tuning_curve[1].N)
#
# ax_frac_sd = fig.add_subplot(428)
# x = np.linspace(tc_lower_bound_conf, tc_upper_bound_conf, ppc_voxel.tuning_curve[1].N)
# ax_frac_sd.bar(x, ppc_voxel.subpopulation_fraction[1], width=width)
# ax_frac_sd.set_ylabel('Fraction')
# ax_frac_sd.set_title('Neural fraction')
#
#
# ax_rate_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_rate_sd.set_xlim([tc_lower_bound_conf, tc_upper_bound_conf])
# ax_ppc_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_ppc_sd.set_xlim([tc_lower_bound_conf, tc_upper_bound_conf])
# ax_dpc_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_dpc_sd.set_xlim([tc_lower_bound_conf, tc_upper_bound_conf])
# ax_frac_mu.set_xlim([tc_lower_bound_mu, tc_upper_bound_mu])
# ax_frac_sd.set_xlim([tc_lower_bound_conf, tc_upper_bound_conf])
#
# plt.show()
#
#
# #Just plot the activity
# if not plot_all_range:
#     fig = plt.figure()
#     x = np.linspace(0, 380, 380)
#     plt.plot(x, dpc_activity)
#     plt.show()
#
#
# #############
#
# Plot the optimal tuning curves

fig = plt.figure()
x = np.linspace(tc_lower_bound_mu,tc_upper_bound_mu,1000)
plt.subplot(111)
for i in range(0, N_mu):
    plt.plot(x, tc_mu.f(x, i))

plt.xlabel('$\mu$', fontsize=xtick_fontsize)
plt.ylabel('Firing rate (Hz)', fontsize=xtick_fontsize)
plt.title('Probability tuning curves (N='+str(tc_mu.N)+')', fontsize=title_fontsize)

plt.show()
#
# fig = plt.figure()
# x = np.linspace(tc_lower_bound_conf,tc_upper_bound_conf,1000)
# for i in range(0, N_conf):
#     plt.plot(x, tc_conf.f(x, i))
#
# plt.xlabel('$-\log(\sigma)$', fontsize=xtick_fontsize)
# plt.ylabel('Firing rate (Hz)', fontsize=xtick_fontsize)
# plt.title('Confidence tuning curves (N='+str(tc_conf.N)+')', fontsize = title_fontsize)
# plt.show()
# #
# #
# # Find the optimal tuning curves' confidence for fixed N : we consider the lowest std such that the sum over the TC is constant
#
# tested_t = [5e-3, 0.01,0.03,0.05,0.08,0.1, 0.15]    # The different std
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

# tested_t = [1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2]    # The different std
# x = np.linspace(tc_lower_bound_conf, tc_upper_bound_conf, 1000)
# tc_sum = np.zeros([len(tested_t), 1000])
#
# for k_t in range(len(tested_t)):
#     tc = tuning_curve(tc_type_conf, N_conf, tested_t[k_t], tc_lower_bound_conf, tc_upper_bound_conf)
#     for i in range(N_conf):
#         tc_sum[k_t, :] += tc.f(x, i)
#
# plt.subplot(212)
# for k_t in range(len(tested_t)):
#     plt.plot(x, tc_sum[k_t, :], label='t='+str(tested_t[k_t]))
#
# plt.xlabel('Confidence')
# plt.ylabel('Sum over the tuning curves')
# plt.legend()
# plt.show()
##################

### SECOND SOLUTION : WE COMPUTE THE VARIANCE OF THE ACTIVITY VECTOR FOR SEVERAL MU'S

# For mu

tested_t = [5e-3, 0.01,0.03,0.05,0.08,0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5] #np.linspace(1e-3, 1e-1, 1000)#    # The different std
x = np.linspace(tc_lower_bound_mu, tc_upper_bound_mu, 1000)
tc_var = np.zeros(len(tested_t))    # Initialization of the variance vector
tc_sum = np.zeros([len(tested_t), 1000])

for k_t in range(len(tested_t)):
    tc_val = np.zeros([1000, N_mu])  # Initialization of the different TC sum's
    tc = tuning_curve(tc_type_mu, N_mu, tested_t[k_t], tc_lower_bound_mu, tc_upper_bound_mu)
    for i in range(N_mu):
        tc_val[:, i] = tc.f(x, i)
        tc_var[k_t] += np.var(tc_val[i, :])
        tc_sum[k_t, :] += tc.f(x, i)
    # pca = PCA(n_components=2)
    # pca.fit(tc_val)
    # print(pca.singular_values_)
fig = plt.figure()
# plt.subplot(111)
# plt.plot(tested_t, tc_var)
#
# plt.xlabel('t')
# plt.ylabel('Sum of variance of the activity vector')
# plt.legend()

plt.subplot(111)
for k_t in range(len(tested_t)):
    plt.plot(x, tc_sum[k_t, :], label='t='+str(tested_t[k_t]))
plt.xlabel('$\mu$', fontsize=xtick_fontsize)
plt.ylabel('Sum over the tuning curves', fontsize=xtick_fontsize)
plt.legend()
plt.show()
# For conf

tested_t = [1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 8e-1]    # The different std
x = np.linspace(tc_lower_bound_conf, tc_upper_bound_conf, 1000)
tc_var = np.zeros(len(tested_t))    # Initialization of the variance vector
tc_sum = np.zeros([len(tested_t), 1000])

for k_t in range(len(tested_t)):
    tc_val = np.zeros([1000, N_conf])  # Initialization of the different TC sum's
    tc = tuning_curve(tc_type_conf, N_conf, tested_t[k_t], tc_lower_bound_conf, tc_upper_bound_conf)
    for i in range(N_conf):
        tc_val[:, i] = tc.f(x, i)
        tc_var[k_t] += np.var(tc_val[i, :])
        tc_sum[k_t, :] += tc.f(x, i)
    # pca = PCA(n_components=2)
    # pca.fit(tc_val)
    # print(pca.singular_values_)
fig = plt.figure()
# plt.subplot(211)
# plt.plot(tested_t, tc_var)
#
# plt.xlabel('t')
# plt.ylabel('Sum of variance of the activity vector')
# plt.legend()

plt.subplot(111)
for k_t in range(len(tested_t)):
    plt.plot(x, tc_sum[k_t, :], label='t='+str(tested_t[k_t]))

plt.xlabel('Confidence', fontsize=xtick_fontsize)
plt.ylabel('Sum over the tuning curves', fontsize=xtick_fontsize)
plt.legend()
plt.show()