import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
import numpy as np
#import decimal
import matplotlib.pyplot as plt

import neural_proba
from neural_proba import distrib
from neural_proba import tuning_curve
from neural_proba import voxel
from neural_proba import experiment
from neural_proba import fmri


# Define the seed to reproduce results from random processes
rand.seed(2);

# Import Matlab structure containing data and defining inputs
data_mat = sio.loadmat('data/ideal_observer_1.mat', struct_as_record=False)

[p1g2_dist, p1g2_mean, p1g2_sd] = neural_proba.import_distrib_param(data_mat)

plot_all_range = False

### Generate equivalent beta distribution

# DISTRIBUTION SIMULATED IN ORDER TO PLOT THE SHAPE OF THE SIGNAL

if plot_all_range:

    # Generate data from beta distributions samples from means and std
    n_mean = 30    # Number of generated means
    n_sigma = 30    # Number of generated standard deviations
    q_mean = np.linspace(np.min(p1g2_mean), np.max(p1g2_mean), n_mean)
    sigma = np.linspace(np.min(p1g2_sd), np.mean(p1g2_sd), n_sigma)    # We don't go to the max to have bell shaped beta

    # Creation of a list of simulated distributions
    simulated_distrib = [[None for j in range(n_sigma)] for i in range(n_mean)]

    for k_mean in range(n_mean):
        for k_sigma in range(n_sigma):
            simulated_distrib[k_mean][k_sigma] = distrib(q_mean[k_mean], sigma[k_sigma])

    # # Plots the distribution
    # fig = plt.figure()
    # k_mean = -1
    # k_sigma = -1
    # x = np.linspace(0, 1, 100)
    # y = simulated_distrib[k_mean][k_sigma].beta(x)
    # plt.plot(x, y)    # Full distribution
    # plt.show()

    # Resolution of the continuous plots
    plot_resolution = n_mean    # we consider the same resolution as for DPC

    # we define the x-axis for varying mean
    x_mean = np.linspace(np.min(q_mean), np.max(q_mean), plot_resolution)
    # we define the x-axis for varying uncertainty
    x_sigma = np.linspace(np.min(sigma), np.max(sigma), plot_resolution)
else:
    n_stimuli = p1g2_mean.shape[1]
    q_mean = np.reshape(p1g2_mean, (n_stimuli,))
    sigma = np.reshape(p1g2_sd, (n_stimuli,))
    # Creation of a list of simulated distributions
    simulated_distrib = [None for i in range(n_stimuli)]
    for k in range(n_stimuli):
        simulated_distrib[k] = distrib(q_mean[k], sigma[k])

# We find the variance of the data in order to scale equally activity from mean and activity from uncertainty
q_mean_sd = np.std(q_mean)    # Variance of the signal of q_mean's
sigma_sd = np.std(sigma)    # Variance of the signal of sigma's

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mean = 0
tc_upper_bound_mean = 1
tc_lower_bound_sigma = np.min(sigma)-np.std(sigma)
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_sigma = np.max(sigma)+np.std(sigma)

### 1) Rate coding simulation


# Properties of the "rate voxel" to be simulated
coding_scheme = 'rate'
population_fraction = [0.5, 0.5]    # one for the mean, one for the std

# Creation of the "rate voxel"
rate_voxel = voxel(coding_scheme, population_fraction)

# Computes the signal
rate_activity = rate_voxel.generate_activity(simulated_distrib, q_mean_sd, sigma_sd)

### PPC simulation

# Properties of the voxel to be simulated
coding_scheme = 'ppc'
population_fraction = [0.5, 0.5]    # Population fraction (one mean, one std)

# TC related to the mean
tc_type_mean = 'gaussian'    # Tuning curve type
N_mean = 10    # Number of tuning curves
t_mean = 0.05    # The best value from the previous "sum" analysis
# Creates the tuning_curve object
tc_mean = tuning_curve(tc_type_mean, N_mean, t_mean, tc_lower_bound_mean, tc_upper_bound_mean)

# TC related to the uncertainty
tc_type_sigma = 'gaussian'    # Tuning curve type
N_sigma = 10    # Number of tuning curves
t_sigma = 5e-3    # The best value from the previous "sum" analysis
# Creates the tuning_curve object
tc_sigma = tuning_curve(tc_type_sigma, N_sigma, t_sigma, tc_lower_bound_sigma, tc_upper_bound_sigma)

# Creation of the "ppc voxel"
ppc_voxel = voxel(coding_scheme, population_fraction, [tc_mean, tc_sigma])

# Computes the signal
ppc_activity = ppc_voxel.generate_activity(simulated_distrib)

### DPC simulation

# Properties of the voxel to be simulated
coding_scheme = 'dpc'
population_fraction = [1]    # One population : related to tuning curves of the mean

# Creation of the "dpc voxel"
dpc_voxel = voxel(coding_scheme, population_fraction, [tc_mean])    # we only input the tuning curve of the mean

# We impose the same neuron fraction for DPC and PPC, for comparison
dpc_voxel.subpopulation_fraction = np.expand_dims(ppc_voxel.subpopulation_fraction[0, :], axis=0)

# Computes the signal
dpc_activity = dpc_voxel.generate_activity(simulated_distrib)

### PLOTS

if plot_all_range:

    # Plot the signal for different voxel types and different distributions
    k_jump = 6    # To skip some curves

    fig = plt.figure()

    # Big subplots for common axes

    ax_left = fig.add_subplot(121)
    # Turn off axis lines and ticks of the big subplot
    ax_left.spines['top'].set_color('none')
    ax_left.spines['bottom'].set_color('none')
    ax_left.spines['left'].set_color('none')
    ax_left.spines['right'].set_color('none')
    ax_left.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax_left.set_xlabel('Inferred probability')

    ax_right = fig.add_subplot(122)
    # Turn off axis lines and ticks of the big subplot
    ax_right.spines['top'].set_color('none')
    ax_right.spines['bottom'].set_color('none')
    ax_right.spines['left'].set_color('none')
    ax_right.spines['right'].set_color('none')
    ax_right.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax_right.set_xlabel('Inferred standard deviation')


    ax_rate_mean = fig.add_subplot(421)
    for k_sigma in range(0,n_sigma,k_jump):
        ax_rate_mean.plot(x_mean, rate_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
    ax_rate_mean.set_ylabel('Signal intensity')
    ax_rate_mean.set_title('Rate coding signal')
    ax_rate_mean.legend()
    ax_rate_mean.get_xaxis().set_ticks([])
    #ax_rate_mean.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')

    ax_rate_sd = fig.add_subplot(422)
    for k_mean in range(0,n_mean,k_jump):
        ax_rate_sd.plot(x_sigma, rate_activity[k_mean, :], label='q_mean='+str(round(q_mean[k_mean],2)))
    ax_rate_sd.set_ylabel('Signal intensity')
    ax_rate_sd.set_title('Rate coding signal')
    ax_rate_sd.legend()
    ax_rate_sd.get_xaxis().set_ticks([])


    ax_ppc_mean = fig.add_subplot(423)
    for k_sigma in range(0,n_sigma,k_jump):
        ax_ppc_mean.plot(x_mean, ppc_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
    ax_ppc_mean.set_ylabel('Signal intensity')
    ax_ppc_mean.set_title('PPC signal')
    ax_ppc_mean.legend()
    ax_ppc_mean.get_xaxis().set_ticks([])

    ax_ppc_sd = fig.add_subplot(424)
    for k_mean in range(0, n_mean, k_jump):
        ax_ppc_sd.plot(x_sigma, ppc_activity[k_mean, :], label='q_mean='+str(round(q_mean[k_mean],2)))
    ax_ppc_sd.set_ylabel('Signal intensity')
    ax_ppc_sd.set_title('PPC signal')
    ax_ppc_sd.legend()
    ax_ppc_sd.get_xaxis().set_ticks([])


    ax_dpc_mean = fig.add_subplot(425)
    for k_sigma in range(0,n_sigma,k_jump):
        ax_dpc_mean.plot(x_mean, dpc_activity[:, k_sigma], label='sigma='+str(round(sigma[k_sigma],2)))
    ax_dpc_mean.set_ylabel('Signal intensity')
    ax_dpc_mean.set_title('DPC signal')
    ax_dpc_mean.legend()
    ax_dpc_mean.get_xaxis().set_ticks([])


    ax_dpc_sd = fig.add_subplot(426)
    for k_mean in range(0,n_mean,k_jump):
        ax_dpc_sd.plot(x_sigma, dpc_activity[k_mean, :], label='q_mean='+str(round(q_mean[k_mean],2)))
    ax_dpc_sd.set_ylabel('Signal intensity')
    ax_dpc_sd.set_title('DPC signal')
    ax_dpc_sd.legend()
    ax_dpc_sd.get_xaxis().set_ticks([])

    width = np.mean(q_mean)/(2*ppc_voxel.tuning_curve[1].N)

    ax_frac_mean = fig.add_subplot(427)
    x = np.linspace(0, 1, ppc_voxel.tuning_curve[0].N)
    ax_frac_mean.bar(x, ppc_voxel.subpopulation_fraction[0], width=width)
    ax_frac_mean.set_ylabel('Fraction')
    ax_frac_mean.set_title('Neural fraction')

    width = np.mean(sigma)/(2*ppc_voxel.tuning_curve[1].N)

    ax_frac_sd = fig.add_subplot(428)
    x = np.linspace(tc_lower_bound_sigma, tc_upper_bound_sigma, ppc_voxel.tuning_curve[1].N)
    ax_frac_sd.bar(x, ppc_voxel.subpopulation_fraction[1], width=width)
    ax_frac_sd.set_ylabel('Fraction')
    ax_frac_sd.set_title('Neural fraction')


    ax_rate_mean.set_xlim([tc_lower_bound_mean, tc_upper_bound_mean])
    ax_rate_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])
    ax_ppc_mean.set_xlim([tc_lower_bound_mean, tc_upper_bound_mean])
    ax_ppc_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])
    ax_dpc_mean.set_xlim([tc_lower_bound_mean, tc_upper_bound_mean])
    ax_dpc_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])
    ax_frac_mean.set_xlim([tc_lower_bound_mean, tc_upper_bound_mean])
    ax_frac_sd.set_xlim([tc_lower_bound_sigma, tc_upper_bound_sigma])

    plt.show()


# #Just plot the activity
# if not plot_all_range:
#     fig = plt.figure()
#     x = np.linspace(0, 380, 380)
#     plt.plot(x, dpc_activity)
#     plt.show()


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
# tested_t = [2e-3, 3e-3, 5e-3, 6e-3, 7e-3, 8e-3, 0.01]    # The different std
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
# plt.xlabel('Standard deviation')
# plt.ylabel('Sum over the tuning curves')
# plt.legend()
# plt.show()

##################

# Experimental design information
eps = 1e-5    # For floating points issues

initial_time = 0+eps
n_blocks = 1
n_stimuli = 10
between_stimuli_duration = 1.3+eps
final_time = between_stimuli_duration*n_stimuli
stimulus_onsets = np.arange(initial_time+between_stimuli_duration, final_time +eps, between_stimuli_duration)
stimulus_durations = 0.01*np.ones_like(stimulus_onsets)    # Dirac-like stimuli

# Creation of experiment object
exp = experiment(initial_time, final_time, n_blocks, stimulus_onsets, stimulus_durations, simulated_distrib)

# fMRI information


initial_frame_time = 0
final_frame_time = 30
dt = 0.01    # Temporal resolution of the fMRI scanner

between_scans_duration = 2    # in seconds
scan_times = np.arange(initial_frame_time+between_scans_duration, exp.final_time, between_scans_duration)
# Creation of fmri object
simu_fmri = fmri(initial_frame_time, final_frame_time, dt, scan_times)

frame_times = simu_fmri.frame_times

# Computes the BOLD signal

amplitudes = np.ones(n_stimuli)    # The amplitudes of the neural signal


hrf_models = [None, 'spm']

#########################################################################
# sample the hrf
fig = plt.figure(figsize=(9, 4))
for i, hrf_model in enumerate(hrf_models):
    signal, name, stim = simu_fmri.get_bold_signal(exp, amplitudes, hrf_model)

    plt.subplot(1, 2, i + 1)
    plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
    for j in range(signal.shape[1]):
        plt.plot(frame_times, signal.T[j], label=name[j])
    plt.xlabel('time (s)')
    plt.legend(loc=1)
    plt.title(hrf_model)

plt.subplots_adjust(bottom=.12)
plt.show()
