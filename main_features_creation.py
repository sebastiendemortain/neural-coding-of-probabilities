import os
import scipy
from scipy import io as sio
import random as rand
from scipy import stats
import numpy as np
#import decimal
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import neural_proba
from neural_proba import distrib
from neural_proba import tuning_curve
from neural_proba import voxel
from neural_proba import experiment
from neural_proba import fmri

# INPUTS

# Properties of the voxel to be simulated
true_coding_scheme = 'rate'
true_population_fraction = [0.5, 0.5]  # one for the mean, one for the std
fmri_gain = 1    # Amplification of the signal

# Fitted model
coding_scheme = 'rate'

# TC related to the mean
tc_type_mu = 'gaussian'    # Tuning curve type
N_mu = 10    # Number of tuning curves
t_mu = 0.05    # The best value from the previous "sum" analysis

# TC related to the uncertainty
tc_type_sigma = 'gaussian'    # Tuning curve type
N_sigma = 10    # Number of tuning curves
t_sigma = 5e-3    # The best value from the previous "sum" analysis

############################################################

# Define the seed to reproduce results from random processes
rand.seed(2);

# Define the subject and block modalities
n_subjects = 20
n_blocks = 4
distrib_type = 'HMM'
n_stimuli = 380

# load the data
[p1g2_dist_array, p1g2_mu_array, p1g2_sd_array] = neural_proba.import_distrib_param(n_subjects, n_blocks, n_stimuli,
                                                                                      distrib_type)

# Initialize the design matrix and the response vector
X = [[None for j in range(n_subjects)] for i in range(n_blocks)]
y = [[None for j in range(n_subjects)] for i in range(n_blocks)]

subject = 0
block = 0

p1g2_dist = p1g2_dist_array[subject][block]
p1g2_mu = p1g2_mu_array[subject][block]
p1g2_sd = p1g2_sd_array[subject][block]

# Plot the distribution over means and sd
# Merge the data all together
all_mu = np.concatenate(p1g2_mu_array, axis=2)
all_mu = np.reshape(all_mu, (n_subjects*n_blocks*n_stimuli,))

all_sigma = np.concatenate(p1g2_sd_array, axis=2)
all_sigma = np.reshape(all_sigma, (n_subjects*n_blocks*n_stimuli,))

plt.figure()
# hist_mu = np.histogram(all_mu, bin=100)
# hist_sigma = np.histogram(all_sigma, bin=100)
# x_mu = np.linspace(0, 1, len(hist_mu))
# x_sigma = np.linspace(np.min(all_sigma), np.max(all_sigma), len(hist_sigma))
ax_mu = plt.subplot(211)
ax_mu.hist(all_mu, bins=100)
ax_mu.set_xlabel('Probability')

ax_sigma = plt.subplot(212)
ax_sigma.hist(all_sigma, bins=100)
ax_sigma.set_xlabel('Standard deviation')
plt.show()

mu = p1g2_mu[0, :n_stimuli]
sigma = p1g2_sd[:n_stimuli]
dist = p1g2_dist[:, :n_stimuli]
# Creation of a list of simulated distributions
simulated_distrib = [None for k in range(n_stimuli)]
# test = np.zeros(n_stimuli)
# x = np.linspace(0, 1, 1000, endpoint=True)
for k in range(n_stimuli):
    # Normalization of the distribution
    norm_dist = dist[:, k] * (len(dist[1:, k]) - 1) / np.sum(dist[1:, k])
    simulated_distrib[k] = distrib(mu[k], sigma[k], norm_dist)
    # test[k] = np.max(simulated_distrib[k].beta(x))
    # if np.isinf(test[k]):
    #     print(k)

# We find the variance of the data in order to scale equally activity from mean and activity from uncertainty
mu_sd = np.std(mu)  # Variance of the signal of mu's
sigma_sd = np.std(sigma)  # Variance of the signal of sigma's

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mu = 0
tc_upper_bound_mu = 1
tc_lower_bound_sigma = np.min(sigma) - np.std(sigma)
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_sigma = np.max(sigma) + np.std(sigma)

# Creates the tuning_curve object
tc_mu = tuning_curve(tc_type_mu, N_mu, t_mu, tc_lower_bound_mu, tc_upper_bound_mu)

# Creates the tuning_curve object
tc_sigma = tuning_curve(tc_type_sigma, N_sigma, t_sigma, tc_lower_bound_sigma, tc_upper_bound_sigma)

# Experimental design information
eps = 1e-5  # For floating points issues

between_stimuli_duration = 1.3
initial_time = between_stimuli_duration + eps
n_blocks = 1
final_time = between_stimuli_duration * (n_stimuli + 1) + eps
stimulus_onsets = np.linspace(initial_time, final_time, n_stimuli)
stimulus_durations = 0.01 * np.ones_like(stimulus_onsets)  # Dirac-like stimuli

# Creation of experiment object
exp = experiment(initial_time, final_time, n_blocks, stimulus_onsets, stimulus_durations, simulated_distrib)

# fMRI information

final_frame_offset = 10  # Frame recording duration after the last stimulus has been shown
initial_frame_time = 0
final_frame_time = exp.final_time + final_frame_offset
dt = 0.01  # Temporal resolution of the fMRI scanner

between_scans_duration = 2  # in seconds
final_scan_offset = 10  # Scan recording duration after the last stimulus has been shown
initial_scan_time = initial_frame_time + between_scans_duration
final_scan_time = exp.final_time + final_scan_offset
scan_times = np.arange(initial_scan_time, final_scan_time, between_scans_duration)

# Creation of fmri object
simu_fmri = fmri(initial_frame_time, final_frame_time, dt, scan_times)

frame_times = simu_fmri.frame_times

# Computes the BOLD signal
# Tuning curves for each coding scheme
if true_coding_scheme == 'rate':
    true_tc = []
elif true_coding_scheme == 'ppc':
    true_tc = [tc_mu, tc_sigma]
elif true_coding_scheme == 'dpc':
    true_tc = [tc_mu]

# if coding_scheme == 'rate':
#     tc = []
# elif coding_scheme == 'ppc':
#     tc = [tc_mu, tc_sigma]
# elif coding_scheme == 'dpc':
#     tc = [tc_mu]

# Creation of the voxel
true_voxel = voxel(true_coding_scheme, true_population_fraction, true_tc)
# pickle.dump(true_voxel, output, pickle.HIGHEST_PROTOCOL)

# Now we compute the activity for each block
# Import Matlab structure containing data and defining inputs
data_mat = sio.loadmat('data/ideal_observer_{}.mat'.format(k_fold+1), struct_as_record=False)

[p1g2_dist, p1g2_mu, p1g2_sd] = neural_proba.import_distrib_param(data_mat)

mu = p1g2_mu[0, :n_stimuli]
sigma = p1g2_sd[:n_stimuli]
dist = p1g2_dist[:, :n_stimuli]
# Creation of a list of simulated distributions
simulated_distrib = [None for k in range(n_stimuli)]
# test = np.zeros(n_stimuli)
#x = np.linspace(0, 1, 1000, endpoint=True)
for k in range(n_stimuli):
    # Normalization of the distribution
    norm_dist = dist[:, k]*(len(dist[1:, k])-1)/np.sum(dist[1:, k])
    simulated_distrib[k] = distrib(mu[k], sigma[k], norm_dist)
    # test[k] = np.max(simulated_distrib[k].beta(x))
    # if np.isinf(test[k]):
    #     print(k)

# Creation of experiment object : all the same for the different dataset
exp = experiment(initial_time, final_time, n_blocks, stimulus_onsets, stimulus_durations, simulated_distrib)

# The amplitudes of the neural signal
true_activity = true_voxel.generate_activity(simulated_distrib, mu_sd, sigma_sd,
                                             use_high_integration_resolution=False)

hrf_model = 'spm'    # No fancy hrf model

# BOLD signal
signal, scan_signal, name, stim = simu_fmri.get_bold_signal(exp, true_activity, hrf_model, fmri_gain)

# Compute the response vector
y[k_fold] = scan_signal
# y = stats.zscore(y)
# Compute the regressors for the selected coding scheme
X[k_fold] = simu_fmri.get_regressor(exp, coding_scheme, true_tc)
# X = stats.zscore(X)

np.save('data/simu/y_{}.npy'.format(true_coding_scheme), y)
np.save('data/simu/X_{}.npy'.format(true_coding_scheme), X)
np.save('data/simu/true_weights_{}'.format(true_coding_scheme), true_voxel.weights)