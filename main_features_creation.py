import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
import numpy as np
#import decimal
# import matplotlib.pyplot as plt

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
tc_type_mean = 'gaussian'    # Tuning curve type
N_mean = 10    # Number of tuning curves
t_mean = 0.05    # The best value from the previous "sum" analysis

# TC related to the uncertainty
tc_type_sigma = 'gaussian'    # Tuning curve type
N_sigma = 10    # Number of tuning curves
t_sigma = 5e-3    # The best value from the previous "sum" analysis

############################################################

# Define the seed to reproduce results from random processes
rand.seed(2);

# Define all the distributional elements from the first distribution

data_mat = sio.loadmat('data/ideal_observer_{}.mat'.format(1), struct_as_record=False)

[p1g2_dist, p1g2_mean, p1g2_sd] = neural_proba.import_distrib_param(data_mat)

n_stimuli = 380
q_mean = p1g2_mean[0, :n_stimuli]
sigma = p1g2_sd[:n_stimuli]
dist = p1g2_dist[:, :n_stimuli]
# Creation of a list of simulated distributions
simulated_distrib = [None for k in range(n_stimuli)]
# test = np.zeros(n_stimuli)
# x = np.linspace(0, 1, 1000, endpoint=True)
for k in range(n_stimuli):
    # Normalization of the distribution
    norm_dist = dist[:, k] * (len(dist[1:, k]) - 1) / np.sum(dist[1:, k])
    simulated_distrib[k] = distrib(q_mean[k], sigma[k], norm_dist)
    # test[k] = np.max(simulated_distrib[k].beta(x))
    # if np.isinf(test[k]):
    #     print(k)

# We find the variance of the data in order to scale equally activity from mean and activity from uncertainty
q_mean_sd = np.std(q_mean)  # Variance of the signal of q_mean's
sigma_sd = np.std(sigma)  # Variance of the signal of sigma's

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mean = 0
tc_upper_bound_mean = 1
tc_lower_bound_sigma = np.min(sigma) - np.std(sigma)
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_sigma = np.max(sigma) + np.std(sigma)

# Creates the tuning_curve object
tc_mean = tuning_curve(tc_type_mean, N_mean, t_mean, tc_lower_bound_mean, tc_upper_bound_mean)

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
    true_tc = [tc_mean, tc_sigma]
elif true_coding_scheme == 'dpc':
    true_tc = [tc_mean]

# if coding_scheme == 'rate':
#     tc = []
# elif coding_scheme == 'ppc':
#     tc = [tc_mean, tc_sigma]
# elif coding_scheme == 'dpc':
#     tc = [tc_mean]

# Creation of the voxel
true_voxel = voxel(true_coding_scheme, true_population_fraction, true_tc)

# Now we compute the activity for each block

y = [None for i in range(4)]
X = [None for i in range(4)]

# Import Matlab structure containing data and defining inputs
for k_fold in range(0, 4):
    data_mat = sio.loadmat('data/ideal_observer_{}.mat'.format(k_fold+1), struct_as_record=False)

    [p1g2_dist, p1g2_mean, p1g2_sd] = neural_proba.import_distrib_param(data_mat)

    q_mean = p1g2_mean[0, :n_stimuli]
    sigma = p1g2_sd[:n_stimuli]
    dist = p1g2_dist[:, :n_stimuli]
    # Creation of a list of simulated distributions
    simulated_distrib = [None for k in range(n_stimuli)]
    # test = np.zeros(n_stimuli)
    #x = np.linspace(0, 1, 1000, endpoint=True)
    for k in range(n_stimuli):
        # Normalization of the distribution
        norm_dist = dist[:, k]*(len(dist[1:, k])-1)/np.sum(dist[1:, k])
        simulated_distrib[k] = distrib(q_mean[k], sigma[k], norm_dist)
        # test[k] = np.max(simulated_distrib[k].beta(x))
        # if np.isinf(test[k]):
        #     print(k)

    # Creation of experiment object : all the same for the different dataset
    exp = experiment(initial_time, final_time, n_blocks, stimulus_onsets, stimulus_durations, simulated_distrib)

    # The amplitudes of the neural signal
    true_activity = true_voxel.generate_activity(simulated_distrib, q_mean_sd, sigma_sd,
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