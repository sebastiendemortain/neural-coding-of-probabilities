# Import useful modules
import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
from scipy.stats.stats import pearsonr
import numpy as np
#import decimal
# import matplotlib
# matplotlib.use('Agg')    # To avoid bugs
#import matplotlib.pyplot as plt
#import matplotlib
width = 18
height = 16
#matplotlib.rcParams['figure.figsize'] = [width, height]
import pandas as pd

import pickle
import itertools
import time

import copy

import multiprocessing as mp

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import neural_proba
from neural_proba import distrib
from neural_proba import tuning_curve
from neural_proba import voxel
from neural_proba import experiment
from neural_proba import fmri

import utils

# All parameters are here

# Define the seed to reproduce results from random processes
rand.seed(5);

# INPUTS

# The parameters related to the scheme
scheme_array = ['gaussian_ppc', 'sigmoid_ppc', 'gaussian_dpc', 'sigmoid_dpc']
n_schemes = len(scheme_array)

# The parameters related to the tuning curves to be explored
N_array = np.array([2, 4, 6, 8, 10, 14, 20])

t_mu_gaussian_array = np.array([0.15, 0.1, 7e-2, 5e-2, 4e-2, 3e-2, 2e-2])
t_conf_gaussian_array = np.array([0.25, 0.15, 0.10, 8e-2, 6e-2, 4e-2, 3e-2])

t_mu_sigmoid_array = np.sqrt(2*np.pi)/4*t_mu_gaussian_array
t_conf_sigmoid_array = np.sqrt(2*np.pi)/4*t_conf_gaussian_array

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mu = 0
tc_upper_bound_mu = 1
tc_lower_bound_conf = 1.1
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_conf = 2.6

# The number of N to be tested
n_N = len(N_array)

# The number of fractions tested (related to W)
n_fractions = 1335#

# Sparsity exponents
sparsity_exp_array = np.array([1, 2, 4, 8])
n_sparsity_exp = len(sparsity_exp_array)

# The number of subjects
n_subjects = 20

# The number of sessions
n_sessions = 4

# The number of stimuli per session
n_stimuli = 380

# Way to compute the distributions from the sequence
distrib_type = 'HMM'

# Load the corresponding data
[p1g2_dist_array, p1g2_mu_array, p1g2_sd_array] = neural_proba.import_distrib_param(n_subjects, n_sessions, n_stimuli,
                                                                                      distrib_type)
# SNR as defined by ||signal||²/(||signal||²+||noise||²)
snr = 0.1

# Type of regression
regr = linear_model.LinearRegression(n_jobs=-1)

# # Load the design matrices and specify their size
# with open("output/design_matrices/X_20sub_f.txt", "rb") as fp: #X_20sub_f.txt", "rb") as fp:   # Unpickling
#     X = pickle.load(fp)

#
fmri_gain = 1    # Amplification of the signal
# Just for now
n_subjects = 3
n_sessions = 4
n_N = len(N_array)
n_schemes = 1

# Initialization of the design matrices and their zscore versions
X = [[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fit_N in range(n_N)]
     for k_fit_scheme in range(n_schemes)]

### WE BEGIN BY CREATING THE DESIGN MATRIX X
start = time.time()

for k_subject in range(n_subjects):
    ### Loop over the sessions : we start with it in order to have the same length whatever N_fit is
    for k_session in range(n_sessions):
        # Get the data of interest
        mu = p1g2_mu_array[k_subject][k_session][0, :n_stimuli]
        sigma = p1g2_sd_array[k_subject][k_session][0, :n_stimuli]
        conf = -np.log(p1g2_sd_array[k_subject][k_session][0, :n_stimuli])
        dist = p1g2_dist_array[k_subject][k_session][:, :n_stimuli]

        # Formatting
        simulated_distrib = [None for k in range(n_stimuli)]
        for k in range(n_stimuli):
            # Normalization of the distribution
            norm_dist = dist[:, k] * (len(dist[1:, k]) - 1) / np.sum(dist[1:, k])
            simulated_distrib[k] = distrib(mu[k], sigma[k], norm_dist)

        # Experimental design information
        eps = 1e-5  # For floating points issues

        between_stimuli_duration = 1.3
        initial_time = between_stimuli_duration + eps
        final_time_tmp = between_stimuli_duration * (n_stimuli + 1) + eps
        # Every 15+-3 trials : one interruption of 8-12s
        stimulus_onsets = np.linspace(initial_time, final_time_tmp, n_stimuli)
        # We add some time to simulate breaks
        stimulus = 0

        while True:
            # Number of regularly spaced stimuli
            n_local_regular_stimuli = rand.randint(12, 18)
            stimulus_shifted = stimulus + n_local_regular_stimuli  # Current stimulus before the break
            if stimulus_shifted > n_stimuli:  # The next break is supposed to occur after all stimuli are shown
                break
            stimulus_onsets[stimulus_shifted:] += rand.randint(8,
                                                               12) - between_stimuli_duration  # We consider a break of 8-12s
            stimulus = stimulus_shifted

        dt = 0.125  # Temporal resolution of the fMRI scanner

        stimulus_durations = dt * np.ones_like(stimulus_onsets)  # Dirac-like stimuli

        # fMRI information
        final_time = stimulus_onsets[-1]
        final_frame_offset = 10  # Frame recording duration after the last stimulus has been shown
        initial_frame_time = 0
        final_frame_time = final_time + final_frame_offset

        between_scans_duration = 2  # in seconds
        final_scan_offset = 10  # Scan recording duration after the last stimulus has been shown
        initial_scan_time = initial_frame_time + between_scans_duration
        final_scan_time = final_time + final_scan_offset
        scan_times = np.arange(initial_scan_time, final_scan_time, between_scans_duration)

        # Creation of fmri object
        simu_fmri = fmri(initial_frame_time, final_frame_time, dt, scan_times)

        # Creation of experiment object
        exp = experiment(initial_time, final_time, n_sessions, stimulus_onsets, stimulus_durations, simulated_distrib)

        ### LOOP OVER THE SCHEME
        for k_fit_scheme in range(n_schemes):

            #k_fit_scheme=0

            # Current schemes
            fit_scheme = scheme_array[k_fit_scheme]

            ### LOOP OVER THE FIT N's
            for k_fit_N in range(n_N):
                # k_fit_N=0
                # k_true_N=0

                # Current N
                fit_N = N_array[k_fit_N]

                # Creation of the true tuning curve objects

                # We replace the right value of the "t"'s according to the type of tuning curve and the N
                if fit_scheme.find('gaussian') != -1:
                    fit_t_mu = t_mu_gaussian_array[k_fit_N]
                    fit_t_conf = t_conf_gaussian_array[k_fit_N]
                    fit_tc_type = 'gaussian'

                elif fit_scheme.find('sigmoid') != -1:
                    fit_t_mu = t_mu_sigmoid_array[k_fit_N]
                    fit_t_conf = t_conf_sigmoid_array[k_fit_N]
                    fit_tc_type = 'sigmoid'

                fit_tc_mu = tuning_curve(fit_tc_type, fit_N, fit_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
                fit_tc_conf = tuning_curve(fit_tc_type, fit_N, fit_t_conf, tc_lower_bound_conf,
                                             tc_upper_bound_conf)

                if fit_scheme.find('ppc') != -1:
                    fit_tc = [fit_tc_mu, fit_tc_conf]
                elif fit_scheme.find('dpc') != -1:
                    fit_tc = [fit_tc_mu]
                elif fit_scheme.find('rate') != -1:
                    fit_tc = []

                # Regressor and BOLD computation
                X[k_fit_scheme][k_fit_N][k_subject][k_session] = simu_fmri.get_regressor(exp, fit_scheme, fit_tc)
                # Just to have Xz with np array of the right structure
    end = time.time()
    print('Design matrix creation : Subject n'+str(k_subject)+' is done ! Time elapsed : '+str(end-start)+'s')

# # Save this matrix
# with open("output/design_matrices/X_par.txt", "wb") as fp:   #Pickling
#     pickle.dump(X, fp)
#
#
# # Whiten the design matrices
#
# # Whitening matrix
# white_mat = sio.loadmat('data/simu/whitening_matrix.mat')
# W = white_mat['W']
# # Complete the in-between session "holes"
# W[300:600, 300:600] = W[20:320, 20:320]
# whitening_done = False
#
# if not whitening_done:
#     # Multiplying the zscored X with the whitening matrix
#     for k_scheme, k_fit_N, k_subject, k_session in itertools.product(range(n_schemes), range(n_N), range(n_subjects), range(n_sessions)):
#         X_tmp = copy.deepcopy(X[k_scheme][k_fit_N][k_subject][k_session])    # Just to lighten code
#         rows_dim, columns_dim = X_tmp.shape
#         X_tmp = np.matmul(W[:rows_dim, :rows_dim], X_tmp)
#         X[k_scheme][k_fit_N][k_subject][k_session] = copy.deepcopy(X_tmp)
#
# whitening_done = True
#
# X_after_whitening = copy.deepcopy(X)
#
# # Creation of y from X to save computational resources
# # Initialization of the response vectors
# y = [[[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fraction in range(n_fractions)]
# for k_true_N in range(n_N)] for k_scheme in range(n_schemes)]
#
# # Initialization of the weights
# weights = [[[[None for k_subject in range(n_subjects)] for k_fraction in range(n_fractions)] for k_true_N in range(n_N)]
#            for k_scheme in range(n_schemes)]
#
#
# ### LOOP OVER THE SCHEME
# for k_scheme in range(n_schemes):
#     true_scheme = scheme_array[k_scheme]
#
#     # We replace the right value of the "t"'s according to the type of tuning curve
#
#     if true_scheme.find('gaussian') != -1:
#         true_t_mu_array = copy.deepcopy(t_mu_gaussian_array)
#         true_t_conf_array = copy.deepcopy(t_conf_gaussian_array)
#         true_tc_type = 'gaussian'
#
#     elif true_scheme.find('sigmoid') != -1:
#         true_t_mu_array = copy.deepcopy(t_mu_sigmoid_array)
#         true_t_conf_array = copy.deepcopy(t_conf_sigmoid_array)
#         true_tc_type = 'sigmoid'
#
#     # We consider combinations of population fractions for PPC and rate codes
#     if true_scheme.find('ppc') != -1 or true_scheme.find('rate') != -1:
#         # The number of population fraction tested (related to W)
#         population_fraction_array = copy.deepcopy(np.array([[0.5, 0.5], [0.25, 0.75], [0, 1], [0.75, 0.25], [1, 0]]))
#     elif true_scheme.find('dpc') != -1:  # DPC case
#         population_fraction_array = copy.deepcopy(np.array([[1]]))
#     n_population_fractions = len(population_fraction_array)
#
#     ### LOOP OVER N_true
#     for k_true_N in range(n_N):
#         true_N = N_array[k_true_N]
#         # Creation of the true tuning curve objects
#         true_t_mu = true_t_mu_array[k_true_N]
#         true_t_conf = true_t_conf_array[k_true_N]
#         true_tc_mu = tuning_curve(true_tc_type, true_N, true_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
#         true_tc_conf = tuning_curve(true_tc_type, true_N, true_t_conf, tc_lower_bound_conf,
#                                      tc_upper_bound_conf)
#
#         if true_scheme.find('ppc') != -1:
#             true_tc = [true_tc_mu, true_tc_conf]
#         elif true_scheme.find('dpc') != -1:
#             true_tc = [true_tc_mu]
#         elif true_scheme.find('rate') != -1:
#             true_tc = []
#
#         ### LOOP OVER THE W's
#         # The number of subpopulation fractions acc. to the scheme
#         n_subpopulation_fractions = int(n_fractions / n_population_fractions)
#         fraction_counter = 0
#         for k_subpopulation_fraction in range(n_subpopulation_fractions):
#             for k_population_fraction, population_fraction in enumerate(population_fraction_array):
#                 # The number of populations acc. to the scheme (2 for PPC and rate, 1 for DPC)
#                 n_population = len(population_fraction)
#                 if true_scheme.find('ppc') != -1 or true_scheme.find('dpc') != -1:
#                     # We consider one sparsity per remainder value of the counter divided by the number
#                     # of combinations to be tested
#                     subpopulation_sparsity_exp = sparsity_exp_array[fraction_counter % n_sparsity_exp]
#                     # Fraction of each neural subpopulation
#                     subpopulation_fraction = neural_proba.get_subpopulation_fraction(n_population, true_N,
#                                                                                      subpopulation_sparsity_exp)
#                 else:  # Rate case
#                     population_fraction = np.array([1, 1])
#
#                 # Generate the data from the voxel
#                 true_voxel = voxel(true_scheme, population_fraction, subpopulation_fraction, true_tc)
#                 n_true_features = n_population * true_N
#                 weights_tmp = copy.deepcopy(np.reshape(true_voxel.weights, (n_true_features,)))
#
#                 ### LOOP OVER THE SUBJECTS
#                 for k_subject in range(n_subjects):
#                     # Allocation of the weight tensor
#                     weights[k_scheme][k_true_N][fraction_counter][k_subject] \
#                         = copy.deepcopy(weights_tmp)
#
#                     ### LOOP OVER THE SESSIONS : simulating the response
#                     for k_session in range(n_sessions):
#                             # We use X to compute y order to save some computation time
#                             # Temporary variables to lighten the reading
#                             X_tmp = copy.deepcopy(X[k_scheme][k_true_N][k_subject][k_session])
#                             y_tmp = copy.deepcopy(np.matmul(X_tmp, weights_tmp))
#
#                             # Allocation of the tensor
#                             y[k_scheme][k_true_N][fraction_counter][k_subject][
#                                 k_session] = copy.deepcopy(y_tmp)
#
#
#                 fraction_counter += 1
#
# y_without_noise = copy.deepcopy(y)
# # with open("output/design_matrices/y_20sub.txt", "wb") as fp:   #Pickling
# #     pickle.dump(y, fp)
#
# # Compute the amplitude of the noise
# noise_sd = np.zeros((n_schemes, n_N))
# added_noise = np.zeros((n_schemes, n_N, 1000))
# for k_scheme, k_true_N in itertools.product(range(n_schemes), range(n_N)):
#     all_y = np.asarray(y[k_scheme][k_true_N]).flatten()  # Concatenation of all y grouped together for SNR computation
#     # print(all_y[0])
#     noise_sd[k_scheme, k_true_N] = np.sqrt(np.var(all_y[0]) * (1 / snr - 1))  # std of the added gaussian noise
#     added_noise[k_scheme, k_true_N, :] = np.random.normal(0, noise_sd[k_scheme, k_true_N], 1000)
#     del all_y    # Free memory
#
# # Compute the amplitude of the noise
# for k_scheme, k_true_N, k_fraction, k_subject, k_session in itertools.product(range(n_schemes), range(n_N),
#                                                                               range(n_fractions), range(n_subjects),
#                                                                               range(n_sessions)):
#     y[k_scheme][k_true_N][k_fraction][k_subject][k_session] = y[k_scheme][k_true_N][k_fraction][k_subject][
#                                                                   k_session] + added_noise[k_scheme, k_true_N, :len(
#         y[k_scheme][k_true_N][k_fraction][k_subject][k_session])]
#
# # y_with_noise = copy.deepcopy(y)
#
# # Create the filtering design matrices and filters out the response
#
# for k_scheme, k_true_N, k_fraction, k_subject, k_sessions in itertools.product(range(n_schemes), range(n_N), range(n_fractions), range(n_subjects), range(n_sessions)):
#     y_tmp = copy.deepcopy(y[k_scheme][k_true_N][k_fraction][k_subject][k_session])
#     N = len(y_tmp)    # Resolution of the signal
#     K = 11    # Highest order of the filter
#     n_grid = np.linspace(0, N-1, N, endpoint=True)    # 1D grid over values
#     k_grid = np.linspace(2, K, K-1, endpoint=True)    # 1D grid over orders
#     X_filter = np.zeros((N, K-1))
#     for kk, k in enumerate(k_grid):
#         X_filter[:, kk] = np.sqrt(2/N) * np.cos(np.pi*(2*n_grid+1)/(2*N)*(k-1))
#     y_tmp = copy.deepcopy(y_tmp - np.matmul(np.matmul(X_filter, np.transpose(X_filter)), y_tmp))    # Regression
#     y[k_scheme][k_true_N][k_fraction][k_subject][k_session] = copy.deepcopy(y_tmp)
#
# y_after_filtering = copy.deepcopy(y)
#
# # # To visualize the matrix
# # k_scheme = 0
# # k_true_N = 2
# # k_fraction = 18
# # k_subject = 0
# # k_session = 3
#
# # fig = plt.figure(figsize=(10, 10))
# # ax = fig.add_subplot(111)
# # plt.imshow(X_filter, cmap=plt.cm.ocean, extent=[2, K, N-1,0], aspect='auto')
# # plt.colorbar()
# # plt.show()
#
# # Z-scoring of X and y
# # Initialization
# Xz = [[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fit_N in range(n_N)]
#       for k_scheme in range(n_schemes)]
#
# X_sd_array = [[[None for k_subject in range(n_subjects)] for k_fit_N in range(n_N)]
#               for k_scheme in range(n_schemes)]
#
# yz = [[[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fraction in
#         range(n_fractions)]
#        for k_true_N in range(n_N)] for k_scheme in range(n_schemes)]
# yz_without_noise = [[[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fraction in
#                       range(n_fractions)]
#                      for k_true_N in range(n_N)] for k_scheme in range(n_schemes)]
#
# y_sd_array = [[[[None for k_subject in range(n_subjects)] for k_fraction in range(n_fractions)]
#                for k_true_N in range(n_N)] for k_scheme in range(n_schemes)]
#
# for k_scheme, k_fit_N, k_subject, k_session in itertools.product(range(n_schemes), range(n_N), range(n_subjects),
#                                                                  range(n_sessions)):
#     Xz[k_scheme][k_fit_N][k_subject][k_session] = np.zeros_like(X[k_scheme][k_fit_N][k_subject][k_session])
#
# # Manual Z-scoring of regressors inside the session
# for k_scheme, k_fit_N, k_subject in itertools.product(range(n_schemes), range(n_N), range(n_subjects)):
#     n_fit_features = len(X[k_scheme][k_fit_N][k_subject][0][0])
#     X_mean = np.mean(np.concatenate(X[k_scheme][k_fit_N][k_subject], axis=0), axis=0)
#     X_sd = np.std(np.concatenate(X[k_scheme][k_fit_N][k_subject], axis=0), axis=0)
#     X_sd_array[k_scheme][k_fit_N][k_subject] = copy.deepcopy(X_sd)
#     for k_session in range(n_sessions):
#         for feature in range(n_fit_features):
#             Xz[k_scheme][k_fit_N][k_subject][k_session][:, feature] \
#                 = (copy.deepcopy(X[k_scheme][k_fit_N][k_subject][k_session][:, feature]) - X_mean[
#                 feature] * np.ones_like(
#                 X[k_scheme][k_fit_N][k_subject][k_session][:, feature])) / X_sd[feature]  # Centering + Standardization
#     # End of z-scoring
#
# for k_scheme, k_true_N, k_fraction, k_subject in itertools.product(range(n_schemes), range(n_N), range(n_fractions),
#                                                                    range(n_subjects)):
#     # Z-scoring of y
#     y_mean = np.mean(np.concatenate(np.asarray(y[k_scheme][k_true_N][k_fraction][k_subject]),
#                                     axis=0), axis=0)
#     y_sd = np.std(np.concatenate(np.asarray(y[k_scheme][k_true_N][k_fraction][k_subject]),
#                                  axis=0))
#     y_sd_array[k_scheme][k_true_N][k_fraction][k_subject] = y_sd
#
#     for k_session in range(n_sessions):
#         yz[k_scheme][k_true_N][k_fraction][k_subject][k_session] = \
#             (copy.deepcopy(y[k_scheme][k_true_N][k_fraction][k_subject][
#                                k_session]) - y_mean) / y_sd  # Centering and standardization
#         yz_without_noise[k_scheme][k_true_N][k_fraction][k_subject][k_session] = \
#             (copy.deepcopy(y_without_noise[k_scheme][k_true_N][k_fraction][k_subject][
#                                k_session]) - y_mean) / y_sd  # Centering and standardization
#
#     ### End of z-scoring of y
#
# # Reajusting the weights after zscoring
# for k_scheme, k_true_N, k_fraction, k_subject in itertools.product(range(n_schemes), range(n_N), range(n_fractions),
#                                                                    range(n_subjects)):
#     for feature in range(weights[k_scheme][k_true_N][k_fraction][k_subject].shape[0]):
#         weights[k_scheme][k_true_N][k_fraction][k_subject][feature] = \
#         weights[k_scheme][k_true_N][k_fraction][k_subject][feature] * X_sd_array[k_scheme][k_true_N][k_subject][
#             feature] / y_sd_array[k_scheme][k_true_N][k_fraction][k_subject]
