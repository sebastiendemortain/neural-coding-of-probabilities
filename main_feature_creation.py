import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
import numpy as np
#import decimal
# import matplotlib
# matplotlib.use('Agg')    # To avoid bugs
# import matplotlib.pyplot as plt
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

# Define the seed to reproduce results from random processes
rand.seed(5);

# INPUTS

# The parameters related to the scheme
scheme_array = ['gaussian_ppc', 'sigmoid_ppc', 'gaussian_dpc', 'sigmoid_dpc']
n_schemes = len(scheme_array)

# The parameters related to the tuning curves to be explored
N_array = np.array([6, 8, 10, 14, 16, 20])
t_mu_sigmoid_array = np.array([9e-2, 6e-2, 5e-2, 4e-2, 3e-2, 3e-2])
t_sigma_sigmoid_array = np.array([3e-2, 2e-2, 2e-2, 1.5e-2, 1e-2, 8e-3])
t_mu_gaussian_array = np.array([1e-1, 8e-2, 5e-2, 3.5e-2, 3e-2, 2.5e-2])
t_sigma_gaussian_array = np.array([3e-2, 2e-2, 1.7e-2, 1.2e-2, 1e-2, 8e-3])

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mu = 0
tc_upper_bound_mu = 1
tc_lower_bound_sigma = 0.04
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_sigma = 0.35

# The number of N to be tested
n_N = len(N_array)

# The number of population fraction tested (related to W)
n_population_fraction = 1

# The number of subpopulation fraction tested (related to W)
n_subpopulation_fraction = 1

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
# Just for now
<<<<<<< HEAD
n_subjects = 8
n_sessions = 4
n_N = 3
n_schemes = 4
=======
#_subjects = 20
#n_sessions = 4
#n_N = len(N_array)
#n_schemes =
>>>>>>> 3f7a39672611c1f6f669a59d753b483185a62327

# Experimental design information
eps = 1e-5  # For floating points issues

between_stimuli_duration = 1.3
initial_time = between_stimuli_duration + eps
final_time = between_stimuli_duration * (n_stimuli + 1) + eps
stimulus_onsets = np.linspace(initial_time, final_time, n_stimuli) # tous les 15 essais +-3 essais : une interruption de 8-12s
stimulus_durations = 0.01 * np.ones_like(stimulus_onsets)  # Dirac-like stimuli

# fMRI information

final_frame_offset = 10  # Frame recording duration after the last stimulus has been shown
initial_frame_time = 0
final_frame_time = final_time + final_frame_offset
dt = 0.01  # Temporal resolution of the fMRI scanner

between_scans_duration = 2  # in seconds
final_scan_offset = 10  # Scan recording duration after the last stimulus has been shown
initial_scan_time = initial_frame_time + between_scans_duration
final_scan_time = final_time + final_scan_offset
scan_times = np.arange(initial_scan_time, final_scan_time, between_scans_duration)

# Creation of fmri object
simu_fmri = fmri(initial_frame_time, final_frame_time, dt, scan_times)
frame_times = simu_fmri.frame_times
hrf_model = 'spm'    # No fancy hrf model
fmri_gain = 1    # Amplification of the signal
noise_coeff = 0

# The quantity to be computed during the cross validation
r2 = np.zeros((n_schemes, n_N, n_N, n_population_fraction, n_subpopulation_fraction, n_subjects, n_sessions))

# Cross-validation parameters
n_train = n_sessions - 1
n_test = 1

# Initialization of the design matrices and their zscore versions
X = [[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fit_N in range(n_N)]
     for k_fit_scheme in range(n_schemes)]

### WE BEGIN BY CREATING THE DESIGN MATRIX X

def X_creation(k_subject):
# , scheme_array=scheme_array, n_schemes=n_schemes, N_array=N_array, n_N=n_N,
#                t_mu_gaussian_array=t_mu_gaussian_array, t_sigma_gaussian_array=t_sigma_gaussian_array,
#                t_mu_sigmoid_array=t_mu_sigmoid_array, t_sigma_sigmoid_array=t_sigma_sigmoid_array,
#                tc_lower_bound_mu=tc_lower_bound_mu, tc_upper_bound_mu=tc_upper_bound_mu,
#                tc_lower_bound_sigma=tc_lower_bound_sigma, tc_upper_bound_sigma=tc_upper_bound_sigma,
#                n_sessions=n_sessions, p1g2_mu_array=p1g2_mu_array, p1g2_sd_array=p1g2_sd_array,
#                p1g2_dist_array=p1g2_dist_array, initial_time=initial_time, final_time=final_time,
#                stimulus_onsets=stimulus_onsets, stimulus_durations=stimulus_durations, X=X, Xz=Xz):

    '''Creation of X per subject'''
    X_tmp = [[[None for k_session in range(n_sessions)] for k_fit_N in range(n_N)] for k_fit_scheme in range(n_schemes)]

    # for k_subject in range(n_subjects):
    # k_subject = 0
    start = time.time()
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
                fit_t_sigma = t_sigma_gaussian_array[k_fit_N]
                fit_tc_type = 'gaussian'

            elif fit_scheme.find('sigmoid') != -1:
                fit_t_mu = t_mu_sigmoid_array[k_fit_N]
                fit_t_sigma = t_sigma_sigmoid_array[k_fit_N]
                fit_tc_type = 'sigmoid'

            fit_tc_mu = tuning_curve(fit_tc_type, fit_N, fit_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
            fit_tc_sigma = tuning_curve(fit_tc_type, fit_N, fit_t_sigma, tc_lower_bound_sigma,
                                         tc_upper_bound_sigma)

            if fit_scheme.find('ppc') != -1:
                fit_tc = [fit_tc_mu, fit_tc_sigma]
            elif fit_scheme.find('dpc') != -1:
                fit_tc = [fit_tc_mu]
            elif fit_scheme.find('rate') != -1:
                fit_tc = []

            ### FIRST LOOP OVER THE SESSIONS : simulating the regressors for this subject
            for k_session in range(n_sessions):
                # k_session = 0

                # Get the data of interest
                mu = p1g2_mu_array[k_subject][k_session][0, :n_stimuli]
                sigma = p1g2_sd_array[k_subject][k_session][0, :n_stimuli]
                dist = p1g2_dist_array[k_subject][k_session][:, :n_stimuli]

                # Formatting
                simulated_distrib = [None for k in range(n_stimuli)]
                for k in range(n_stimuli):
                    # Normalization of the distribution
                    norm_dist = dist[:, k]*(len(dist[1:, k])-1)/np.sum(dist[1:, k])
                    simulated_distrib[k] = distrib(mu[k], sigma[k], norm_dist)

                # Creation of experiment object
                exp = experiment(initial_time, final_time, n_sessions, stimulus_onsets, stimulus_durations, simulated_distrib)

                # Regressor and BOLD computation
                X_tmp[k_fit_scheme][k_fit_N][k_session] = simu_fmri.get_regressor(exp, fit_scheme, fit_tc)
                # Just to have Xz with np array of the right structure

    end = time.time()
    print('Design matrix creation : Subject n'+str(k_subject)+' is done ! Time elapsed : '+str(end-start)+'s')
    return X_tmp

### LOOP OVER THE SUBJECTS
# Parallelization
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())  # Create a multiprocessing Pool
    X_tmp = pool.map(X_creation, range(n_subjects))  # proces inputs iterable with pool

### WE JUST END THE LOOP TO CREATE MATRICES X and end initializing the zscore version
for k_fit_scheme, k_fit_N, k_subject, k_session in itertools.product(range(n_schemes), range(n_N), range(n_subjects), range(n_sessions)):
    X[k_fit_scheme][k_fit_N][k_subject][k_session] = copy.deepcopy(X_tmp[k_subject][k_fit_scheme][k_fit_N][k_session])

<<<<<<< HEAD
# Save this matrix
with open("output/design_matrices/X_par.txt", "wb") as fp:   #Pickling
    pickle.dump(X, fp)

# with open("output/design_matrices/Xz_all.txt", "wb") as fpz:  # Pickling
=======
# # Save these matrices
with open("output/design_matrices/X_par.txt", "wb") as fp:   #Pickling
    pickle.dump(X, fp)
#     with open("output/design_matrices/Xz_all.txt", "wb") as fpz:  # Pickling
>>>>>>> 3f7a39672611c1f6f669a59d753b483185a62327
#     pickle.dump(Xz, fpz)

# # Load the design matrices
# with open("output/design_matrices/X.txt", "rb") as fp:   # Unpickling
<<<<<<< HEAD
#     X = pickle.load(fp)
#
# with open("output/design_matrices/Xz.txt", "rb") as fpz:   # Unpickling
#     Xz = pickle.load(fpz)

# ### LOOP OVER THE SCHEME
=======
#    X0 = pickle.load(fp)
#
# with open("output/design_matrices/Xz.txt", "rb") as fpz:   # Unpickling
#     Xz = pickle.load(fpz)
a=1
### LOOP OVER THE SCHEME
>>>>>>> 3f7a39672611c1f6f669a59d753b483185a62327
# for k_fit_scheme in range(n_schemes):
#
#     # k_fit_scheme=0
#     k_true_scheme = k_fit_scheme
#
#     # Current schemes
#     fit_scheme = scheme_array[k_fit_scheme]
#     true_scheme = scheme_array[k_true_scheme]
#
#     # We replace the right value of the "t"'s according to the type of tuning curve
#     if fit_scheme.find('gaussian') != -1:
#         fit_t_mu_array = t_mu_gaussian_array
#         fit_t_sigma_array = t_sigma_gaussian_array
#         fit_tc_type = 'gaussian'
#
#     elif fit_scheme.find('sigmoid') != -1:
#         fit_t_mu_array = t_mu_sigmoid_array
#         fit_t_sigma_array = t_sigma_sigmoid_array
#         fit_tc_type = 'sigmoid'
#
#     if true_scheme.find('gaussian') != -1:
#         true_t_mu_array = t_mu_gaussian_array
#         true_t_sigma_array = t_sigma_gaussian_array
#         true_tc_type = 'gaussian'
#
#     elif true_scheme.find('sigmoid') != -1:
#         true_t_mu_array = t_mu_sigmoid_array
#         true_t_sigma_array = t_sigma_sigmoid_array
#         true_tc_type = 'sigmoid'
#
#     ### LOOP OVER THE FIT N's
#     for k_fit_N in range(n_N):
#         # k_fit_N=0
#         # k_true_N=0
#
#         # Current N
#         fit_N = N_array[k_fit_N]
#
#         # Creation of the true tuning curve objects
#         fit_t_mu = fit_t_mu_array[k_fit_N]
#         fit_t_sigma = fit_t_sigma_array[k_fit_N]
#         fit_tc_mu = tuning_curve(fit_tc_type, fit_N, fit_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
#         fit_tc_sigma = tuning_curve(fit_tc_type, fit_N, fit_t_sigma, tc_lower_bound_sigma,
#                                     tc_upper_bound_sigma)
#
#         if fit_scheme.find('ppc') != -1:
#             fit_tc = [fit_tc_mu, fit_tc_sigma]
#         elif fit_scheme.find('dpc') != -1:
#             fit_tc = [fit_tc_mu]
#         elif fit_scheme.find('rate') != -1:
#             fit_tc = []
#
#         ### LOOP OVER THE SUBJECTS
#         for k_subject in range(n_subjects):
#             # k_subject = 0
#
#             ### LOOP OVER N_true
#             for k_true_N in range(n_N):
#                 true_N = N_array[k_true_N]
#                 # Creation of the true tuning curve objects
#                 true_t_mu = true_t_mu_array[k_true_N]
#                 true_t_sigma = true_t_sigma_array[k_true_N]
#                 true_tc_mu = tuning_curve(true_tc_type, true_N, true_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
#                 true_tc_sigma = tuning_curve(true_tc_type, true_N, true_t_sigma, tc_lower_bound_sigma,
#                                              tc_upper_bound_sigma)
#
#                 if true_scheme.find('ppc') != -1:
#                     true_tc = [true_tc_mu, true_tc_sigma]
#                 elif true_scheme.find('dpc') != -1:
#                     true_tc = [true_tc_mu]
#                 elif true_scheme.find('rate') != -1:
#                     true_tc = []
#
#                 ### LOOP OVER THE W's
#                 for k_population_fraction, k_subpopulation_fraction in itertools.product(range(n_population_fraction),
#                                                                                          range(n_subpopulation_fraction)):
#
#                     # k_population_fraction = 0
#                     # k_subpopulation_fraction = 0
#
#                     # Current weights
#                     # Fraction of each neural population
#                     population_fraction = neural_proba.get_population_fraction(true_scheme)
#                     n_population = len(population_fraction)
#
#                     # Fraction of each neural subpopulation
#                     # No rate coding here
#                     subpopulation_fraction = neural_proba.get_subpopulation_fraction(n_population, true_N)
#
#                     # Generate the data from the voxel
#                     true_voxel = voxel(true_scheme, population_fraction, subpopulation_fraction,
#                                        [true_tc_mu, true_tc_sigma])
#                     n_true_features = n_population * true_N
#                     weights_tmp = np.reshape(true_voxel.weights, (n_true_features,))
#
#                     ### LOOP OVER THE SESSIONS : simulating the response
#                     for k_session in range(n_sessions):
#                             # We use X to compute y order to save some computation time
#                             # Temporary variables to lighten the reading
#                             X_tmp = X[k_true_scheme][k_true_N][k_subject][k_session]
#                             y_tmp = np.dot(X_tmp, weights_tmp)
#                             # Noise injection
#                             y_tmp = y_tmp + noise_coeff*np.random.normal(0, 1, len(y_tmp))
#
#                             # Allocation of the tensor
#                             y[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N][k_subject][
#                                 k_session] = y_tmp
#
#                         # else:    # We need to compute the response independently from the regressors
#                         #     # Get the data of interest
#                         #     mu = p1g2_mu_array[k_subject][k_session][0, :n_stimuli]
#                         #     sigma = p1g2_sd_array[k_subject][k_session][0, :n_stimuli]
#                         #     dist = p1g2_dist_array[k_subject][k_session][:, :n_stimuli]
#                         #
#                         #     # Formatting
#                         #     simulated_distrib = [None for k in range(n_stimuli)]
#                         #     for k in range(n_stimuli):
#                         #         # Normalization of the distribution
#                         #         norm_dist = dist[:, k] * (len(dist[1:, k]) - 1) / np.sum(dist[1:, k])
#                         #         simulated_distrib[k] = distrib(mu[k], sigma[k], norm_dist)
#                         #
#                         #     # Creation of experiment object
#                         #     exp = experiment(initial_time, final_time, n_sessions, stimulus_onsets, stimulus_durations,
#                         #                      simulated_distrib)
#                         #
#                         #     true_activity = true_voxel.generate_activity(simulated_distrib)
#                         #     signal, scan_signal, name, stim = simu_fmri.get_bold_signal(exp, true_activity, hrf_model, 1)
#                         #     y[k_session] = scan_signal
#
#                     # Z-scoring of y
#                     y_mean = np.mean(np.concatenate(np.asarray(y[k_true_scheme][k_population_fraction]
#                                                                [k_subpopulation_fraction][k_true_N][k_subject]),
#                                                     axis=0))
#                     y_sd = np.std(np.concatenate(np.asarray(y[k_true_scheme][k_population_fraction]
#                                                                [k_subpopulation_fraction][k_true_N][k_subject]),
#                                                  axis=0))
#                     # Allocation of the weight tensor
#                     weights[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N] \
#                         = weights_tmp
#
#                     for k_session in range(n_sessions):
#                         yz[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N][k_subject]
#                         [k_session] = y[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N]
#                         [k_subject][k_session] - y_mean    # Centering
#                         yz[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N][k_subject]
#                         [k_session] = y[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N]
#                         [k_subject][k_session]/y_sd    # Standardization
#
#                     ### End of z-scoring of y
#
<<<<<<< HEAD
#
#                     ### BEGINNING OF THIRS LOOP OVER SESSIONS (CROSS-VALIDATION)
#                     # Current cross-validation matrice and response
#                     X_cv = Xz[k_fit_scheme][k_fit_N][k_subject]
#                     y_cv = yz[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N][k_subject]
#                     for k_session in range(n_sessions):
#                         X_train = np.concatenate(np.asarray(X_cv[:k_session]+Xz[k_session+1:]), axis=0)
#                         y_train = np.concatenate(np.asarray(y_cv[:k_session]+y_cv[k_session+1:]), axis=0)
#                         X_test = X_cv[k_session]
#                         y_test = y_cv[k_session]
#                         # Create linear regression object
#                         regr = linear_model.LinearRegression(fit_intercept=True)
#
#                         # Train the model using the training set
#                         regr.fit(X_train, y_train)
#                         # Make predictions using the testing set
#                         y_pred = regr.predict(X_test)
#                         # mse[block] = mean_squared_error(y_test, y_pred)
#                         # Updates the big tensor
#                         r2[k_fit_scheme, k_fit_N, k_true_N, k_population_fraction, k_subpopulation_fraction, k_subject, k_session] \
#                             = r2_score(y_test, y_pred)
#                         # print('R2 = '+str(r2_score(y_test, y_pred)))
#                         # # The coefficients
#                         # print('True coefficients: \n', true_voxel.weights)
#                         # print('Fitted coefficients: \n', regr.coef_)
#                         with open("output/results/Output.txt", "w") as text_file:
#                             text_file.write('k_fit_scheme={} k_fit_N={} k_true_N={} k_subject={} k_population_fraction={} k_subpopulation_fraction={} k_session={} \nr2={} \n'.format(
#                                 k_fit_scheme, k_fit_N, k_true_N, k_subject, k_population_fraction, k_subpopulation_fraction, k_session,
#                                 r2_score(y_test, y_pred)))
#                         print('Completed : k_fit_scheme={} k_fit_N={} k_true_N={} k_subject={} k_population_fraction={} k_subpopulation_fraction={} k_session={} \nr2={} \n'.format(
#                                 k_fit_scheme, k_fit_N, k_true_N, k_subject, k_population_fraction, k_subpopulation_fraction, k_session,
#                                 r2_score(y_test, y_pred)))
#                         a=1
#
# np.save('output/results/r2_snr0.npy', r2)

# column_labels = ['Rate', 'PPC', 'DPC']
# row_labels = ['True rate', 'True PPC', 'True DPC']
# data = r2_mean
=======
>>>>>>> 3f7a39672611c1f6f669a59d753b483185a62327
#
#                     ### BEGINNING OF THIRS LOOP OVER SESSIONS (CROSS-VALIDATION)
#                     # Current cross-validation matrice and response
#                     X_cv = Xz[k_fit_scheme][k_fit_N][k_subject]
#                     y_cv = yz[k_true_scheme][k_population_fraction][k_subpopulation_fraction][k_true_N][k_subject]
#                     for k_session in range(n_sessions):
#                         X_train = np.concatenate(np.asarray(X_cv[:k_session]+Xz[k_session+1:]), axis=0)
#                         y_train = np.concatenate(np.asarray(y_cv[:k_session]+y_cv[k_session+1:]), axis=0)
#                         X_test = X_cv[k_session]
#                         y_test = y_cv[k_session]
#                         # Create linear regression object
#                         regr = linear_model.LinearRegression(fit_intercept=True)
#
#                         # Train the model using the training set
#                         regr.fit(X_train, y_train)
#                         # Make predictions using the testing set
#                         y_pred = regr.predict(X_test)
#                         # mse[block] = mean_squared_error(y_test, y_pred)
#                         # Updates the big tensor
#                         r2[k_fit_scheme, k_fit_N, k_true_N, k_population_fraction, k_subpopulation_fraction, k_subject, k_session] \
#                             = r2_score(y_test, y_pred)
#                         # print('R2 = '+str(r2_score(y_test, y_pred)))
#                         # # The coefficients
#                         # print('True coefficients: \n', true_voxel.weights)
#                         # print('Fitted coefficients: \n', regr.coef_)
#                         with open("output/results/Output.txt", "w") as text_file:
#                             text_file.write('k_fit_scheme={} k_fit_N={} k_true_N={} k_subject={} k_population_fraction={} k_subpopulation_fraction={} k_session={} \nr2={} \n'.format(
#                                 k_fit_scheme, k_fit_N, k_true_N, k_subject, k_population_fraction, k_subpopulation_fraction, k_session,
#                                 r2_score(y_test, y_pred)))
#                         print('Completed : k_fit_scheme={} k_fit_N={} k_true_N={} k_subject={} k_population_fraction={} k_subpopulation_fraction={} k_session={} \nr2={} \n'.format(
#                                 k_fit_scheme, k_fit_N, k_true_N, k_subject, k_population_fraction, k_subpopulation_fraction, k_session,
#                                 r2_score(y_test, y_pred)))
#                         a=1
#
# np.save('output/results/r2_snr0.npy', r2)
#
# # column_labels = ['Rate', 'PPC', 'DPC']
# # row_labels = ['True rate', 'True PPC', 'True DPC']
# # data = r2_mean
# #
# # fig, ax = plt.subplots()
# # heatmap = ax.pcolor(data, cmap=plt.cm.Blues, vmin=0, vmax=1)
# #
# # # put the major ticks at the middle of each cell
# # ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
# # ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
# #
# # # want a more natural, table-like display
# # ax.invert_yaxis()
# # ax.xaxis.tick_top()
# #
# # ax.set_xticklabels(column_labels, minor=False)
# # ax.set_yticklabels(row_labels, minor=False)
# # cbar = fig.colorbar(heatmap, ticks=[0, 1])
# # plt.show()
