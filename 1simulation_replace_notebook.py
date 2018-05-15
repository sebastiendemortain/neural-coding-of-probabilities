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
n_fractions = 1335

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

with open("output/design_matrices/X_20sub_f.txt", "rb") as fp: #X_20sub_f.txt", "rb") as fp:   # Unpickling
    X = pickle.load(fp)

# Whiten the design matrices

# Whitening matrix
white_mat = sio.loadmat('data/simu/whitening_matrix.mat')
W = white_mat['W']
# Complete the in-between session "holes"
W[300:600, 300:600] = W[20:320, 20:320]

whitening_done = False

if not whitening_done:
    # Multiplying the zscored X with the whitening matrix
    for k_scheme, k_fit_N, k_subject, k_session in itertools.product(range(n_schemes), range(n_N), range(n_subjects), range(n_sessions)):
        X_tmp = copy.deepcopy(X[k_scheme][k_fit_N][k_subject][k_session])    # Just to lighten code
        rows_dim, columns_dim = X_tmp.shape
        X_tmp = np.matmul(W[:rows_dim, :rows_dim], X_tmp)
        X[k_scheme][k_fit_N][k_subject][k_session] = copy.deepcopy(X_tmp)

whitening_done = True

# Creation of y from X to save computational resources
# Initialization of the response vectors
y = [[[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fraction in range(n_fractions)]
for k_true_N in range(n_N)] for k_scheme in range(n_schemes)]

# Initialization of the weights
weights = [[[[None for k_subject in range(n_subjects)] for k_fraction in range(n_fractions)] for k_true_N in range(n_N)]
           for k_scheme in range(n_schemes)]


### LOOP OVER THE SCHEME
for k_scheme in range(n_schemes):
    true_scheme = scheme_array[k_scheme]

    # We replace the right value of the "t"'s according to the type of tuning curve

    if true_scheme.find('gaussian') != -1:
        true_t_mu_array = copy.deepcopy(t_mu_gaussian_array)
        true_t_conf_array = copy.deepcopy(t_conf_gaussian_array)
        true_tc_type = 'gaussian'

    elif true_scheme.find('sigmoid') != -1:
        true_t_mu_array = copy.deepcopy(t_mu_sigmoid_array)
        true_t_conf_array = copy.deepcopy(t_conf_sigmoid_array)
        true_tc_type = 'sigmoid'

    # We consider combinations of population fractions for PPC and rate codes
    if true_scheme.find('ppc') != -1 or true_scheme.find('rate') != -1:
        # The number of population fraction tested (related to W)
        population_fraction_array = copy.deepcopy(np.array([[0.5, 0.5], [0.25, 0.75], [0, 1], [0.75, 0.25], [1, 0]]))
    elif true_scheme.find('dpc') != -1:  # DPC case
        population_fraction_array = copy.deepcopy(np.array([[1]]))
    n_population_fractions = len(population_fraction_array)

    ### LOOP OVER N_true
    for k_true_N in range(n_N):
        true_N = N_array[k_true_N]
        # Creation of the true tuning curve objects
        true_t_mu = true_t_mu_array[k_true_N]
        true_t_conf = true_t_conf_array[k_true_N]
        true_tc_mu = tuning_curve(true_tc_type, true_N, true_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
        true_tc_conf = tuning_curve(true_tc_type, true_N, true_t_conf, tc_lower_bound_conf,
                                     tc_upper_bound_conf)

        if true_scheme.find('ppc') != -1:
            true_tc = [true_tc_mu, true_tc_conf]
        elif true_scheme.find('dpc') != -1:
            true_tc = [true_tc_mu]
        elif true_scheme.find('rate') != -1:
            true_tc = []

        ### LOOP OVER THE W's
        # The number of subpopulation fractions acc. to the scheme
        n_subpopulation_fractions = int(n_fractions / n_population_fractions)
        fraction_counter = 0
        for k_subpopulation_fraction in range(n_subpopulation_fractions):
            for k_population_fraction, population_fraction in enumerate(population_fraction_array):
                # The number of populations acc. to the scheme (2 for PPC and rate, 1 for DPC)
                n_population = len(population_fraction)
                if true_scheme.find('ppc') != -1 or true_scheme.find('dpc') != -1:
                    # We consider one sparsity per remainder value of the counter divided by the number
                    # of combinations to be tested
                    subpopulation_sparsity_exp = sparsity_exp_array[fraction_counter % n_sparsity_exp]
                    # Fraction of each neural subpopulation
                    subpopulation_fraction = neural_proba.get_subpopulation_fraction(n_population, true_N,
                                                                                     subpopulation_sparsity_exp)
                else:  # Rate case
                    population_fraction = np.array([1, 1])

                # Generate the data from the voxel
                true_voxel = voxel(true_scheme, population_fraction, subpopulation_fraction, true_tc)
                n_true_features = n_population * true_N
                weights_tmp = copy.deepcopy(np.reshape(true_voxel.weights, (n_true_features,)))

                ### LOOP OVER THE SUBJECTS
                for k_subject in range(n_subjects):
                    # Allocation of the weight tensor
                    weights[k_scheme][k_true_N][fraction_counter][k_subject] \
                        = copy.deepcopy(weights_tmp)

                    ### LOOP OVER THE SESSIONS : simulating the response
                    for k_session in range(n_sessions):
                            # We use X to compute y order to save some computation time
                            # Temporary variables to lighten the reading
                            X_tmp = copy.deepcopy(X[k_scheme][k_true_N][k_subject][k_session])
                            y_tmp = copy.deepcopy(np.matmul(X_tmp, weights_tmp))

                            # Allocation of the tensor
                            y[k_scheme][k_true_N][fraction_counter][k_subject][
                                k_session] = copy.deepcopy(y_tmp)


                fraction_counter += 1

# Noise injection

# Compute the amplitude of the noise
for k_scheme, k_true_N in itertools.product(range(n_schemes), range(n_N)):
    all_y = np.asarray(y[k_scheme][k_true_N]).flatten()    # Concatenation of all y grouped together for SNR computation
    #print(all_y[0])
    noise_sd = np.sqrt(np.var(all_y[0])*(1/snr-1))    # std of the added gaussian noise
    del all_y    # Free memory
    for k_fraction, k_subject, k_session in itertools.product(range(n_fractions), range(n_subjects), range(n_sessions)):
        y_tmp = copy.deepcopy(y[k_scheme][k_true_N][k_fraction][k_subject][k_session])
        y_tmp = y_tmp + np.random.normal(0, noise_sd, len(y_tmp))
        y[k_scheme][k_true_N][k_fraction][k_subject][k_session] = copy.deepcopy(y_tmp)

# Create the filtering design matrices and filters out the response

for k_scheme, k_true_N, k_fraction, k_subject, k_sessions in itertools.product(range(n_schemes), range(n_N), range(n_fractions), range(n_subjects), range(n_sessions)):
    y_tmp = copy.deepcopy(y[k_scheme][k_true_N][k_fraction][k_subject][k_session])
    N = len(y_tmp)    # Resolution of the signal
    K = 11    # Highest order of the filter
    n_grid = np.linspace(0, N-1, N, endpoint=True)    # 1D grid over values
    k_grid = np.linspace(2, K, K-1, endpoint=True)    # 1D grid over orders
    X_filter = np.zeros((N, K-1))
    for kk, k in enumerate(k_grid):
        X_filter[:, kk] = np.sqrt(2/N) * np.cos(np.pi*(2*n_grid+1)/(2*N)*(k-1))
    y_tmp = copy.deepcopy(y_tmp - np.matmul(np.matmul(X_filter, np.transpose(X_filter)), y_tmp))    # Regression
    y[k_scheme][k_true_N][k_fraction][k_subject][k_session] = copy.deepcopy(y_tmp)

# Z-scoring of X and y
# Initialization
Xz = [[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fit_N in range(n_N)]
      for k_scheme in range(n_schemes)]

X_sd_array = [[[None for k_subject in range(n_subjects)] for k_fit_N in range(n_N)]
              for k_scheme in range(n_schemes)]

yz = [[[[[None for k_session in range(n_sessions)] for k_subject in range(n_subjects)] for k_fraction in
        range(n_fractions)]
       for k_true_N in range(n_N)] for k_scheme in range(n_schemes)]

y_sd_array = [[[[None for k_subject in range(n_subjects)] for k_fraction in range(n_fractions)]
               for k_true_N in range(n_N)] for k_scheme in range(n_schemes)]

for k_scheme, k_fit_N, k_subject, k_session in itertools.product(range(n_schemes), range(n_N), range(n_subjects),
                                                                 range(n_sessions)):
    Xz[k_scheme][k_fit_N][k_subject][k_session] = np.zeros_like(X[k_scheme][k_fit_N][k_subject][k_session])

# Manual Z-scoring of regressors inside the session
for k_scheme, k_fit_N, k_subject in itertools.product(range(n_schemes), range(n_N), range(n_subjects)):
    n_fit_features = len(X[k_scheme][k_fit_N][k_subject][0][0])
    X_mean = np.mean(np.concatenate(X[k_scheme][k_fit_N][k_subject], axis=0), axis=0)
    X_sd = np.std(np.concatenate(X[k_scheme][k_fit_N][k_subject], axis=0), axis=0)
    X_sd_array[k_scheme][k_fit_N][k_subject] = copy.deepcopy(X_sd)
    for k_session in range(n_sessions):
        for feature in range(n_fit_features):
            Xz[k_scheme][k_fit_N][k_subject][k_session][:, feature] \
                = (copy.deepcopy(X[k_scheme][k_fit_N][k_subject][k_session][:, feature]) - X_mean[
                feature] * np.ones_like(
                X[k_scheme][k_fit_N][k_subject][k_session][:, feature])) / X_sd[feature]  # Centering + Standardization
    # End of z-scoring

for k_scheme, k_true_N, k_fraction, k_subject in itertools.product(range(n_schemes), range(n_N), range(n_fractions),
                                                                   range(n_subjects)):
    # Z-scoring of y
    y_mean = np.mean(np.concatenate(np.asarray(y[k_scheme][k_true_N][k_fraction][k_subject]),
                                    axis=0), axis=0)
    y_sd = np.std(np.concatenate(np.asarray(y[k_scheme][k_true_N][k_fraction][k_subject]),
                                 axis=0))
    y_sd_array[k_scheme][k_true_N][k_fraction][k_subject] = y_sd

    for k_session in range(n_sessions):
        yz[k_scheme][k_true_N][k_fraction][k_subject][k_session] = \
            (copy.deepcopy(y[k_scheme][k_true_N][k_fraction][k_subject][
                               k_session]) - y_mean) / y_sd  # Centering and standardization

    ### End of z-scoring of y

# Reajusting the weights after zscoring
for k_scheme, k_true_N, k_fraction, k_subject in itertools.product(range(n_schemes), range(n_N), range(n_fractions),
                                                                   range(n_subjects)):
    for feature in range(weights[k_scheme][k_true_N][k_fraction][k_subject].shape[0]):
        weights[k_scheme][k_true_N][k_fraction][k_subject][feature] = \
        weights[k_scheme][k_true_N][k_fraction][k_subject][feature] * X_sd_array[k_scheme][k_true_N][k_subject][
            feature] / y_sd_array[k_scheme][k_true_N][k_fraction][k_subject]

# The loops
# The quantity to be computed during the cross validation
r2_test = np.zeros((n_schemes, n_N, n_N, n_fractions, n_subjects, n_sessions))
r2_train = np.zeros((n_schemes, n_N, n_N, n_fractions, n_subjects, n_sessions))
rho_test = np.zeros((n_schemes, n_N, n_N, n_fractions, n_subjects, n_sessions))
rho_train = np.zeros((n_schemes, n_N, n_N, n_fractions, n_subjects, n_sessions))

### BEGINNING OF LOOPS OVER HYPERPARAMETERS
for k_scheme, k_fit_N, k_true_N, k_fraction, k_subject in itertools.product(range(n_schemes),
                                                                            range(n_N), range(n_N), range(n_fractions), range(n_subjects)):
    # Current cross-validation matrix and response
    X_cv = copy.deepcopy(Xz[k_scheme][k_fit_N][k_subject])
    y_cv = copy.deepcopy(yz[k_scheme][k_true_N][k_fraction][k_subject])
    # LOOP OVER SESSIONS (CV)
    for k_session in range(n_sessions):
        X_train = copy.deepcopy(np.concatenate(X_cv[:k_session]+X_cv[k_session+1:], axis=0))
        y_train = copy.deepcopy(np.concatenate(y_cv[:k_session]+y_cv[k_session+1:], axis=0))
        X_test = copy.deepcopy(X_cv[k_session])
        y_test = copy.deepcopy(y_cv[k_session])

        # Train the model using the training set
        regr.fit(X_train, y_train)
        # Make predictions using the testing set
        y_pred = regr.predict(X_test)
        # Updates the big tensor
        y_hat_train = regr.predict(X_train)
        r2_train[k_scheme, k_fit_N, k_true_N, k_fraction, k_subject, k_session] \
            = r2_score(y_train, y_hat_train)
        r2_test[k_scheme, k_fit_N, k_true_N, k_fraction, k_subject, k_session] \
            = r2_score(y_test, y_pred)
        rho_train[k_scheme, k_fit_N, k_true_N, k_fraction, k_subject, k_session] \
            = pearsonr(y_train, y_hat_train)[0]
        rho_test[k_scheme, k_fit_N, k_true_N, k_fraction, k_subject, k_session] \
            = pearsonr(y_pred, y_test)[0]
        # with open("output/results/Output.txt", "w") as text_file:
        #     text_file.write('k_fit_scheme={} k_fit_N={} k_true_N={} k_subject={} k_population_fraction={} k_subpopulation_fraction={} k_session={} \nr2={} \n'.format(
        #         k_fit_scheme, k_fit_N, k_true_N, k_subject, k_population_fraction, k_subpopulation_fraction, k_session,
        #         r2_score(y_test, y_pred)))
        # print('Completed : k_fit_scheme={} k_fit_N={} k_true_N={} k_subject={} k_population_fraction={} k_subpopulation_fraction={} k_session={} \nr2={} \nr2_train={}'.format(
        #         k_fit_scheme, k_fit_N, k_true_N, k_subject, k_population_fraction, k_subpopulation_fraction, k_session,
        #         r2_score(y_test, y_pred), r2_train))

# Histogram of r2_train for each fit_N and true_N

r2_train_summary = np.zeros((n_schemes, n_N, n_N, n_fractions*n_subjects*n_sessions))
r2_test_summary = np.zeros((n_schemes, n_N, n_N, n_fractions*n_subjects*n_sessions))
rho_train_summary = np.zeros((n_schemes, n_N, n_N, n_fractions*n_subjects*n_sessions))
rho_test_summary = np.zeros((n_schemes, n_N, n_N, n_fractions*n_subjects*n_sessions))

np.save('output/results/r2_test_snr'+str(snr)+'.npy', r2_test)
np.save('output/results/r2_train_snr'+str(snr)+'.npy', r2_train)
np.save('output/results/rho_test_snr'+str(snr)+'.npy', rho_test)
np.save('output/results/rho_train_snr'+str(snr)+'.npy', rho_train)
