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

# Experimental design information
eps = 1e-5  # For floating points issues

between_stimuli_duration = 1.3
initial_time = between_stimuli_duration + eps
final_time = between_stimuli_duration * (n_stimuli + 1) + eps
stimulus_onsets = np.linspace(initial_time, final_time, n_stimuli)
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

### LOOP OVER THE SCHEME
for k_fit_scheme in range(n_schemes):

k_fit_scheme=0
k_true_scheme=k_fit_scheme

# Current schemes
fit_scheme = scheme_array[k_fit_scheme]
true_scheme = scheme_array[k_true_scheme]

# We replace the right value of the "t"'s according to the type of tuning curve
if fit_scheme.find('gaussian') != -1:
    fit_t_mu_array = t_mu_gaussian_array
    fit_t_sigma_array = t_sigma_gaussian_array
    fit_tc_type = 'gaussian'

elif fit_scheme.find('sigmoid') != -1:
    fit_t_mu_array = t_mu_sigmoid_array
    fit_t_sigma_array = t_sigma_sigmoid_array
    fit_tc_type = 'sigmoid'

if true_scheme.find('gaussian') != -1:
    true_t_mu_array = t_mu_gaussian_array
    true_t_sigma_array = t_sigma_gaussian_array
    true_tc_type = 'gaussian'

elif true_scheme.find('sigmoid') != -1:
    true_t_mu_array = t_mu_sigmoid_array
    true_t_sigma_array = t_sigma_sigmoid_array
    true_tc_type = 'sigmoid'

### LOOP OVER THE N's

k_fit_N=0
k_true_N=0

# Current N
true_N = N_array[k_true_N]
fit_N = N_array[k_fit_N]

# Current t
true_t_mu = true_t_mu_array[k_true_N]
fit_t_mu = true_t_mu_array[k_fit_N]
true_t_sigma = true_t_sigma_array[k_true_N]
fit_t_sigma = true_t_sigma_array[k_fit_N]

# Creation of the tuning curve objects
true_tc_mu = tuning_curve(true_tc_type, true_N, true_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
true_tc_sigma = tuning_curve(true_tc_type, true_N, true_t_sigma, tc_lower_bound_sigma, tc_upper_bound_sigma)

if true_scheme.find('ppc') != -1:
    true_tc = [true_tc_mu, true_tc_sigma]
elif true_scheme.find('dpc') != -1:
    true_tc = [true_tc_mu]
elif true_scheme.find('rate') != -1:
    true_tc = []

### LOOP OVER THE W's

k_population_fraction = 0
k_subpopulation_fraction = 0

# Current weights
# Fraction of each neural population
population_fraction = neural_proba.get_population_fraction(true_scheme)
n_population = len(population_fraction)

# Fraction of each neural subpopulation
# No rate coding here
subpopulation_fraction = neural_proba.get_subpopulation_fraction(scheme_array, n_population, true_N)

# Generate the data from the voxel
true_voxel = voxel(true_scheme, population_fraction, subpopulation_fraction, [true_tc_mu, true_tc_sigma])


### LOOP OVER THE SUBJECTS
k_subject = 0

# Initialization of the design matrix and response
X = [None for k in range(n_sessions)]
y = [None for k in range(n_sessions)]

### FIRST LOOP OVER THE SESSIONS : simulating the data and regressors for this subject
k_session = 0

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

X[k_session] = simu_fmri.get_regressor(exp, fit_scheme, true_tc)
n_features = len(X[0][0, :])

if fit_scheme==true_scheme:    # In order to save some computation time
    weights = np.reshape(true_voxel.weights, (n_features,))
    y[k_session] = np.dot(X[k_session], weights)
else:    # We need to compute both the regressors and the response separately
    true_activity = true_voxel.generate_activity(simulated_distrib)
    signal, scan_signal, name, stim = simu_fmri.get_bold_signal(exp, true_activity, hrf_model, 1)
    y[k_session] = scan_signal

# Noise injection
y[k_session] = y[k_session] + noise_coeff*np.random.normal(0, 1, len(y[k_session]))

### END OF FIRST LOOP OVER SESSIONS

# Manual z-scoring
X_mean = np.mean(np.concatenate(np.asarray(X), axis=0), axis=0)
X_sd = np.std(np.concatenate(np.asarray(X), axis=0), axis=0)

y_mean = np.mean(np.concatenate(np.asarray(y), axis=0))
y_sd = np.std(np.concatenate(np.asarra(y), axis=0))

### BEGINNING OF SECOND LOOP OVER SESSIONS (Z-SCORING)

for k_session in range(n_sessions):
    y[k_session] = y[k_session] - y_mean    # Centering
    y[k_session] = y[k_session]/y_sd    # Standardization
    for feature in range(n_features):
        X[k_session][:, feature] = X[k_session][:, feature]-X_mean[feature]*np.ones_like(X[k_session][:, feature])    # Centering
        X[k_session][:, feature] = X[k_session][:, feature]/X_sd[feature]     # Standardization

### END OF SECOND LOOP OVER SESSIONS


### BEGINNING OF THIRS LOOP OVER SESSIONS (CROSS-VALIDATION)

for k_session in range(n_sessions):
    X_train = np.concatenate(np.asarray(X[:k_session]+X[k_session+1:]), axis=0)
    y_train = np.concatenate(np.asarray(y[:k_session]+y[k_session+1:]), axis=0)
    X_test = X[k_session]
    y_test = y[k_session]
    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=True)

    # Train the model using the training set
    regr.fit(X_train, y_train)
    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    # mse[block] = mean_squared_error(y_test, y_pred)
    # Updates the big tensor
    r2[k_fit_scheme, k_fit_N, k_true_N, k_population_fraction, k_subpopulation_fraction, k_subject, k_session] \
        = r2_score(y_test, y_pred)
    # print('R2 = '+str(r2_score(y_test, y_pred)))
    # # The coefficients
    # print('True coefficients: \n', true_voxel.weights)
    # print('Fitted coefficients: \n', regr.coef_)
    with open("output/results/Output.txt", "w") as text_file:
        text_file.write('k_fit_scheme={} k_fit_N={} k_true_N={}, k_population_fraction={} k_subpopulation_fraction={} k_subject={} k_session={} \nr2={} \n'.format(
            k_fit_scheme, k_fit_N, k_true_N, k_population_fraction, k_subpopulation_fraction, k_subject, k_session,
            r2_score(y_test, y_pred)))


np.save('output/results/r2_snr0.npy', r2)

# column_labels = ['Rate', 'PPC', 'DPC']
# row_labels = ['True rate', 'True PPC', 'True DPC']
# data = r2_mean
#
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(data, cmap=plt.cm.Blues, vmin=0, vmax=1)
#
# # put the major ticks at the middle of each cell
# ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
# ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
#
# # want a more natural, table-like display
# ax.invert_yaxis()
# ax.xaxis.tick_top()
#
# ax.set_xticklabels(column_labels, minor=False)
# ax.set_yticklabels(row_labels, minor=False)
# cbar = fig.colorbar(heatmap, ticks=[0, 1])
# plt.show()