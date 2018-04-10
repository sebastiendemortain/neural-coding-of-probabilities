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

# INPUTS
true_coding_scheme_array = ['rate', 'ppc', 'dpc']
fit_coding_scheme_array = ['rate', 'ppc', 'dpc']
r2_mean = np.zeros((3, 3))
noise_coeff = 0
aqw
for k_true, true_coding_scheme in enumerate(true_coding_scheme_array):
    for k_fit, fit_coding_scheme in enumerate(fit_coding_scheme_array):
        # # Properties of the voxel to be simulated
        # true_coding_scheme = 'ppc'
        # true_population_fraction = [1]  # one for the mean, one for the std
        fmri_gain = 1    # Amplification of the signal

        # # Fitted model
        # fit_coding_scheme = 'rate'

        X = np.load('X_{}.npy'.format(fit_coding_scheme))
        y = np.load('y_{}.npy'.format(true_coding_scheme))
        true_weights = np.load('true_weights_{}.npy'.format(true_coding_scheme))

        ############################################################
        # Define the seed to reproduce results from random processes
        rand.seed(2);
        n_stimuli = 380
        n_blocks = 4
        n_train = 3
        n_test = 1
        n_features = X[0].shape[1]

        # Noise injection
        for block in range(n_blocks):
            y[block] = y[block] + noise_coeff*np.random.normal(0, 1, len(y[block]))

        # Manual z-scoring
        X_mean = np.mean(np.concatenate(X, axis=0), axis=0)
        X_sd = np.std(np.concatenate(X, axis=0), axis=0)

        y_mean = np.mean(np.concatenate(y, axis=0))
        y_sd = np.std(np.concatenate(y, axis=0))

        for block in range(n_blocks):
            y[block] = y[block] - y_mean    # Centering
            y[block] = y[block]/y_sd    # Standardization
            for feature in range(n_features):
                X[block][:, feature] = X[block][:, feature]-X_mean[feature]*np.ones_like(X[block][:, feature])    # Centering
                X[block][:, feature] = X[block][:, feature]/X_sd[feature]     # Standardization

        mse = np.zeros(n_blocks)
        r2 = np.zeros(n_blocks)
        # Create the folds from the data
        for block in range(n_blocks):
            mask = [True for k in range(4)]
            mask[block] = False
            X_train = np.concatenate(X[mask], axis=0)
            y_train = np.concatenate(y[mask], axis=0)
            X_test = X[block]
            y_test = y[block]
            # Create linear regression object
            regr = linear_model.LinearRegression(fit_intercept=True)

            # Train the model using the training set
            regr.fit(X_train, y_train)
            # Make predictions using the testing set
            y_pred = regr.predict(X_test)
            mse[block] = mean_squared_error(y_test, y_pred)
            r2[block] = r2_score(y_test, y_pred)

        # The coefficients
        # print('True coefficients: \n', true_weights)
        # print('Fitted coefficients: \n', regr.coef_)
        # The mean squared error
        # print("Mean squared error: ", mse)
        # # Explained variance score: 1 is perfect prediction
        # print('Variance score:', r2)

        r2_mean[k_true, k_fit] = np.mean(r2[block])

print(r2_mean)

# fig, ax = plt.subplots()
plt.imshow(r2_mean);
# # Turn off tick labels
# ax.set_yticklabels([])
# ax.set_xticklabels([])

plt.colorbar()
plt.show()
