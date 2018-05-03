import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
import numpy as np
#import decimal
# import matplotlib
# matplotlib.use('Agg')    # To avoid bugs
import matplotlib.pyplot as plt
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

r2 = np.load('output/results/r2_test_snr0.4.npy')

scheme_array = ['gaussian_ppc', 'sigmoid_ppc', 'gaussian_dpc', 'sigmoid_dpc']
N_array = np.array([6, 8, 10, 14, 16, 20])


n_N = len(N_array)
n_fractions = 20
n_subjects = 20
n_sessions = 4
n_schemes = len(scheme_array)

scheme = 'sigmoid_ppc'

# r2_1 = r2[:, :, :, 0, :, :]
# r2_1 = np.median(r2_1, axis=(3, 4))
for k_fit_scheme, k_fit_N, k_true_N, k_fraction, k_subject, k_session in itertools.product(range(n_schemes), range(n_N), range(n_N), range(n_fractions), range(n_subjects), range(n_sessions)):
    if r2[k_fit_scheme][k_fit_N][k_true_N][k_fraction][k_subject][k_session] < 0:
        r2[k_fit_scheme][k_fit_N][k_true_N][k_fraction][k_subject][k_session] = 0

r2_1 = r2.mean(axis=(3, 4, 5))
#r2[k_fit_scheme, k_fit_N, k_true_N, fraction_counter, k_subject, k_session]
column_labels = copy.deepcopy(N_array)
row_labels = copy.deepcopy(N_array)
# Find the right index related to the desired scheme
k_scheme = 0
for i, scheme_tmp in enumerate(scheme_array):
    if scheme_tmp.find(scheme)!=-1:
        k_scheme = i

data = r2_1[k_scheme]    # First simulation

fig, ax = plt.subplots()
fontsize = 15
heatmap = ax.pcolor(data, cmap=plt.cm.Blues, vmin=0, vmax=0.3)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(column_labels, minor=False, fontsize=fontsize)
ax.set_yticklabels(row_labels, minor=False, fontsize=fontsize)
ax.set_ylabel('N_fit', fontsize=fontsize)
ax.set_xlabel('N_true', fontsize=fontsize)
plt.title('Results of simulation 1 for '+scheme, y=1.08, fontsize=20)
cbar = fig.colorbar(heatmap, ticks=[0, 1])
plt.show()

print(data[1, 4])
a=1