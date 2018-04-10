"""
Example of hemodynamic reponse functions.
=========================================
We consider the hrf model in SPM together with the hrf shape proposed by
G.Glover, as well as their time and dispersion derivatives.

Requires matplotlib

Author : Bertrand Thirion: 2009-2015
"""

import numpy as np
import matplotlib.pyplot as plt
from nistats import hemodynamic_models


#########################################################################
# A first step: looking at our data
# ----------------------------------
#
# Let's quickly plot this file:
dt = 0.01    # Temporal step
initial_time = 0
final_time = 30
frame_times = np.arange(initial_time, final_time, dt)

# Experimental stimuli1
n_events = 20
onset = [None]*n_events
amplitude = [None]*n_events
duration = [None]*n_events

# One event every second
onset = np.linspace(1, n_events, n_events)
amplitude = np.ones(n_events)
duration = np.ones(n_events)*10

stim = np.zeros_like(frame_times)
for k in range(n_events):
    stim[(frame_times > onset[k]) * (frame_times <= onset[k] + duration[k])] = amplitude[k]
exp_condition = np.array((onset, duration, amplitude)).reshape(3, n_events)
hrf_models = [None, 'spm', 'glover + derivative']

#########################################################################
# sample the hrf
fig = plt.figure(figsize=(9, 4))
for i, hrf_model in enumerate(hrf_models):
    signal, name = hemodynamic_models.compute_regressor(
        exp_condition, hrf_model, frame_times, con_id='main',
        oversampling=16, fir_delays=np.array([1., 2.]))

    plt.subplot(1, 3, i + 1)
    plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
    for j in range(signal.shape[1]):
        plt.plot(frame_times, signal.T[j], label=name[j])
    plt.xlabel('time (s)')
    plt.legend(loc=1)
    plt.title(hrf_model)

plt.subplots_adjust(bottom=.12)
plt.show()
