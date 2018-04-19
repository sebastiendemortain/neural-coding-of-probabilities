import random as rand
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from nistats import hemodynamic_models

import utils
from scipy import io as sio


'''Related to the distribution'''


def import_distrib_param(n_subjects, n_sessions, n_stimuli, distrib_type):

    # Initialization of the outputs
    p1g2_dist_array = [[None for j in range(n_sessions)] for i in range(n_subjects)]
    p1g2_mu_array = [[None for j in range(n_sessions)] for i in range(n_subjects)]
    p1g2_sd_array = [[None for j in range(n_sessions)] for i in range(n_subjects)]

    data_mat = sio.loadmat('data/simu/ideal_observer_{}subjects_{}sessions_{}stimuli_HMM.mat'.format(n_subjects, n_sessions, n_stimuli, distrib_type), struct_as_record=False)
    out = data_mat['out_io']

    for subject in range(n_subjects):
        for session in range(n_sessions):
                out_tmp = out[subject][session]
                p1g2_dist_array[subject][session] = out_tmp[0, 0].p1g2_dist
                p1g2_mu_array[subject][session] = out_tmp[0, 0].p1g2_mean
                p1g2_sd_array[subject][session] = out_tmp[0, 0].p1g2_sd

    return [p1g2_dist_array, p1g2_mu_array, p1g2_sd_array]


class distrib:

    '''This class specifies attributes of a specific distribution'''

    def __init__(self, mu, sigma, dist = []):
        self.mean = mu    # Mean of the distribution
        self.sd = sigma    # Standard deviation of the distribution
        self.dist = dist    # Distribution itself (used sometimes for DPC)
        self.a = ((1-self.mean)/self.sd**2-1/self.mean)*self.mean**2    # First parameter of to build the beta distribution
        self.b = ((1-self.mean)/self.sd**2-1/self.mean)*self.mean**2 * (1/self.mean-1)    # Second parameter to build the beta distribution

    # Equivalent beta distribution
    def beta(self, x):
        return stats.beta.pdf(x, self.a, self.b)


class tuning_curve:
    '''This class defines the tuning curve object
    '''

    # Initialization of the tuning curve attributes
    def __init__(self, tc_type, N, t, lower_bound, upper_bound):
        self.tc_type = tc_type
        self.N = N
        self.t = t
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    #Returns the value in x of tuning curve of index i
    def f(self, x, i):
        if (self.tc_type=='gaussian'):
            # Spacing between each tuning curve
            delta_mu = (self.upper_bound-self.lower_bound)/(self.N-1)
            # Mean of the tuning curve
            mean = self.lower_bound+i*delta_mu
            # Variance of the tuning curve
            sigma2_f = self.t**2
            tc_value = np.exp(-0.5*(x-mean)**2/sigma2_f)
            return tc_value
        elif (self.tc_type=='sigmoid'):
            # Spacing between each tuning curve
            n = self.N/2    # Number of tuning curve of the same monotony
            delta_mu = (self.upper_bound-self.lower_bound)/(n-1)
            # Mean of the tuning curve
            if i<n:    # Increasing sigmoids
                mu = self.lower_bound+i*delta_mu
                tc_value = 1/(1+np.exp(-(x-mu)/self.t))
            else: # Decreasing sigmoids
                mu = self.lower_bound + (i-n) * delta_mu
                tc_value = 1 / (1 + np.exp((x - mu) / self.t))
            return tc_value

        return


    def compute_projection(self, distrib_array, i, use_high_integration_resolution):
        '''Returns the vector/matrix of projection onto the i-th tuning curve of the 2D-grid/the sequence of
        distributions.'''
        # Finds out whether we use 2D grid or just a sequence of distributions
        n_dims = utils.get_dimension_list(distrib_array)
        if n_dims == 2:
            n_mu = len(distrib_array)
            n_sigma = len(distrib_array[0])
            proj = np.zeros([n_mu, n_sigma])  # Initialization
        elif n_dims == 1:
            n_stimuli = len(distrib_array)
            proj = np.zeros(n_stimuli)  # Initialization

        # Decide the numerical way the integral are computed
        if use_high_integration_resolution:
            res = 1e5  # Number of points used for the numerical integral
        else:
            res = 51  # We take the distribution from the Matlab file

        eps = 0  # Reduction of the interval for dealing with boundary issues of the beta function
        # x-axis for integration
        x = np.linspace(self.lower_bound + eps, self.upper_bound - eps, res,
                        endpoint=True)
        x = np.delete(x, -1)  # res-1 points shall be considered for the numerical integration
        # Integral step
        delta_x = (self.upper_bound - self.lower_bound - 2 * eps) / (res - 1)

        # Value of the distribution on tuning curve i
        f = self.f(x, i)  # tuning curve i's values along the x-axis

        # Double list containing the array of the beta function at the desired resolution
        if n_dims == 2:
            beta = [[None for j in range(n_sigma)] for i in range(n_mu)]
            # sum_err = 0
            # err_array = np.array([0])
            for k_mu in range(n_mu):
                for k_sigma in range(n_sigma):
                    # Projection of the distribution on tuning curve i
                    distrib = distrib_array[k_mu][k_sigma]
                    if use_high_integration_resolution:
                        beta[k_mu][k_sigma] = distrib.beta(x)
                    else:
                        beta[k_mu][k_sigma] = distrib.beta(x)
                        # beta[k_mu][k_sigma] = distrib.dist

                    proj[k_mu, k_sigma] = np.dot(beta[k_mu][k_sigma], f) * delta_x
                    # proj_num = integrate.quad(lambda y: distrib.beta(y)*self.tuning_curve[0].f(y, i), self.tuning_curve[0].lower_bound+eps, self.tuning_curve[0].upper_bound-eps)[0]
                    # err = np.abs(proj[k_mu, k_sigma]-proj_num)#/np.sqrt(np.dot(proj[k_mu, k_sigma],proj[k_mu, k_sigma]))
                    # sum_err += err
                    # err_array = np.append(err_array, err)
                    # if np.abs(proj[k_mu, k_sigma]-proj_num) > 0.01:
                    #     print('ISSUE WITH THE NUMERICAL INTEGRATION. See the dpc case in voxel.activity')
                    #print(err)


                    # # Plots the difference between theoretical and sampled distributions
                    # fig = plt.figure()
                    # y1 = beta[k_mu][k_sigma]
                    # x_ = np.linspace(0,1, 10000, endpoint=True)
                    # y2 = distrib.beta(x_)
                    # ax = plt.subplot(111)
                    # ax.bar(x - 0.005, y1, width=0.005, color='b', align='center')
                    # ax.bar(x, y2, width=0.005, color='r', align='center')
                    # plt.show()
                    #
                    # a = 1

        # Single list containing the array of the beta function at the desired resolution
        elif n_dims == 1:
            beta = [None for k in range(n_stimuli)]

            # sum_err = 0
            # Projection of the distribution on tuning curve i
            f = self.f(x, i)  # tuning curve i's values along the x-axis

            for k in range(n_stimuli):
                distrib = distrib_array[k]
                if use_high_integration_resolution:
                    beta[k] = distrib.beta(x)
                else:
                    beta[k] = distrib.dist

                proj[k] = np.dot(beta[k], f) * delta_x
                # proj_num = integrate.quad(lambda y: distrib.beta(y)*self.tuning_curve[0].f(y, i), self.tuning_curve[0].lower_bound+eps, self.tuning_curve[0].upper_bound-eps)[0]
                # sum_err += np.abs(proj-proj_num)/np.sqrt(np.dot(proj,proj))
                # if np.abs(proj-proj_num) > 0.01:
                #     print('ISSUE WITH THE NUMERICAL INTEGRATION. See the dpc case in voxel.activity')
                # print(sum_err)

                # proj_num = integrate.quad(lambda y: distrib.beta(y)*self.tuning_curve[0].f(y, i), self.tuning_curve[0].lower_bound+eps, self.tuning_curve[0].upper_bound-eps)[0]
                # err = np.abs(proj-proj_num)/np.sqrt(np.dot(proj,proj))
                # sum_err += err

                # # Plots the difference between theoretical and sampled distributions
                # fig = plt.figure()
                # y1 = beta[k]
                # y2=distrib.beta(x)
                # ax = plt.subplot(111)
                # ax.bar(x - 0.005, y1, width=0.005, color='b', align='center')
                # ax.bar(x, y2, width=0.005, color='r', align='center')
                # plt.show()
                #
                # a=1
        return proj

def get_population_fraction(scheme, population_sparsity_exp = 1):
    if (scheme.find('ppc')!=-1 or scheme.find('rate')!=-1):
        n_population = 2  # One population for the mean, one for sigma
        population_fraction = np.zeros(n_population)
        population_fraction[0] = rand.uniform(0, 1) ** population_sparsity_exp
        population_fraction[1] = 1 - population_fraction[0]
    elif scheme.find('dpc')!=-1:  # DPC case
        population_fraction = np.array([1])  # DPC contains only one population
    else:
        return
    return population_fraction

def get_subpopulation_fraction(n_population, n_subpopulation, subpopulation_sparsity_exp = 1):
    # Fraction of each neural population
    subpopulation_fraction = np.zeros([n_population, n_subpopulation])
    n_neuron = np.zeros([n_population, n_subpopulation])  # Number of neuron per subpopulation
    n_total_neuron = np.zeros(n_population)
    for pop in range(n_population):
        for subpop in range(n_subpopulation):
            # Number of neuron per subpopulation controlled by a sparsity exponent
            n_neuron[pop, subpop] = (rand.uniform(0,1)) ** subpopulation_sparsity_exp
        n_total_neuron[pop] = np.sum(n_neuron[pop, :])
    for pop in range(n_population):
        for subpop in range(n_subpopulation):
            # Fraction of neuron for each population. It sums up to 1 over each subpopulation
            subpopulation_fraction[pop, subpop] = n_neuron[pop, subpop] / n_total_neuron[pop]

    return subpopulation_fraction

class voxel:

    '''
    This class defines the properties of the voxel we will generate data from.
    '''

    # Gain of the signal due to neural activity fluctuation
    neural_gain = 1

    # Initialization of the attributes of the voxel object
    def __init__(self, coding_scheme, population_fraction, subpopulation_fraction, tc = []):
        self.coding_scheme = coding_scheme   # the type of coding scheme
        self.population_fraction = population_fraction
        self.subpopulation_fraction = subpopulation_fraction
        self.n_population = len(self.population_fraction)    # number of independent neural populations
        self.n_subpopulation = len(self.subpopulation_fraction[0])
        self.tuning_curve = tc
        self.weights = np.zeros(
            (self.n_population, self.n_subpopulation))  # the weights in the activity linear combination
        for pop in range(self.n_population):
            for subpop in range(self.n_subpopulation):
                self.weights[pop, subpop] = self.neural_gain\
                                            *self.population_fraction[pop]*self.subpopulation_fraction[pop, subpop]

    # Neural activity given one distribution
    def generate_activity(self, distrib_array, mu_sd=np.nan, sigma_sd=np.nan,
                          use_high_integration_resolution=False):
        # 2 if 2D-grid for plotting the signal, 1 if one single continuous experiment
        n_dims = utils.get_dimension_list(distrib_array)
        if n_dims == 2:
            n_mu = len(distrib_array)
            n_sigma = len(distrib_array[0])
            activity = np.zeros([n_mu, n_sigma])  # Initialization
        elif n_dims == 1:
            n_stimuli = len(distrib_array)
            activity = np.zeros(n_stimuli)  # Initialization

        if self.coding_scheme.find('rate') != -1:    # Rate coding case
            if self.n_population == 2:    # If the mean and the variance are encoded by two independent populations
                # Activity due to mean and uncertainty encoding (linear combination). Only one neuron per subpopulation
                '''
                We multiply the contribution of the uncertainty through a scaling factor (ratio of the std) in order 
                to have same neural impact between mean and uncertainty
                '''
                # Case of double array (when 2D-grid of distributions)
                if n_dims == 2:
                    for k_mu in range(n_mu):
                        for k_sigma in range(n_sigma):
                            distrib = distrib_array[k_mu][k_sigma]
                            mu = distrib.mean  # Mean of the distribution
                            sigma = distrib.sd  # Standard deviation of the distribution
                            activity[k_mu, k_sigma] = self.population_fraction[0]*self.subpopulation_fraction[0, 0]\
                                                        *mu + (mu_sd/sigma_sd)*self.population_fraction[1]\
                                                                  *self.subpopulation_fraction[1, 0]*sigma
                # Case of single array (when 1D-grid of distributions)
                elif n_dims == 1:
                    for k in range(n_stimuli):
                        distrib = distrib_array[k]
                        mu = distrib.mean  # Mean of the distribution
                        sigma = distrib.sd  # Standard deviation of the distribution
                        activity[k] = self.population_fraction[0] * self.subpopulation_fraction[0, 0] \
                                                    * mu + (mu_sd / sigma_sd) * self.population_fraction[1] \
                                                               * self.subpopulation_fraction[1, 0] * sigma
                # Correction of the rate weights related to sigma
                self.weights[1, :] = self.weights[1, :] * (mu_sd / sigma_sd)

        elif self.coding_scheme.find('ppc') != -1:    # Probabilistic population coding case

            if n_dims == 2:
                # Defines the 2D-grid of means and sigma
                x_mu = np.zeros(n_mu)
                x_sigma = np.zeros(n_sigma)
                for k_mu in range(n_mu):
                    distrib = distrib_array[k_mu][0]
                    x_mu[k_mu] = distrib.mean
                for k_sigma in range(n_sigma):
                    distrib = distrib_array[0][k_sigma]
                    x_sigma[k_sigma] = distrib.sd

                for i in range(self.tuning_curve[0].N):
                    f_mu = self.tuning_curve[0].f(x_mu, i)
                    f_sigma = self.tuning_curve[1].f(x_sigma, i)
                    for k_mu in range(n_mu):
                        for k_sigma in range(n_sigma):
                            activity[k_mu, k_sigma] += self.weights[0, i]*f_mu[k_mu]\
                                                         + self.weights[1, i]*f_sigma[k_sigma]

            elif n_dims == 1:
                # Defines the 1D-grid of (mean, sigma)
                x_mu = np.zeros(n_stimuli)
                x_sigma = np.zeros(n_stimuli)
                for k in range(n_stimuli):
                    distrib = distrib_array[k]
                    x_mu[k] = distrib.mean
                    x_sigma[k] = distrib.sd

                for i in range(self.tuning_curve[0].N):
                    frac_mu = self.subpopulation_fraction[0, i]
                    frac_sigma = self.subpopulation_fraction[1, i]
                    f_mu = self.tuning_curve[0].f(x_mu, i)
                    f_sigma = self.tuning_curve[1].f(x_sigma, i)

                    for k in range(n_stimuli):
                        activity[k] += self.weights[0, i]*f_mu[k]\
                                                         + self.weights[1, i]*f_sigma[k]
        elif self.coding_scheme.find('dpc') != -1:    # Distributional population coding case

            for i in range(self.tuning_curve[0].N):
                proj = self.tuning_curve[0].compute_projection(distrib_array, i, use_high_integration_resolution)
                activity += self.weights[0, i] * proj

        # Multiplication by the gain of the signal
        activity = self.neural_gain * activity
        return activity

class experiment:

    def __init__(self, initial_time, final_time, n_sessions, stimulus_onsets, stimulus_durations, distributions):
        self.initial_time = initial_time
        self.final_time = final_time
        self.n_sessions = n_sessions
        self.n_stimuli = len(stimulus_onsets)
        self.stimulus_onsets = stimulus_onsets
        self.stimulus_durations = stimulus_durations
        self.distributions = distributions    # Encoded distributions


class fmri:

    ''' This class defines the fMRI modalities '''

    def __init__(self, initial_frame_time, final_frame_time, dt, scan_times):
        self.initial_frame_time = initial_frame_time
        self.final_frame_time = final_frame_time
        self.dt = dt    # Temporal resolution
        self.frame_times = np.arange(self.initial_frame_time, self.final_frame_time, self.dt)    # for now
        self.scan_times = scan_times
        self.n_scans = len(scan_times)

    def get_bold_signal(self, exp, amplitudes, hrf_model, fmri_gain = 1):
        '''To get the response vector'''
        # To get the stimuli signal within the frame_times framework
        stim = np.zeros_like(self.frame_times)  # Contains amplitude of the stimulus in the frame_times space
        for k in range(exp.n_stimuli):
            stim[(self.frame_times > exp.stimulus_onsets[k])
                 * (self.frame_times <= exp.stimulus_onsets[k] + exp.stimulus_durations[k])] \
                = amplitudes[k]

        # Build experimental condition vector
        exp_condition = np.array((exp.stimulus_onsets, exp.stimulus_durations, amplitudes[:exp.n_stimuli]))\
            .reshape(3, exp.n_stimuli)

        signal, name = hemodynamic_models.compute_regressor(
                exp_condition, hrf_model, self.frame_times, con_id='main',
                oversampling=16)

        # Amplify the signal
        signal = fmri_gain * signal

        # Take the signal only at the scan values
        scan_signal = np.zeros_like(self.scan_times)
        for k, scan_time in enumerate(self.scan_times):
            idx = utils.find_nearest(self.frame_times, scan_time)[0]
            scan_signal[k] = signal[idx]

        return signal, scan_signal, name, stim

    def get_regressor(self, exp, coding_scheme, tc=[], reg_fmri_gain=1, use_high_integration_resolution=False):
        hrf_model = 'spm'    # Simple regressor computation

        if coding_scheme.find('rate')!=-1:
            # Get the features before convolution
            mu = np.zeros(exp.n_stimuli)
            sigma = np.zeros(exp.n_stimuli)
            for k in range(exp.n_stimuli):
                mu[k] = exp.distributions[k].mean
                sigma[k] = exp.distributions[k].sd

            q_signal, q_scan_signal, name, stim = self.get_bold_signal(exp, mu, hrf_model)
            sigma_signal, sigma_scan_signal, name, stim = self.get_bold_signal(exp, sigma, hrf_model)

            X = np.array([q_scan_signal, sigma_scan_signal])
            X = np.transpose(X)

        elif coding_scheme.find('ppc')!=-1:
            mu = np.zeros(exp.n_stimuli)
            sigma = np.zeros(exp.n_stimuli)
            for k in range(exp.n_stimuli):
                mu[k] = exp.distributions[k].mean
                sigma[k] = exp.distributions[k].sd
            tc_mu = tc[0]
            tc_sigma = tc[1]
            # Initialization of the design matrix
            X = np.zeros((len(self.scan_times), tc_mu.N+tc_sigma.N))
            q_scan_signal = np.zeros((len(self.scan_times), tc_mu.N))
            sigma_scan_signal = np.zeros((len(self.scan_times), tc_sigma.N))

            for i in range(tc_mu.N):
                q_signal_tmp, q_scan_signal_tmp, name, stim = \
                    self.get_bold_signal(exp, tc_mu.f(mu, i), hrf_model)
                q_scan_signal[:, i] = q_scan_signal_tmp.reshape((len(q_scan_signal_tmp,)))
                X[:, i] = q_scan_signal[:, i]

            for i in range(tc_sigma.N):
                sigma_signal_tmp, sigma_scan_signal_tmp, name, stim = \
                    self.get_bold_signal(exp, tc_sigma.f(sigma, i), hrf_model)
                sigma_scan_signal[:, i] = sigma_scan_signal_tmp.reshape((len(sigma_scan_signal_tmp,)))

                # Design matrix filling
                X[:, tc_mu.N+i] = sigma_scan_signal[:, i]

        elif coding_scheme.find('dpc')!=-1:

            tc_mu = tc[0]    # Tuning curves of interest for DPC

            # Initialization of the design matrix
            X = np.zeros((len(self.scan_times), tc_mu.N))
            scan_signal = np.zeros((len(self.scan_times), tc_mu.N))

            for i in range(tc_mu.N):
                # Projection calculation
                proj = tc_mu.compute_projection(exp.distributions, i, use_high_integration_resolution)
                signal_tmp, scan_signal_tmp, name, stim = \
                    self.get_bold_signal(exp, proj, hrf_model)
                scan_signal[:, i] = scan_signal_tmp.reshape((len(scan_signal_tmp,)))

                # Design matrix filling
                X[:, i] = scan_signal[:, i]

        # # Amplify the signal
        X = reg_fmri_gain * X

        return X