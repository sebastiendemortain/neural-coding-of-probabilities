import random as rand
import numpy as np
from scipy import stats
from scipy import integrate
import utils


'''Related to the distribution'''


def import_distrib_param(data_mat):
    out = data_mat['out']
    out_HMM = out[0, 0].HMM

    # The full distribution P(1|2) (resolution of 50 values and 600 trials)
    p1g2_dist = out_HMM[0, 0].p1g2_dist

    n_val = np.size(p1g2_dist, 0)
    n_trial = np.size(p1g2_dist, 1)

    # The mean of this distribution
    p1g2_mean = out_HMM[0, 0].p1g2_mean

    # The standard deviation of this distribution
    p1g2_sd = out_HMM[0, 0].p1g2_sd
    p1g2_sd = p1g2_sd[0, :]

    return [p1g2_dist, p1g2_mean, p1g2_sd]


class distrib:

    '''This class specifies attributes of a specific distribution'''

    def __init__(self, q_mean, sigma):
        self.mean = q_mean    # Mean of the distribution
        self.sd = sigma    # Standard deviation of the distribution
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
            mu = self.lower_bound+i*delta_mu
            # Variance of the tuning curve
            sigma2_f = self.t**2
            tc_value = np.exp(-0.5*(x-mu)**2/sigma2_f)
            return tc_value
        else:
            return

class voxel:

    '''
    This class defines the properties of the voxel we will generate data from.
    '''

    # Gain of the signal due to neural activity fluctuation
    gain_neural = 1

    # Initialization of the attributes of the voxel object
    def __init__(self, coding_scheme, population_fraction, tc = [], sparsity_exp = 1):
        self.coding_scheme = coding_scheme   # the type of coding scheme
        self.population_fraction = population_fraction
        self.n_population = len(self.population_fraction)    # number of independent neural populations
        self.sparsity_exp = sparsity_exp    # Exponent controlling the sparsity of the neuron fraction

        self.tuning_curve = tc
        # If Probabilistic population code or distributional population code, we define one tuning curves' list
        if self.coding_scheme == 'ppc':
            for pop in range(self.n_population):
                self.tuning_curve[pop] = tc[pop]
            # Number of neurons within one population (shall be the same for all populations of PPC) : number of TC
            self.n_subpopulation = self.tuning_curve[0].N
        elif self.coding_scheme == 'dpc':
            for pop in range(self.n_population):
                self.tuning_curve[pop] = tc[pop]
            self.n_subpopulation = self.tuning_curve[0].N    # this is the number of tuning curve
        elif self.coding_scheme == 'rate':
            self.n_subpopulation = 1    # By definition of rate coding

        # Fraction of each neural population
        subpopulation_fraction = np.zeros([self.n_population, self.n_subpopulation])
        n_neuron = np.zeros([self.n_population, self.n_subpopulation])    # Number of neuron per subpopulation
        n_total_neuron = np.zeros(self.n_population)
        for pop in range(self.n_population):
            for subpop in range(self.n_subpopulation):
                # Number of neuron per subpopulation controlled by a sparsity exponent
                n_neuron[pop, subpop] = (rand.randrange(self.n_subpopulation)+1)**sparsity_exp
            n_total_neuron[pop] = np.sum(n_neuron[pop, :])
        for pop in range(self.n_population):
            for subpop in range(self.n_subpopulation):
                # Fraction of neuron for each population. It sums up to 1 over each subpopulation
                subpopulation_fraction[pop, subpop] = n_neuron[pop, subpop]/n_total_neuron[pop]

        self.subpopulation_fraction = subpopulation_fraction

    # Neural activity given one distribution
    def generate_activity(self, distrib_array, sigma_mean=np.nan, sigma_sd=np.nan):
        # 2 if 2D-grid for plotting the signal, 1 if one single continuous experiment
        n_dims = utils.get_dimension_list(distrib_array)
        if n_dims == 2:
            n_mean = len(distrib_array)
            n_sigma = len(distrib_array[0])
            activity = np.zeros([n_mean, n_sigma])  # Initialization
        elif n_dims == 1:
            n_stimuli = len(distrib_array)
            activity = np.zeros(n_stimuli)  # Initialization

        if self.coding_scheme == 'rate':    # Rate coding case
            if self.n_population == 2:    # If the mean and the variance are encoded by two independent populations
                # Activity due to mean and uncertainty encoding (linear combination). Only one neuron per subpopulation
                '''
                We multiply the contribution of the uncertainty through a scaling factor (ratio of the std) in order 
                to have same neural impact between mean and uncertainty
                '''
                # Case of double array (when 2D-grid of distributions)
                if n_dims == 2:
                    for k_mean in range(n_mean):
                        for k_sigma in range(n_sigma):
                            distrib = distrib_array[k_mean][k_sigma]
                            q_mean = distrib.mean  # Mean of the distribution
                            sigma = distrib.sd  # Standard deviation of the distribution
                            activity[k_mean, k_sigma] = self.population_fraction[0]*self.subpopulation_fraction[0, 0]\
                                                        *q_mean + (sigma_mean/sigma_sd)*self.population_fraction[1]\
                                                                  *self.subpopulation_fraction[1, 0]*sigma
                # Case of single array (when 1D-grid of distributions)
                elif n_dims == 1:
                    for k in range(n_stimuli):
                        distrib = distrib_array[k]
                        q_mean = distrib.mean  # Mean of the distribution
                        sigma = distrib.sd  # Standard deviation of the distribution
                        activity[k] = self.population_fraction[0] * self.subpopulation_fraction[0, 0] \
                                                    * q_mean + (sigma_mean / sigma_sd) * self.population_fraction[1] \
                                                               * self.subpopulation_fraction[1, 0] * sigma

        elif self.coding_scheme == 'ppc':    # Probabilistic population coding case

            if n_dims == 2:
                # Defines the 2D-grid of means and sigma
                x_mean = np.zeros(n_mean)
                x_sigma = np.zeros(n_sigma)
                for k_mean in range(n_mean):
                    distrib = distrib_array[k_mean][0]
                    x_mean[k_mean] = distrib.mean
                for k_sigma in range(n_sigma):
                    distrib = distrib_array[0][k_sigma]
                    x_sigma[k_sigma] = distrib.sd

                for i in range(self.tuning_curve[0].N):
                    frac_mean = self.subpopulation_fraction[0, i]
                    frac_sigma = self.subpopulation_fraction[1, i]
                    f_mean = self.tuning_curve[0].f(x_mean, i)
                    f_sigma = self.tuning_curve[1].f(x_sigma, i)
                    for k_mean in range(n_mean):
                        for k_sigma in range(n_sigma):
                            activity[k_mean, k_sigma] += self.population_fraction[0]*frac_mean*f_mean[k_mean]\
                                                         + self.population_fraction[1]*frac_sigma*f_sigma[k_sigma]

            elif n_dims == 1:
                # Defines the 1D-grid of (mean, sigma)
                x_mean = np.zeros(n_stimuli)
                x_sigma = np.zeros(n_stimuli)
                for k in range(n_stimuli):
                    distrib = distrib_array[k]
                    x_mean[k] = distrib.mean
                    x_sigma[k] = distrib.sd

                for i in range(self.tuning_curve[0].N):
                    frac_mean = self.subpopulation_fraction[0, i]
                    frac_sigma = self.subpopulation_fraction[1, i]
                    f_mean = self.tuning_curve[0].f(x_mean, i)
                    f_sigma = self.tuning_curve[1].f(x_sigma, i)
                    for k in range(n_stimuli):
                        activity[k] += self.population_fraction[0]*frac_mean*f_mean[k]\
                                                         + self.population_fraction[1]*frac_sigma*f_sigma[k]

        elif self.coding_scheme == 'dpc':    # Distributional population coding case
            res = 1e5    # Number of points used for the numerical integral
            eps = 0  # Reduction of the interval for dealing with boundary issues of the beta function
            # x-axis for integration
            x = np.linspace(self.tuning_curve[0].lower_bound+eps, self.tuning_curve[0].upper_bound-eps, res)
            x = np.delete(x, 0)    # res-1 points shall be considered for the numerical integration
            delta_x = (self.tuning_curve[0].upper_bound-self.tuning_curve[0].lower_bound-2*eps)/(res-2)    # Integral step

            # Double list containing the array of the beta function at the desired resolution
            if n_dims == 2:
                beta = [[None for j in range(n_sigma)] for i in range(n_mean)]
                for k_mean in range(n_mean):
                    for k_sigma in range(n_sigma):
                        distrib = distrib_array[k_mean][k_sigma]
                        beta[k_mean][k_sigma] = distrib.beta(x)

                # sum_err = 0
                # Activity due to the projection on the tuning curves
                for i in range(self.tuning_curve[0].N):
                    # Projection of the distribution on tuning curve i
                    f = self.tuning_curve[0].f(x, i)    # tuning curve i's values along the x-axis
                    frac = self.subpopulation_fraction[0, i]
                    for k_mean in range(n_mean):
                        for k_sigma in range(n_sigma):
                            # distrib = distrib_array[k_mean][k_sigma]
                            proj = np.dot(beta[k_mean][k_sigma], f)*delta_x
                            # proj_num = integrate.quad(lambda y: distrib.beta(y)*self.tuning_curve[0].f(y, i), self.tuning_curve[0].lower_bound+eps, self.tuning_curve[0].upper_bound-eps)[0]
                            activity[k_mean, k_sigma] += frac*proj
                            # sum_err += np.abs(proj-proj_num)/np.sqrt(np.dot(proj,proj))
                            # if np.abs(proj-proj_num) > 0.01
                            #     print('ISSUE WITH THE NUMERICAL INTEGRATION. See the dpc case in voxel.activity')
                    # print(sum_err)
            # Single list containing the array of the beta function at the desired resolution
            elif n_dims == 1:
                beta = [None for k in range(n_stimuli)]
                for k in range(n_stimuli):
                    distrib = distrib_array[k]
                    beta[k] = distrib.beta(x)

                # sum_err = 0
                # Activity due to the projection on the tuning curves
                for i in range(self.tuning_curve[0].N):
                    # Projection of the distribution on tuning curve i
                    f = self.tuning_curve[0].f(x, i)  # tuning curve i's values along the x-axis
                    frac = self.subpopulation_fraction[0, i]
                    for k in range(n_stimuli):
                            proj = np.dot(beta[k], f) * delta_x
                            # proj_num = integrate.quad(lambda y: distrib.beta(y)*self.tuning_curve[0].f(y, i), self.tuning_curve[0].lower_bound+eps, self.tuning_curve[0].upper_bound-eps)[0]
                            activity[k] += frac * proj
                            # sum_err += np.abs(proj-proj_num)/np.sqrt(np.dot(proj,proj))
                            # if np.abs(proj-proj_num) > 0.01
                            #     print('ISSUE WITH THE NUMERICAL INTEGRATION. See the dpc case in voxel.activity')
                            # print(sum_err)

        # Multiplication by the gain of the signal
        activity = self.gain_neural * activity
        return activity


class fmri:


    ''' This class defines the MRI session modalities '''

    # SNR of the signal due to fMRI noise
    snr = 1

    def __init___(self, n_blocks, n_stimuli_per_block, time_between_stimuli, time_between_scans, dt):
        self.n_blocks = n_blocks
        self.n_stimuli_per_block = n_stimuli_per_block
        self.time_between_stimuli = time_between_stimuli
        self.time_between_scans = time_between_scans
        self.dt = dt    # Temporal resolution

    #def generate_bold(self, activity):
