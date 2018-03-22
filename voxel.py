import random as rand
import numpy as np
from scipy import stats
from scipy import integrate

class voxel:

    '''
    This class defines the properties of the voxel we will generate data from.
    '''

    # Gain of the signal
    alpha = 1

    # Initialization of the attributes of the voxel object
    def __init__(self, coding_scheme, n_population, n_subpopulation, tc = [], sparsity_exp = 1):
        self.coding_scheme = coding_scheme   # the type of coding scheme
        self.n_population = n_population    # number of independent neural populations
        self.sparsity_exp = sparsity_exp    # Exponent controlling the sparsity of the neuron fraction
        # Fraction of each neural population
        neuron_fraction = np.zeros([n_population, n_subpopulation])
        n_neuron = np.zeros([n_population, n_subpopulation])    # Number of neuron per subpopulation
        n_total_neuron = np.zeros(n_population)
        for pop in range(n_population):
            for subpop in range(n_subpopulation):
                # Number of neuron per subpopulation controlled by a sparsity exponent
                n_neuron[pop, subpop] = (rand.randrange(n_subpopulation)+1)**sparsity_exp
            n_total_neuron[pop] = np.sum(n_neuron)
        for pop in range(n_population):
            for subpop in range(n_subpopulation):
                # Fraction of neuron for each population. It sums up to 1 over each subpopulation
                neuron_fraction[pop, subpop] = n_neuron[pop, subpop]/n_total_neuron[pop]

        self.neuron_fraction = neuron_fraction

        self.tuning_curve = tc
        # If Probabilistic population code or distributional population code, we define one tuning curves' list
        if self.coding_scheme == 'ppc':
            self.tuning_curve[0] = tc[0]
            self.tuning_curve[1] = tc[1]
        elif self.coding_scheme == 'dpc':
            self.tuning_curve[0] = tc[0]

    # Neural activity given one distribution
    def activity(self, distrib_array, x_mean, x_sigma=[np.nan], sigma_mean=np.nan, sigma_sd=np.nan):
        n_mean = len(x_mean)
        n_sigma = len(x_sigma)

        activity = np.zeros([len(distrib_array), len(distrib_array[0])])  # Initialization

        if self.coding_scheme == 'rate':    # Rate coding case
            if self.n_population == 2:    # If the mean and the variance are encoded by two independent populations
                # Activity due to mean and uncertainty encoding (linear combination). Only one neuron per subpopulation
                '''
                We multiply the contribution of the uncertainty through a scaling factor (ratio of the std) in order 
                to have same neural impact between mean and uncertainty
                '''
                for k_mean in range(len(distrib_array)):
                    for k_sigma in range(len(distrib_array[0])):
                        distrib = distrib_array[k_mean][k_sigma]
                        q_mean = distrib.mean  # Mean of the distribution
                        sigma_q = distrib.sd  # Standard deviation of the distribution
                        activity[k_mean, k_sigma] = self.neuron_fraction[0, 0]*q_mean \
                                                    + (sigma_mean/sigma_sd)*self.neuron_fraction[1, 0]*sigma_q

        elif self.coding_scheme == 'ppc':    # Probabilistic population coding case
            # Activity due to mean encoding
            for i in range(self.tuning_curve[0].N):
                frac_mean = self.neuron_fraction[0, i]
                frac_sigma = self.neuron_fraction[1, i]
                f_mean = self.tuning_curve[0].f(x_mean, i)
                f_sigma = self.tuning_curve[1].f(x_sigma, i)
                for k_mean in range(n_mean):
                    for k_sigma in range(n_sigma):
                        activity[k_mean, k_sigma] += frac_mean*f_mean[k_mean]+frac_sigma*f_sigma[k_sigma]

        elif self.coding_scheme == 'dpc':    # Distributional population coding case
            res = 10000    # Number of points used for the numerical integral
            delta_x = (self.tuning_curve[0].upper_bound-self.tuning_curve[0].lower_bound)/(res-1)    # Integral step
            # x-axis for integration
            x = np.linspace(self.tuning_curve[0].lower_bound, self.tuning_curve[0].upper_bound, res)
            x = np.delete(x, -1)    # res-1 points shall be considered for the numerical integration

            # Double list containing the array of the beta function at the desired resolution
            beta = [[None for j in range(n_sigma)] for i in range(n_mean)]
            for k_mean in range(n_mean):
                for k_sigma in range(n_sigma):
                    distrib = distrib_array[k_mean][k_sigma]
                    beta[k_mean][k_sigma] = distrib.beta(x)

            # Activity due to the projection on the tuning curves
            for i in range(self.tuning_curve[0].N):
                # Projection of the distribution on tuning curve i
                f = self.tuning_curve[0].f(x, i)    # tuning curve i's values along the x-axis
                frac = self.neuron_fraction[0, i]
                for k_mean in range(n_mean):
                    for k_sigma in range(n_sigma):
                        proj = np.dot(beta[k_mean][k_sigma], f)*delta_x
                        # proj_num = integrate.quad(lambda x: distrib.beta(x)*self.tuning_curve[0].f(x, i), self.tuning_curve[0].lower_bound, self.tuning_curve[0].upper_bound)[0]

                        activity[k_mean, k_sigma] += frac*proj

        # Multiplication by the gain of the signal
        activity = self.alpha * activity
        return activity
