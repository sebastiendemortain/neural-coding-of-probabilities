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
                        sigma = distrib.sigma  # Standard deviation of the distribution
                        activity[k_mean, k_sigma] = self.population_fraction[0]*self.subpopulation_fraction[0, 0]\
                                                    *q_mean + (sigma_mean/sigma_sd)*self.population_fraction[1]\
                                                              *self.subpopulation_fraction[1, 0]*sigma

        elif self.coding_scheme == 'ppc':    # Probabilistic population coding case
            # Activity due to mean encoding
            for i in range(self.tuning_curve[0].N):
                frac_mean = self.subpopulation_fraction[0, i]
                frac_sigma = self.subpopulation_fraction[1, i]
                f_mean = self.tuning_curve[0].f(x_mean, i)
                f_sigma = self.tuning_curve[1].f(x_sigma, i)
                for k_mean in range(n_mean):
                    for k_sigma in range(n_sigma):
                        activity[k_mean, k_sigma] += self.population_fraction[0]*frac_mean*f_mean[k_mean]\
                                                     + self.population_fraction[1]*frac_sigma*f_sigma[k_sigma]

        elif self.coding_scheme == 'dpc':    # Distributional population coding case
            res = 1e5    # Number of points used for the numerical integral
            eps = 0  # Reduction of the interval for dealing with boundary issues of the beta function
            # x-axis for integration
            x = np.linspace(self.tuning_curve[0].lower_bound+eps, self.tuning_curve[0].upper_bound-eps, res)
            x = np.delete(x, 0)    # res-1 points shall be considered for the numerical integration
            delta_x = (self.tuning_curve[0].upper_bound-self.tuning_curve[0].lower_bound-2*eps)/(res-2)    # Integral step

            # Double list containing the array of the beta function at the desired resolution
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
        # Multiplication by the gain of the signal
        activity = self.alpha * activity
        return activity
