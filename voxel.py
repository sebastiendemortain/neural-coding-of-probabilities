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
    def activity(self, distrib, sigma_mean=0, sigma_sd=0):
        q_mean = distrib.mean    # Mean of the distribution
        sigma_q = distrib.sd    # Standard deviation of the distribution

        activity = 0  # Initialization

        if self.coding_scheme == 'rate':    # Rate coding case
            if self.n_population == 2:    # If the mean and the variance are encoded by two independent populations
                # Activity due to mean and uncertainty encoding (linear combination). Only one neuron per subpopulation
                '''
                We multiply the contribution of the uncertainty through a scaling factor (ratio of the std) in order 
                to have same neural impact between mean and uncertainty
                '''
                #for k_mean in range[]
                activity = self.neuron_fraction[0, 0]*q_mean + (sigma_mean/sigma_sd)*self.neuron_fraction[1, 0]*sigma_q

        elif self.coding_scheme == 'ppc':    # Probabilistic population coding case
            # Activity due to mean encoding
            for i in range(self.tuning_curve[0].N):
                activity += self.neuron_fraction[0, i]*self.tuning_curve[0].f(q_mean, i)
                # Activity due to uncertainty encoding
            for i in range(self.tuning_curve[1].N):
                activity += self.neuron_fraction[1, i]*self.tuning_curve[1].f(sigma_q, i)

        elif self.coding_scheme == 'dpc':    # Distributional population coding case
            proj_num = np.zeros(self.tuning_curve[0].N)    # Initialization of the projections onto the tuning curves
            proj = np.zeros(self.tuning_curve[0].N)
            res = 10000    # Number of points used for the discretized integral computation
            delta_x = (self.tuning_curve[0].upper_bound-self.tuning_curve[0].lower_bound)/(res-1)    # Integral step
            x = np.linspace(self.tuning_curve[0].lower_bound, self.tuning_curve[0].upper_bound, res)     # x-axis
            beta = distrib.beta(x)
            # Activity due to the projection on the tuning curves
            for i in range(self.tuning_curve[0].N):
                # Projection of the distribution on tuning curve i
                # proj_num[i] = integrate.quad(lambda x: distrib.beta(x)*self.tuning_curve[0].f(x, i), self.tuning_curve[0].lower_bound, self.tuning_curve[0].upper_bound)[0]
                f = self.tuning_curve[0].f(x, i)    # tuning curve vector
                for k_x in range(res-1):
                    proj[i] += beta[k_x]*f[k_x]*delta_x
                activity += self.neuron_fraction[0, i]*proj[i]

        # Multiplication by the gain of the signal
        activity = self.alpha * activity
        return activity
