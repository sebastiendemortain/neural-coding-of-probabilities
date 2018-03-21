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
                n_neuron[pop, subpop] = rand.randrange(n_subpopulation)**sparsity_exp
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
        a = distrib.a    # First parameter of the beta distribution
        b = distrib.b    # Second parameter of the beta distribution

        activity = 0  # Initialization

        if self.coding_scheme == 'rate':    # Rate coding case
            if self.n_population == 2:    # If the mean and the variance are encoded by two independent populations
                # Activity due to mean and uncertainty encoding (linear combination). Only one neuron per subpopulation
                '''
                We multiply the contribution of the uncertainty through a scaling factor (ratio of the std) in order 
                to have same neural impact between mean and uncertainty
                '''
                activity = self.neuron_fraction[0, 0]*q_mean + (sigma_mean/sigma_sd)*self.neuron_fraction[1, 0]*sigma_q

        elif self.coding_scheme == 'ppc':    # Probabilistic population coding case
                # Activity due to mean encoding
                for i in range(self.tuning_curve[0].N):
                        activity += self.neuron_fraction[0, i]*self.tuning_curve[0].f(q_mean, i)
                # Activity due to uncertainty encoding
                for i in range(self.tuning_curve[1].N):
                    activity += self.neuron_fraction[1, self.tuning_curve[0].N+i]*self.tuning_curve[1].f(sigma_q, i)

        elif self.coding_scheme == 'dpc':
                proj = np.zeros(self.tuning_curve[0].N)    # Initialization of the projections onto the tuning curves
                # Activity due to the projection on the tuning curves
                for i in range(self.tuning_curve[0].N):
                    # A MODIFIERRRRR
                    proj[i] = integrate.quad(lambda x: stats.beta.pdf(x, a, b)*self.tuning_curve[0].f(x, i), self.tuning_curve[0].lower_bound, self.tuning_curve[0].upper_bound)[0]
                    activity += self.neuron_fraction[0, i]*proj[i]

        # Multiplication by the gain of the signal
        activity = self.alpha * activity
        return activity
