import random as rand
import numpy as np

class brain:

    '''
    This class defines the properties of the brain we will generate data from.
    '''

    # Gain of the signal
    alpha = 1

    # Initialization of the attributes of the brain object
    def __init__(self, coding_scheme, n_voxel, n_population):
        self.n_voxel = n_voxel    # the number of voxels where the probability inference's activity takes place
        self.coding_scheme = coding_scheme   # the type of coding scheme
        self.n_population = n_population    # number of independent neural populations

        # Fraction of each neural population per voxel
        for k in range(n_population):
            # Random partition
            n_max_neuron = 1000*n_voxel    # Maximal number of neurons
            n_neuron = np.zeros(n_voxel)
            neuron_fraction = np.zeros([n_voxel, n_population])
            for voxel in range(n_voxel):
                n_neuron[voxel] = rand.randrange(n_max_neuron)
            n_total_neuron = np.sum(n_neuron)
            for voxel in range(n_voxel):
                neuron_fraction[voxel, k] = n_neuron[voxel]/n_total_neuron    # Fraction of neuron per voxel

        self.neuron_fraction = neuron_fraction

    # Neural activity per voxel
    def activity(self, voxel, q_map, sigma2_q, sigma2_map, sigma2_sigma):
        if self.coding_scheme=='rate':    # Rate coding case
            if self.n_population == 2:    # If the MAP and the variance are encoded by two independent populations
                '''
                We multiply the contribution of the uncertainty through a scaling factor (ratio of the std) in order 
                to have same neural impact between MAP and uncertainty
                '''
                activity = self.alpha * (self.neuron_fraction[voxel, 0]*q_map
                                    + np.sqrt(sigma2_map/sigma2_sigma)*self.neuron_fraction[voxel, 1]*sigma2_q)
                return activity
