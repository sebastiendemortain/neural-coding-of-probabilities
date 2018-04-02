import numpy as np
import math

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