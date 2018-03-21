from scipy import stats

class distrib:

    '''This class specifies attributes of a specific distribution'''

    def __init__(self, q_mean, sigma_q):
        self.mean = q_mean    # Mean of the distribution
        self.sd = sigma_q    # Standard deviation of the distribution
        self.a = ((1-self.mean)/self.sd**2-1/self.mean)*self.mean**2    # First parameter of to build the beta distribution
        self.b = ((1-self.mean)/self.sd**2-1/self.mean)*self.mean**2 * (1/self.mean-1)    # Second parameter to build the beta distribution

    # Equivalent beta distribution
    def beta(self, x):
        return stats.beta.pdf(x, self.a, self.b)
