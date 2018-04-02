import numpy as np

def import_distrib_param(data_mat):

    out = data_mat['out']
    out_HMM = out[0,0].HMM

    # The full distribution P(1|2) (resolution of 50 values and 600 trials)
    p1g2_dist = out_HMM[0,0].p1g2_dist

    n_val = np.size(p1g2_dist, 0)
    n_trial = np.size(p1g2_dist, 1)

    # The mean of this distribution
    p1g2_mean = out_HMM[0, 0].p1g2_mean

    # The standard deviation of this distribution
    p1g2_sd = out_HMM[0, 0].p1g2_sd
    p1g2_sd = p1g2_sd[0, :]

    return [p1g2_dist, p1g2_mean, p1g2_sd]
