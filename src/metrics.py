import numpy as np


def kappa_m(labels, ground_truth, contamination):
    p_0 = np.sum(labels == ground_truth)/len(labels)
    p_c = 1-contamination
    return (p_0-p_c)/(1-p_c)

