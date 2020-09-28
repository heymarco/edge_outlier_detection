import numpy as np
from scipy.stats import t, tmean, zscore, norm
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def t_statistic(sample, mean, std, n):
    standard_error = std / np.sqrt(n)
    return (sample-mean)/standard_error


def plot_os_star(arr):
    os_star = np.mean(arr, axis=-1)
    sns.distplot(os_star, bins=20)
    plt.show()


def evaluate_array_t_statistic(arr):
    os_star = np.mean(arr, axis=-1)
    os_star = zscore(os_star)
    mean = np.mean(os_star)
    std = np.std(os_star)
    props = [norm(mean, std).sf(val) for val in os_star]
    results = [os_star, props]
    print(np.mean(os_star))
    print(np.std(os_star))
    return np.array(results).T


