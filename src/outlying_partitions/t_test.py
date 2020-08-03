import numpy as np
from scipy.stats import t
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def t_statistic(sample, mean, std, n):
    return (sample-mean)/(std/(np.sqrt(n)))


def plot_os_star(arr):
    os_star = np.mean(arr, axis=-1)
    print(os_star)
    sns.distplot(os_star, bins=80)
    plt.show()


def evaluate_array_t_statistic(arr):
    os_star = np.mean(arr, axis=-1)
    mean = np.mean(os_star)
    std = np.std(os_star)
    results = []
    for value in os_star:
        t_val = t_statistic(value, mean, std, len(os_star))
        p_val = t.cdf(t_val)
        results.append([t_val, p_val])
    return np.array(results)


