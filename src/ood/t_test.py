import numpy as np
from scipy.stats import t, tmean, zscore
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
    # plot_os_star(arr)
    os_star = np.mean(arr, axis=-1)
    low = np.quantile(os_star, 0.1)
    high = np.quantile(os_star, 0.9)
    os_star_cleaned = [os for os in os_star if low < os < high]
    mean = np.mean(os_star_cleaned)
    std = np.std(os_star_cleaned, ddof=1)
    dof = len(os_star_cleaned) - 1
    t_values = []
    for value in os_star:
        t_val = t_statistic(value, mean, std, len(os_star))
        t_values.append(t_val)
    t_values = zscore(t_values)
    p_values = []
    scale = np.std(t_values)
    for value in t_values:
        p_val = t.sf(value, df=dof, loc=0, scale=scale)  # survival function
        p_values.append(p_val)
    results = [t_values, p_values]
    return np.array(results).T


