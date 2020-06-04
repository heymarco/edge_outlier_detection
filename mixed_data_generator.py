import os
import argparse

import math
import numpy as np
from scipy.stats import truncnorm
from src.pipelines import dataset_local_outliers
from src.data_ import add_global_outliers, add_local_outliers, create_data

import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

parser = argparse.ArgumentParser()
parser.add_argument("-sf", type=float, default=0.5)
args = parser.parse_args()

# configuration
num_devices = 10
num_data = 1000
dims = 100
subspace_frac = args.sf
frac_outlying_devices = 1.0
frac_outlying_data = 0.01

# create local outliers
gamma = 0.5
delta = 0.3

raw_data = create_data(num_devices, num_data, dims, gamma, delta)

device_indices = np.random.choice(np.arange(num_devices), int(frac_outlying_devices*num_devices), replace=False)


def add_mixed_outliers(data, indices, subspace_size, frac_outlying_data):
    data, labels_local = add_local_outliers(data, indices, subspace_size, frac_outlying_data/2)
    data, labels_global = add_global_outliers(data, indices, subspace_size, frac_outlying_data/2)
    return data, labels_global, labels_local


def create_dataset(raw_data, indices, subspace_size, frac_outlying_data):
    data, labels_global, labels_local = add_mixed_outliers(raw_data, indices, subspace_size, frac_outlying_data)
    labels = labels_local.astype(np.int32)
    labels[labels_global] = 2
    if np.sum(np.amax(labels, axis=-1) > 0) != int(num_devices*num_data*frac_outlying_data):
        create_dataset(raw_data, indices, subspace_size, frac_outlying_data)
    else:
        # shuffle
        for i in range(len(data)):
            shuffled_indices = np.arange(num_data)
            np.random.shuffle(shuffled_indices)
            data[i] = data[i][shuffled_indices]
            labels[i] = labels[i][shuffled_indices]

        # write to file
        params_str = "{}_{}_{}_{}_{}_{}_{}_{}_mixed".format(num_devices,
                                                            num_data,
                                                            dims,
                                                            subspace_frac,
                                                            frac_outlying_devices,
                                                            frac_outlying_data,
                                                            gamma,
                                                            delta)
        dataname = os.path.join(os.getcwd(), "data", "synth", params_str + "_d")
        outname = os.path.join(os.getcwd(), "data", "synth", params_str + "_o")
        np.save(dataname, data)
        np.save(outname, np.amax(labels, axis=-1))

        # if subspace_frac == 1:
        #     for i, d in enumerate(data):
        #         plt.scatter(d.T[0], d.T[1], alpha=0.4)
        #         plt.scatter(d.T[0][labels[i].T[0] == 1], d.T[1][labels[i].T[1] == 1], color="black")
        #         plt.scatter(d.T[0][labels[i].T[0] == 2], d.T[1][labels[i].T[1] == 2], color="grey")
        #     plt.show()


for i in range(5, 101, 5):
    subspace_frac = i/100
    subspace_size = int(subspace_frac * dims)
    create_dataset(raw_data, device_indices, subspace_size, frac_outlying_data)