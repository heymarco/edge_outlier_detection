import os
import argparse

import numpy as np

from src.data.synthetic_data import create_raw_data, add_random_correlation, add_global_outliers, add_local_outliers, add_deviation

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

parser = argparse.ArgumentParser()
parser.add_argument("-sf", type=float, default=0.2)
parser.add_argument("-dims", type=int, default=100)
parser.add_argument("-dev", type=int, default=100)
parser.add_argument("-frac_local", type=float, default=0.005)
parser.add_argument("-frac_global", type=float, default=0.005)
parser.add_argument("-dir", type=str, default="synth")
args = parser.parse_args()

# configuration
num_devices = args.dev
num_data = 1000
dims = args.dims
subspace_frac = args.sf
frac_outlying_devices = 1.0

# create local outliers
gamma = 1.2
delta = 0.3

subspace_size = int(subspace_frac * dims)

raw_data = create_raw_data(num_devices, num_data, dims)
data, labels_global = add_global_outliers(raw_data, subspace_size, frac_outlying=args.frac_global)
data = add_random_correlation(data)
data = add_deviation(data, gamma, delta)
data, labels_local = add_local_outliers(raw_data, subspace_size, args.frac_local)

# create labels
labels = labels_local.astype(np.int32)
labels[labels_global] = 2

# shuffle
for i in range(len(data)):
    shuffled_indices = np.arange(num_data)
    np.random.shuffle(shuffled_indices)
    data[i] = data[i][shuffled_indices]
    labels[i] = labels[i][shuffled_indices]

# write to file
params_str = "{}_{}_{}_{}_{}_{}_{}_{}_{}_mixed".format(num_devices,
                                                       num_data,
                                                       dims,
                                                       subspace_frac,
                                                       frac_outlying_devices,
                                                       args.frac_local,
                                                       args.frac_global,
                                                       gamma,
                                                       delta)
dataname = os.path.join(os.getcwd(), "data", args.dir, params_str + "_d")
outname = os.path.join(os.getcwd(), "data", args.dir, params_str + "_o")
np.save(dataname, data)
np.save(outname, np.amax(labels, axis=-1))

if subspace_frac == 1:
    for i, d in enumerate(data):
        plt.scatter(d.T[0], d.T[1], alpha=0.4)
        plt.scatter(d.T[0][labels[i].T[0] == 1], d.T[1][labels[i].T[1] == 1], color="black")
        plt.scatter(d.T[0][labels[i].T[0] == 2], d.T[1][labels[i].T[1] == 2], color="grey")
    plt.show()

print("Num outliers = {}".format((np.sum(np.amax(labels, axis=-1) > 0))))
