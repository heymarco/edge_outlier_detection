import argparse
import os
import logging

import numpy as np

from src.data.synthetic_data import create_raw_data, add_random_correlation, add_global_outliers, add_local_outliers, add_deviation

parser = argparse.ArgumentParser()
parser.add_argument("-sf", type=float, default=0.2)
parser.add_argument("-dims", type=int, default=100)
parser.add_argument("-dev", type=int, default=100)
parser.add_argument("-frac_local", type=float, default=0.01)
parser.add_argument("-frac_global", type=float, default=0.01)
parser.add_argument("-dir", type=str, default="synth")
args = parser.parse_args()

# configuration
num_devices = args.dev
num_data = 1000
dims = args.dims
subspace_frac = args.sf
frac_outlying_devices = 1.0

# create local outliers
sigma_g = 3
sigma_l = 3

subspace_size = int(subspace_frac * dims)

data = create_raw_data(num_devices, num_data, dims)
data, labels_global = add_global_outliers(data, subspace_size, frac_outlying=args.frac_global, sigma=sigma_g)
data = add_deviation(data, sigma_l)
data, labels_local = add_local_outliers(data, subspace_size, args.frac_local)
data = add_random_correlation(data)

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
                                                       sigma_l,
                                                       sigma_g)
dataname = os.path.join(os.getcwd(), "data", args.dir, params_str + "_d")
outname = os.path.join(os.getcwd(), "data", args.dir, params_str + "_o")
np.save(dataname, data)
np.save(outname, np.amax(labels, axis=-1))

logging.info("Num outliers = {}".format((np.sum(np.amax(labels, axis=-1) > 0))))
