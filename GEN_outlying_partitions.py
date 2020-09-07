import os
import argparse

import numpy as np
from src.data.synthetic_data import create_raw_data, add_random_correlation, add_global_outliers, add_local_outliers, \
    add_deviation

parser = argparse.ArgumentParser()
parser.add_argument("-sf", type=float, default=1.0)
parser.add_argument("-device_frac", type=float, default=0.1)
parser.add_argument("-cont", type=float, default=0.5)
parser.add_argument("-dims", type=int, default=100)
parser.add_argument("-dev", type=int, default=100)
parser.add_argument("-shift", type=float, default=1.5)
parser.add_argument("-dir", type=str, default="synth")
args = parser.parse_args()

# configuration
num_devices = args.dev
num_data = 1000
dims = args.dims
subspace_frac = args.sf
frac_outlying_devices = args.device_frac
frac_outlying_data = args.cont

# create local outliers
sigma_l = 0

subspace_size = int(subspace_frac * dims)

data = create_raw_data(num_devices, num_data, dims)
# data, labels_global = add_global_outliers(data, subspace_size, frac_outlying=args.frac_global, sigma=sigma_g)
data = add_deviation(data, sigma_l)
# data, labels_local = add_local_outliers(data, subspace_size, args.frac_local)
data = add_random_correlation(data)

device_indices = np.random.choice(np.arange(num_devices), int(frac_outlying_devices * num_devices), replace=False)
subspace_size = int(subspace_frac * dims)
absolute_contamination = int(frac_outlying_data * num_data)

subspace = np.random.choice(np.arange(dims), subspace_size, replace=False)
point_indices = np.random.choice(np.arange(num_data), absolute_contamination, replace=False)

labels = np.empty(shape=data.shape, dtype=bool)
labels.fill(0)

for dev in device_indices:
    std = args.shift
    shift = np.random.choice([-std, std], size=subspace_size)
    labels[dev].fill(True)
    for p in point_indices:
        for i, s in enumerate(subspace):
            data[dev, p, s] = data[dev, p, s] + shift[i]

# write to file
params_str = "{}_{}_{}_{}_{}_{}_{}_ood".format(num_devices,
                                                        num_data,
                                                        dims,
                                                        subspace_frac,
                                                        frac_outlying_devices,
                                                        sigma_l,
                                                        args.shift)
dataname = os.path.join(os.getcwd(), "data", args.dir, params_str + "_d")
outname = os.path.join(os.getcwd(), "data", args.dir, params_str + "_o")
np.save(dataname, data)
np.save(outname, np.amax(labels, axis=-1))
