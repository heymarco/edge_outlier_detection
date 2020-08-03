import os
import argparse

import numpy as np
from src.data.synthetic_data import create_data

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

parser = argparse.ArgumentParser()
parser.add_argument("-sf", type=float, default=0.2)
parser.add_argument("-device_frac", type=float, default=0.1)
parser.add_argument("-cont", type=float, default=0.5)
parser.add_argument("-dims", type=int, default=100)
parser.add_argument("-dev", type=int, default=100)
parser.add_argument("-shift", type=float, default=1.5)
args = parser.parse_args()

# configuration
num_devices = args.dev
num_data = 1000
dims = args.dims
subspace_frac = args.sf
frac_outlying_devices = args.device_frac
frac_outlying_data = args.cont

# create local outliers
gamma = 0.5
delta = 0.3

data = create_data(num_devices, num_data, dims, gamma, delta)

device_indices = np.random.choice(np.arange(num_devices), int(frac_outlying_devices*num_devices), replace=False)
subspace_size = int(subspace_frac*dims)
absolute_contamination = int(frac_outlying_data*num_data)

std = args.shift
shift = np.random.choice([-std, std], size=subspace_size)

subspace = np.random.choice(np.arange(dims), subspace_size, replace=False)
point_indices = np.random.choice(np.arange(num_data), absolute_contamination, replace=False)

labels = np.empty(shape=data.shape, dtype=bool)
labels.fill(0)

for dev in device_indices:
    labels[dev].fill(True)
    for p in point_indices:
        for i, s in enumerate(subspace):
            data[dev, p, s] = data[dev, p, s] + shift[i]

# write to file
params_str = "{}_{}_{}_{}_{}_{}_{}_{}_mixed".format(num_devices,
                                                    num_data,
                                                    dims,
                                                    subspace_frac,
                                                    frac_outlying_devices,
                                                    frac_outlying_data,
                                                    gamma,
                                                    delta)
dataname = os.path.join(os.getcwd(), "data", "out_part", params_str + "_d")
outname = os.path.join(os.getcwd(), "data", "out_part", params_str + "_o")
np.save(dataname, data)
np.save(outname, np.amax(labels, axis=-1))

# for i, d in enumerate(data):
#     if labels[i].any():
#         plt.scatter(d.T[0], d.T[1], alpha=0.4, color="black")
#     else:
#         plt.scatter(d.T[0], d.T[1], alpha=0.4)
    # plt.scatter(d.T[0][labels[i].T[0] == 1], d.T[1][labels[i].T[1] == 1], color="black")
    # plt.scatter(d.T[0][labels[i].T[0] == 2], d.T[1][labels[i].T[1] == 2], color="grey")
# plt.show()

print("Num outliers = {}".format((np.sum(np.amax(labels, axis=-1) > 0))))
