import os
import argparse

import numpy as np
from src.data.synthetic_data import create_raw_data, add_random_correlation, add_deviation, add_outlying_partitions

parser = argparse.ArgumentParser()
parser.add_argument("-sf", type=float, default=1.0)
parser.add_argument("-device_frac", type=float, default=0.05)
parser.add_argument("-cont", type=float, default=1.0)
parser.add_argument("-dims", type=int, default=100)
parser.add_argument("-dev", type=int, default=100)
parser.add_argument("-shift", type=float, default=1)
parser.add_argument("-dir", type=str, default="synth")
args = parser.parse_args()

# configuration
num_devices = args.dev
num_data = 1000
dims = args.dims
subspace_frac = args.sf
frac_outlying_devices = args.device_frac
frac_outlying_data = args.cont

data = create_raw_data(num_devices, num_data, dims)
data = add_deviation(data, sigma=0)
data, labels = add_outlying_partitions(data,
                                       frac_outlying_data=frac_outlying_data,
                                       frac_outlying_devices=frac_outlying_devices,
                                       subspace_frac=subspace_frac,
                                       sigma_p=args.shift)
data = add_random_correlation(data)

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
