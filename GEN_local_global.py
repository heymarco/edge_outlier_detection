import argparse
import os
import logging

import numpy as np

from src.data.synthetic_data import create_raw_data, add_random_correlation
from src.data.synthetic_data import add_global_outliers, add_local_outliers, add_deviation


def create_data(num_devices: int = 100,
                num_data: int = 1000,
                dims: int = 100,
                subspace_frac: float = 0.2,
                frac_local: float = 0.01,
                frac_global: float = 0.01,
                sigma_l: float = 3.0,
                sigma_g: float = 3.0):
    subspace_size = int(subspace_frac * dims)
    # create data
    data = create_raw_data(num_devices, num_data, dims)
    data, labels_global = add_global_outliers(data, subspace_size, frac_outlying=frac_global, sigma=sigma_g)
    data = add_deviation(data, sigma_l)
    data, labels_local = add_local_outliers(data, subspace_size, frac_local)
    data = add_random_correlation(data)

    # create labels
    labels = labels_local.astype(np.int32)
    labels[labels_global] = 2
    labels = np.amax(labels, axis=-1)

    # shuffle
    for i in range(len(data)):
        shuffled_indices = np.arange(num_data)
        np.random.shuffle(shuffled_indices)
        data[i] = data[i][shuffled_indices]
        labels[i] = labels[i][shuffled_indices]

    params_str = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(num_devices,
                                                           num_data,
                                                           dims,
                                                           subspace_frac,
                                                           0.0,
                                                           frac_local,
                                                           frac_global,
                                                           sigma_l,
                                                           sigma_g)

    return data, labels, params_str
