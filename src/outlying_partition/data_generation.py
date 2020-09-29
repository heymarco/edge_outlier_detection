import numpy as np

from src.data.synthetic_data import create_raw_data, add_random_correlation, add_deviation, add_outlying_partitions


def create_dataset(num_devices: int = 100,
                   num_data: int = 1000,
                   dims: int = 100,
                   subspace_frac: float = 1.0,
                   frac_outlying_devices: float = 0.05,
                   cont: float = 1.0,
                   sigma_p: float = 0.25):
    sigma_l = 0.0

    data = create_raw_data(num_devices, num_data, dims)
    data = add_deviation(data, sigma=0)
    data, labels = add_outlying_partitions(data,
                                           frac_outlying_data=cont,
                                           frac_outlying_devices=frac_outlying_devices,
                                           subspace_frac=subspace_frac,
                                           sigma_p=sigma_p)
    data = add_random_correlation(data)
    labels = np.amax(labels, axis=-1)

    # write to file
    params_str = "{}_{}_{}_{}_{}_{}_{}".format(num_devices,
                                                   num_data,
                                                   dims,
                                                   subspace_frac,
                                                   frac_outlying_devices,
                                                   sigma_l,
                                                   sigma_p)

    return data, labels, params_str
