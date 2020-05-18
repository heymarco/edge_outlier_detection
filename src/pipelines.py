import numpy as np

from src.data_ import create_data, alter_device_data, add_outliers, add_hidden_outliers
from src.utils import normalize


def dataset_outliers(num_devices, n, dims, subspace_frac=0.1,
                     frac_outlying_devices=0.1, frac_outlying_data=0.1,
                     gamma=0.0, delta=0.0):
    data, params = create_data(num_devices, n, dims, gamma=gamma, delta=delta)
    outlying_devices = [i for i in range(int(frac_outlying_devices*num_devices))]
    data, is_outlier = add_outliers(data,
                                    indices=outlying_devices,
                                    subspace_size=int(dims * subspace_frac),
                                    frac_outlying=frac_outlying_data,
                                    params=params)
    return data, is_outlier


def dataset_hidden_outliers(num_devices, n, dims, subspace_frac=0.1,
                            frac_outlying_devices=0.1, frac_outlying_data=0.1,
                            gamma=0.0, delta=0.0):
    data, params = create_data(num_devices, n, dims, gamma=gamma, delta=delta)
    outlying_devices = np.array([i for i in range(int(num_devices*frac_outlying_devices))])
    data, is_outlier = add_hidden_outliers(data,
                                           indices=outlying_devices,
                                           subspace_size=int(dims * subspace_frac),
                                           frac_outlying=frac_outlying_data)
    return data, is_outlier


def dataset_drift(num_devices, n, dims, drift_start, drift_end, total_drift=0.2,
                  frac_drifting_devices=0.1, subspace_frac=0.1):
    assert drift_end-drift_start < n
    amount = total_drift/(drift_end-drift_start)
    indices = [i for i in range(int(num_devices * frac_drifting_devices))]
    subspace_size = int(dims*subspace_frac)
    subspaces = [np.random.choice(range(dims), subspace_size, replace=False) for _ in range(len(indices))]
    data = create_data(num_devices, n, dims)
    t_drift = np.array([i for i in range(n)])[drift_start: drift_end]
    for t in t_drift:
        # concept drift stays present in data on device
        d = data[:, t:]
        d = alter_device_data(d, max_shift=amount, indices=indices, subspaces=subspaces)
        data[:, t:] = d
    is_outlier = np.empty(shape=data.shape)
    is_outlier.fill(False)
    is_outlier[indices, drift_start:drift_end] = True
    return normalize(data), is_outlier


def dataset_constant_deviation(num_devices, n, dims, drift_start, drift_end, total_drift=0.2,
                               frac_drifting_devices=0.1, subspace_frac=0.1):
    assert drift_end-drift_start <= n
    indices = [i for i in range(int(num_devices * frac_drifting_devices))]
    subspace_size = int(dims * subspace_frac)
    subspaces = [np.random.choice(range(dims), subspace_size, replace=False) for _ in range(len(indices))]
    data = create_data(num_devices, n, dims)
    t_drift = np.array([i for i in range(n)])[drift_start: drift_end]
    for t in t_drift:
        # concept drift stays present in data on device
        d = data[:, t:t+1]
        d = alter_device_data(d, max_shift=total_drift, indices=indices, subspaces=subspaces)
        data[:, t:t+1] = d
    is_outlier = np.empty(shape=data.shape)
    is_outlier.fill(False)
    is_outlier[indices, drift_start:drift_end] = True
    return normalize(data), is_outlier



