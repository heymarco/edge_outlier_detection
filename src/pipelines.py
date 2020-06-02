import numpy as np

from src.data_ import *
from src.utils import normalize


def dataset_dummy(num_devices, n, dims, subspace_frac=0.1,
                  frac_outlying_devices=0.1, frac_outlying_data=0.7,
                  gamma=0.0, delta=0.0):
    data = np.random.uniform(low=0, high=1, size=dims)
    data = np.expand_dims(data, axis=0)
    data = np.repeat(data, n, axis=0)
    data = np.expand_dims(data, axis=0)
    data = np.repeat(data, num_devices, axis=0)
    out = np.empty(shape=data.shape)
    out.fill(False)
    return data, out


def dataset_abnormal_devices(num_devices, n, dims, subspace_frac=0.1,
                             frac_outlying_devices=0.1, frac_outlying_data=0.7,
                             gamma=0.0, delta=0.0):
    data = create_data(num_devices, n, dims, gamma=gamma, delta=delta)
    print(subspace_frac)
    outlying_devices = [i for i in range(int(frac_outlying_devices*num_devices))]
    data, is_outlier = add_abnormal_devices(data,
                                            indices=outlying_devices,
                                            subspace_size=int(dims * subspace_frac),
                                            frac_outlying=frac_outlying_data)
    return data, is_outlier


def dataset_outliers(num_devices, n, dims, subspace_frac=0.1,
                     frac_outlying_devices=0.1, frac_outlying_data=0.1,
                     gamma=0.0, delta=0.0):
    data = create_data(num_devices, n, dims, gamma=gamma, delta=delta)
    outlying_devices = [i for i in range(int(frac_outlying_devices*num_devices))]
    data, is_outlier = add_outliers(data,
                                    indices=outlying_devices,
                                    subspace_size=int(dims * subspace_frac),
                                    frac_outlying=frac_outlying_data)
    return data, is_outlier


def dataset_global_outliers(num_devices, n, dims, subspace_frac=0.1,
                     frac_outlying_devices=0.1, frac_outlying_data=0.1,
                     gamma=0.0, delta=0.0):
    data = create_data(num_devices, n, dims, gamma=gamma, delta=delta)
    outlying_devices = [i for i in range(int(frac_outlying_devices*num_devices))]
    print(subspace_frac)
    data, is_outlier = add_global_outliers(data,
                                           indices=outlying_devices,
                                           subspace_size=int(dims * subspace_frac),
                                           frac_outlying=frac_outlying_data)
    return data, is_outlier


def dataset_local_outliers(num_devices, n, dims, subspace_frac=0.1,
                           frac_outlying_devices=0.1, frac_outlying_data=0.1,
                           gamma=0.0, delta=0.0):
    data = create_data(num_devices, n, dims, gamma=gamma, delta=delta)
    outlying_devices = [i for i in range(int(frac_outlying_devices*num_devices))]
    data, is_outlier = add_local_outliers(data,
                                          indices=outlying_devices,
                                          subspace_size=int(dims * subspace_frac),
                                          frac_outlying=frac_outlying_data)
    return data, is_outlier


def dataset_hidden_outliers(num_devices, n, dims, subspace_frac=0.1,
                            frac_outlying_devices=0.1, frac_outlying_data=0.1,
                            gamma=0.0, delta=0.0):
    data = create_data(num_devices, n, dims, gamma=gamma, delta=delta)
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



