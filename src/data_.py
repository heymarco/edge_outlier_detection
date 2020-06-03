import numpy as np
from scipy.stats import truncnorm
import math
import random

from src.utils import normalize


def create_data(num_devices, n, dims, gamma, delta):

    def create_blueprint(size):
        return np.random.uniform(low=-1, high=1, size=size)

    def create_deviation(size, dev):
        return np.random.normal(size=size)*dev

    def create_gaussian_noise(size, snr=0.2):
        return np.random.normal(size=size)*snr

    data = create_blueprint(dims)
    data = np.expand_dims(data, axis=0)
    data = np.repeat(data, num_devices, axis=0)
    deviation = np.array([create_deviation(dims, gamma) for _ in range(num_devices)])
    data = data + deviation
    data = np.expand_dims(data, axis=1)
    data = np.repeat(data, n, axis=1)

    noise = create_gaussian_noise(size=data.shape, snr=delta)
    data = data + noise

    return data


def alter_device_data(data, max_shift, indices=[], subspaces=[]):
    if len(indices) == 0:
        indices = range(len(data))
    relevant_data = data[indices]
    if len(subspaces) == 0:
        subspaces = [range(len(data[i][0])) for i in range(len(indices))]
    else:
        assert len(subspaces) == len(indices)
    for i, timeseries in enumerate(relevant_data):
        subspace = subspaces[i]
        alpha = 0.2
        noise = (1.0-alpha) + alpha*np.random.uniform(size=len(subspace))
        noise = np.expand_dims(noise, axis=0)
        noise = np.repeat(noise, len(timeseries), axis=0)
        relevant_data[i][:, subspace] = timeseries[:, subspace] + noise*max_shift
    data[indices] = relevant_data
    return data


def add_hidden_outliers(data, indices, subspace_size, frac_outlying=0.05):
    """
    Attention when normalizing: Circle shape might get lost! TODO: correct
    :param data:
    :param indices:
    :param subspace_size:
    :param frac_outlying:
    :return:
    """
    def circle(shape, snr=0.05):
        random_data = np.random.uniform(low=-1, high=1, size=shape)
        c = random_data / np.linalg.norm(random_data, keepdims=True, axis=-1)
        noise = np.random.uniform(low=-1, high=1, size=shape) * snr
        c = c * (1 - 2 * snr)
        c = c * 0.5
        c = c + 0.5
        c = c + noise
        return c
    c = circle(shape=(len(data), len(data[0]), subspace_size))
    data[:, :, :subspace_size] = c
    outliers = np.empty(shape=data.shape, dtype=bool)
    outliers.fill(False)
    for i in indices:
        num_outlying_data = int(len(data[0]) * frac_outlying)
        outlier_indices = np.random.choice(range(len(data[i])), num_outlying_data, replace=False)
        data[i][outlier_indices, :subspace_size] = np.random.uniform(low=0.95, high=1.0, size=subspace_size)
        outliers[i][outlier_indices, :subspace_size] = True
    return data, outliers


def add_local_outliers(data, indices, subspace_size, frac_outlying=0.03):
    outliers = np.empty(shape=data.shape, dtype=bool)
    outliers.fill(False)
    relevant_outliers = outliers[indices]

    global_mean = np.mean(data, axis=(0, 1))

    data_with_outliers = data[indices]
    mean_outlier_data = np.mean(data_with_outliers, axis=1)
    std_outlier_data = np.std(data_with_outliers, axis=1)

    mean_difference = mean_outlier_data - global_mean

    mean_gtz = mean_difference > 0
    sign = np.ones(shape=mean_gtz.shape)
    sign[mean_gtz] = -1

    def to_outlier(p, param, sign):
        subspace = np.random.choice(np.arange(len(p)), subspace_size, replace=False)
        o = np.empty(shape=p.shape, dtype=bool)
        o.fill(False)
        a = np.random.uniform(low=3.0, high=3.3, size=subspace_size) * sign[subspace]
        b = np.random.uniform(low=3.3, high=3.8, size=subspace_size) * sign[subspace]
        ab = np.sort(np.vstack((a, b)).T)
        a, b = ab.T[0], ab.T[1]
        out = truncnorm.rvs(a=a, b=b, loc=param[0][subspace], scale=param[1][subspace], size=p[subspace].shape)
        p[subspace] = out
        o[subspace] = True
        return p, o

    for i, points in enumerate(data_with_outliers):
        param = np.array([mean_outlier_data[i], std_outlier_data[i]])
        sign_point = sign[i]
        num_out = int(len(points)*frac_outlying)
        o_indices = np.random.choice(range(len(points)), num_out, replace=False)

        for j in o_indices:
            point, o = to_outlier(points[j], param, sign_point)
            points[j] = point
            relevant_outliers[i][j] = o
        data_with_outliers[i] = points

    data[indices] = data_with_outliers
    outliers[indices] = relevant_outliers
    return data, outliers


def add_global_outliers(data, indices, subspace_size, frac_outlying=0.03):
    def to_outlier(p, param):
        subspace = np.random.choice(np.arange(len(p)), subspace_size, replace=False)
        o = np.empty(shape=p.shape, dtype=bool)
        o.fill(False)
        sign = np.random.choice([-1, 1], subspace_size, replace=True)
        a = np.random.uniform(low=3.0, high=3.5, size=subspace_size)*sign
        b = np.random.uniform(low=3.5, high=10.0, size=subspace_size)*sign
        ab = np.sort(np.vstack((a, b)).T)
        a, b = ab.T[0], ab.T[1]
        out = truncnorm.rvs(a=a, b=b, loc=param[0][subspace], scale=param[1][subspace], size=p[subspace].shape)
        p[subspace] = out
        o[subspace] = True
        return p, o

    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    mean_param = np.array([mean, std])
    print(mean_param)
    outliers = np.empty(shape=data.shape, dtype=bool)
    outliers.fill(False)

    relevant_data = data[indices]
    relevant_outliers = outliers[indices]
    for i, points in enumerate(relevant_data):
        num_out = int(len(points) * frac_outlying)
        o_indices = np.random.choice(range(len(points)), num_out, replace=False)
        for j in o_indices:
            point, o = to_outlier(points[j], mean_param)
            points[j] = point
            relevant_outliers[i][j] = o
        relevant_data[i] = points
    data[indices] = relevant_data
    outliers[indices] = relevant_outliers
    return data, outliers


def add_abnormal_devices(data, indices, subspace_size, frac_outlying=0.9):
    def to_outlier(p, deviation, subspace):
        p[subspace] = p[subspace] + deviation
        o = np.empty(shape=p.shape, dtype=bool)
        o.fill(False)
        o[subspace] = True
        return p, o

    outliers = np.empty(shape=data.shape, dtype=bool)
    outliers.fill(False)

    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))

    relevant_data = data[indices]
    relevant_outliers = outliers[indices]
    for i, points in enumerate(relevant_data):
        num_out = int(len(points) * frac_outlying)
        subspace = np.random.choice(range(points.shape[-1]), subspace_size, replace=False)
        o_indices = np.random.choice(range(len(points)), num_out, replace=False)
        sign = np.random.choice([-1, 1], size=len(subspace), replace=True)
        outlier_mean = sign * (3 * std[subspace])
        for j in o_indices:
            point, o = to_outlier(points[j], outlier_mean, subspace)
            points[j] = point
            relevant_outliers[i][j] = o
    data[indices] = relevant_data
    outliers[indices] = relevant_outliers
    return data, outliers


def normalize_along_axis(data, axis):
    maxval = data.max(axis=axis, keepdims=True)
    minval = data.min(axis=axis, keepdims=True)
    data = (data-minval)/(maxval-minval)
    return data


def trim_data(data, max_length=10000):
    min_length = min([len(d) for d in data])
    indices = np.random.choice(np.arange(min_length), min(min_length, max_length))
    data = np.array([d[indices] for d in data])
    return data

