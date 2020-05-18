import numpy as np
from scipy.stats import truncnorm
import math
import random

from src.utils import normalize


def create_data(num_devices, n, dims, gamma, delta):
    def sample_distribution_parameters(gamma, delta):
        mu = np.random.normal(scale=gamma)
        sigma = np.random.uniform(low=1-delta, high=1)
        return mu, sigma
    distribution_params = np.array([
        sample_distribution_parameters(gamma, delta) for _ in range(num_devices)
    ])
    data = np.array([
        np.random.normal(loc=distribution_params[i][0], scale=distribution_params[i][1], size=(n, dims))
        for i in range(num_devices)
    ])
    return data, distribution_params


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


def add_outliers(data, params, indices, subspace_size, frac_outlying=0.03):
    def to_outlier(p, param):
        subspace = np.array(range(subspace_size))
        o = np.empty(shape=p.shape)
        o.fill(False)
        sign = np.random.choice([-1, 1], subspace_size, replace=True)
        a = np.random.uniform(low=2.5, high=3.0, size=subspace_size)*sign
        b = np.random.uniform(low=3.0, high=3.5, size=subspace_size)*sign
        ab = np.sort(np.vstack((a, b)).T)
        a, b = ab.T[0], ab.T[1]
        out = truncnorm.rvs(a=a, b=b, loc=param[0], scale=param[1], size=p[subspace].shape)
        p[subspace] = out
        o[subspace] = True
        return p, o

    outliers = np.empty(shape=data.shape)
    outliers.fill(False)

    relevant_data = data[indices]
    relevant_outliers = outliers[indices]
    for i, points in enumerate(relevant_data):
        num_out = int(len(points)*frac_outlying)
        o_indices = np.random.choice(range(len(points)), num_out, replace=False)
        for j in o_indices:
            point, o = to_outlier(points[j], params[indices[i]])
            points[j] = point
            relevant_outliers[i][j] = o
        relevant_data[i] = points
    data[indices] = relevant_data
    outliers[indices] = relevant_outliers
    return data, outliers



