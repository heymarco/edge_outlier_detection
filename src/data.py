import numpy as np
import tensorflow as tf
import math
import random


def create_blueprint(dims):
    return np.random.normal(size=dims)


def generate_from_blueprint(bp, n, params=(0, 1), snr=0.1):
    data = np.expand_dims(bp, axis=0)
    data = np.repeat(data, n, axis=0)
    for i, d in enumerate(data):
        data[i] = add_gaussian_noise(d, params, snr)
    return data


def sample_distribution_parameters(n):
    return [(np.random.normal(), np.random.uniform(low=0.5)) for _ in range(n)]


def add_gaussian_noise(array_like, params=(0, 1), snr=0.1):
    mu, var = params
    sigma = math.sqrt(var)
    noise = np.random.normal(loc=mu, scale=sigma, size=array_like.shape)*snr
    array_like = array_like + noise
    return array_like


def normalize(array_like):
    min_val = array_like.min()
    max_val = array_like.max()
    return (array_like-min_val)/(max_val-min_val)


def generate_outliers(blueprint, num_outliers):
    o_indices = np.random.choice(range(len(blueprint)), num_outliers)
    outliers = np.empty(len(blueprint))
    outliers.fill(-1)
    for i in o_indices:
        outliers[i] = 1.0 if blueprint[i] > 0.5 else 0.0
    return outliers


def get_observation(params, dims):
    mu, var = params
    sigma = math.sqrt(var)
    x = np.random.normal(loc=mu, scale=sigma, size=dims)
    return x


def symmetric_matrix(m):
    matrix = np.random.rand(m, m)
    matrix = np.maximum(matrix, matrix.transpose())
    np.fill_diagonal(matrix, 1.0)
    return matrix


def add_noise_to_matrix(matrix, snr):
    noise = np.random.uniform(size=matrix.shape) * snr
    matrix = matrix + noise
    np.fill_diagonal(matrix, 1.0)
    matrix = matrix / np.amax(matrix)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def add_outliers_to_matrix(matrix, n):
    rows = range(len(matrix))
    cols = range(   len(matrix))
    assert n < len(matrix)*(len(matrix)-1)/2
    outliers = np.copy(matrix)
    outliers.fill(0.0)
    outlying_rows = []
    outlying_cols = []
    while len(outlying_rows) < n:
        r = np.random.choice(rows)
        c = np.random.choice(cols)
        if r != c:
            outlying_rows.append(r)
            outlying_cols.append(c)
    for i in range(n):
        row = outlying_rows[i]
        col = outlying_cols[i]
        val = matrix[row][col]
        matrix[row][col] = 0.0 if val > 0.5 else 1.0
        matrix[col][row] = 0.0 if val > 0.5 else 1.0
        outliers[row][col] = 1.0
        outliers[col][row] = 1.0
    return matrix, outliers


def average_weights(models):
    weights = [model.get_weights() for model in models]
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
    return new_weights
