import numpy as np
from scipy.stats import zscore, random_correlation


def create_raw_data(num_devices, n, dims):
    mat = np.random.normal(size=(num_devices, n, dims))
    return mat


def add_global_outliers(data, subspace_size, frac_outlying=0.03, sigma=5):
    num_outliers = int(data.shape[1]*frac_outlying)
    outliers = np.random.normal(size=(data.shape[0], num_outliers, subspace_size))
    mean_norm = np.linalg.norm(np.ones(data.shape[-1]))  # we have data which is normally distributed
    dist = np.random.uniform(sigma, sigma+1, size=(data.shape[0], num_outliers, subspace_size))
    outliers = outliers / np.linalg.norm(outliers, axis=-1, keepdims=True) * dist * mean_norm
    mask = np.zeros(data.shape)
    for i in range(len(mask)):
        point_indices = np.random.choice(range(data.shape[1]), num_outliers, replace=False)
        for j in point_indices:
            subspace_indices = range(mask.shape[-1])
            subspace = np.random.choice(subspace_indices, subspace_size, replace=False)
            mask[i][j][subspace] = 1
    np.putmask(data, mask, outliers)
    return data, mask.astype(bool)


def add_random_correlation(data):
    dims = data.shape[-1]
    evs = np.random.uniform(0.01, 1, size=dims)
    evs = evs / np.sum(evs) * dims
    random_corr_matrix = random_correlation.rvs(evs)
    cholesky_transform = np.linalg.cholesky(random_corr_matrix)
    for i in range(data.shape[0]):
        normal_eq_mean = cholesky_transform.dot(data[i].T)  # Generating random MVN (0, cov_matrix)
        normal_eq_mean = normal_eq_mean.transpose()
        normal_eq_mean = normal_eq_mean.transpose()  # Transposing back
        data[i] = normal_eq_mean.T
    return data


# def add_deviation(data, sigma):
#
#     def create_deviation(size, dev):
#         deviation = zscore(np.random.uniform(low=-1, high=1, size=size))
#         sign = np.random.choice([-1, 1], size=size)
#         deviation = deviation * sign
#         return deviation * dev
#
#     std = np.std(data, axis=1)
#     for i in range(len(data)):
#         deviation = create_deviation(data.shape[-1], sigma)
#         data[i] = data[i] + deviation * std[i]
#
#     return data


def add_deviation(data, sigma):

    shift_direction = np.random.normal(size=data.shape[-1])
    shift_direction = shift_direction / np.linalg.norm(shift_direction)  # hypersphere

    half_number_of_devices = int(len(data) / 2)

    for i in range(half_number_of_devices):
        data[i] = data[i] + shift_direction*sigma

    return data


def add_local_outliers(data, subspace_size, frac_outlying=0.03):
    outliers = np.empty(shape=data.shape, dtype=bool)
    outliers.fill(False)

    global_mean = np.mean(data, axis=(0, 1))

    mean_outlier_data = np.mean(data, axis=1)
    mean_difference = mean_outlier_data - global_mean

    mean_gtz = mean_difference > 0
    sign = np.ones(shape=mean_gtz.shape)
    sign[mean_gtz] = -1

    def to_outlier(p, device_index):
        o = np.empty(shape=p.shape, dtype=bool)
        o.fill(False)
        subspace = np.random.choice(np.arange(len(p)), subspace_size, replace=False)
        distances = []
        for device in np.arange(data.shape[0]):
            mean_this_device = np.mean(data[device_index][subspace], axis=-2)
            mean_other_device = np.mean(data[device][subspace], axis=-2)
            dist = np.linalg.norm(mean_this_device-mean_other_device)
            distances.append(dist)
        other_device = np.argmax(distances)
        random_point_index = np.random.choice(np.arange(data.shape[1]))
        random_point = data[other_device][random_point_index]
        p[subspace] = random_point[subspace]
        o[subspace] = True
        return p, o

    for i, points in enumerate(data):
        num_out = int(len(points) * frac_outlying)
        o_indices = np.random.choice(range(len(points)), num_out, replace=False)

        for j in o_indices:
            point, o = to_outlier(points[j], i)
            points[j] = point
            data[i][j] = point
            outliers[i][j] = o

    return data, outliers


def add_abnormal_devices(data, indices, subspace_size, frac_outlying=0.9, mean_deviation=1.5):
    def to_outlier(p, deviation, subspace):
        p[subspace] = p[subspace] + deviation
        o = np.empty(shape=p.shape, dtype=bool)
        o.fill(False)
        o[subspace] = True
        return p, o

    outliers = np.empty(shape=data.shape, dtype=bool)
    outliers.fill(False)

    std = np.mean(np.std(data, axis=(0, 1)))

    relevant_data = data[indices]
    relevant_outliers = outliers[indices]
    for i, points in enumerate(relevant_data):
        num_out = int(len(points) * frac_outlying)
        subspace = np.random.choice(range(points.shape[-1]), subspace_size, replace=False)
        o_indices = np.random.choice(range(len(points)), num_out, replace=False)
        sign = np.random.choice([-1, 1], size=len(subspace), replace=True)
        outlier_mean = sign * (mean_deviation * std[subspace])
        for j in o_indices:
            point, o = to_outlier(points[j], outlier_mean, subspace)
            points[j] = point
            relevant_outliers[i][j] = o
    data[indices] = relevant_data
    outliers[indices] = relevant_outliers
    return data, outliers


def normalize_along_axis(data, axis, minimum=0.2, maximum=0.8):
    maxval = data.max(axis=axis, keepdims=True)
    minval = data.min(axis=axis, keepdims=True)
    data = (data - minval) / (maxval - minval)
    data = data * (maximum - minimum) + minimum
    return data


def trim_data(data, max_length=10000):
    min_length = min([len(d) for d in data])
    max_length = min(min_length, max_length)
    data = np.array([d[:max_length] for d in data])
    return data
