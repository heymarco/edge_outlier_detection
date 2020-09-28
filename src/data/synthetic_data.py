import numpy as np
from scipy.stats import zscore, random_correlation


def create_raw_data(num_devices, n, dims):
    return np.random.normal(size=(num_devices, n, dims))


def add_global_outliers(data, subspace_size, frac_outlying=0.03, sigma=2.8):
    num_outliers = int(data.shape[1] * frac_outlying)
    outliers = np.random.normal(size=(data.shape[0], num_outliers, subspace_size))
    mean_norm_in_subspace = np.linalg.norm(np.ones(subspace_size))  # we have data which is normally distributed
    dist = np.random.uniform(sigma, sigma + 1, size=(data.shape[0], num_outliers, subspace_size))
    outliers = outliers / np.linalg.norm(outliers, axis=-1, keepdims=True) * dist * mean_norm_in_subspace
    mask = np.zeros(data.shape)
    for i in range(len(mask)):
        point_indices = np.random.choice(range(data.shape[1]), num_outliers, replace=False)
        for j in point_indices:
            subspace_indices = range(data.shape[-1])
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


def add_deviation(data, sigma):
    shift_direction = np.ones(size=data.shape[-1])
    half_number_of_devices = int(len(data) / 2)

    for i in range(half_number_of_devices):
        data[i] = data[i] + shift_direction * sigma

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
            dist = np.linalg.norm(mean_this_device - mean_other_device)
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


def add_outlying_partitions(data, frac_outlying_data, frac_outlying_devices, subspace_frac, sigma_p):
    num_devices = data.shape[0]
    num_data = data.shape[1]
    dims = data.shape[-1]
    device_indices = np.random.choice(np.arange(num_devices), int(frac_outlying_devices * num_devices), replace=False)
    subspace_size = int(subspace_frac * dims)
    absolute_contamination = int(frac_outlying_data * num_data)
    labels = np.zeros(shape=data.shape).astype(bool)
    for dev in device_indices:
        point_on_circle = np.random.normal(size=subspace_size)
        point_on_circle / np.linalg.norm(point_on_circle)
        shift = point_on_circle * sigma_p
        labels[dev].fill(True)
        subspace = np.random.choice(np.arange(dims), subspace_size, replace=False)
        point_indices = np.random.choice(np.arange(num_data), absolute_contamination, replace=False)
        for p in point_indices:
            for i, s in enumerate(subspace):
                data[dev, p, s] = data[dev, p, s] + shift[i]

    return data, labels


def normalize_along_axis(data, axis, minimum=0.2, maximum=0.8):
    maxval = data.max(axis=axis, keepdims=True)
    minval = data.min(axis=axis, keepdims=True)
    data = (data - minval) / (maxval - minval)
    data = data * (maximum - minimum) + minimum
    return data
