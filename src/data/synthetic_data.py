import numpy as np
from scipy.stats import zscore, random_correlation
import matplotlib.pyplot as plt


def create_raw_data(num_devices, n, dims):
    return np.random.normal(size=(num_devices, n, dims))


def add_global_outliers(data, subspace_size, frac_outlying=0.03):
    num_outliers = int(data.shape[1]*frac_outlying)
    # def to_outlier(p, param):
    #     subspace = np.random.choice(np.arange(len(p)), subspace_size, replace=False)
    #     o = np.empty(shape=p.shape, dtype=bool)
    #     o.fill(False)
    #     sign = np.random.choice([-1, 1], subspace_size, replace=True)
    #     a = 2.5 * sign
    #     b = 3 * sign
    #     ab = np.sort(np.vstack((a, b)).T)
    #     a, b = ab.T[0], ab.T[1]
    #     mean = param[0][subspace]
    #     std = param[1][subspace]
    #     out = mean + np.random.uniform(a, b, size=p[subspace].shape) * std
    #     # out = truncnorm.rvs(a=a, b=b, loc=param[0][subspace], scale=param[1][subspace], size=p[subspace].shape)
    #     p[subspace] = out
    #     o[subspace] = True
    #     return p, o
    #
    # mean = np.mean(data, axis=(0, 1))
    # std = np.std(data, axis=(0, 1))
    # mean_param = np.array([mean, std])
    # outliers = np.empty(shape=data.shape, dtype=bool)
    # outliers.fill(False)
    #
    # relevant_data = data
    # relevant_outliers = outliers
    # for i, points in enumerate(relevant_data):
    #     num_out = int(len(points) * frac_outlying)
    #     o_indices = np.random.choice(range(len(points)), num_out, replace=False)
    #     for j in o_indices:
    #         point, o = to_outlier(points[j], mean_param)
    #         points[j] = point
    #         relevant_outliers[i][j] = o
    #     relevant_data[i] = points
    # data = relevant_data
    # outliers = relevant_outliers
    std = np.std(data)
    outliers = np.random.normal(size=(data.shape[0], num_outliers, subspace_size))
    dist = np.random.uniform(5, 10, size=(data.shape[0], num_outliers, subspace_size))
    outliers = outliers / np.linalg.norm(outliers, axis=-1, keepdims=True) * dist * std
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
    # plt.scatter(data[0].T[0], data[0].T[1])
    # plt.show()
    return data


def add_deviation(data, gamma, delta):
    old_data = data
    def create_deviation(size, dev):
        std = np.std(old_data.reshape(old_data.shape[0]*old_data.shape[1], old_data.shape[-1]), axis=0)
        deviation = np.random.uniform(low=-1, high=1, size=size)
        deviation = zscore(deviation)
        return deviation * dev * std

    for i in range(len(data)):
        deviation = create_deviation(data.shape[-1], gamma)
        data[i] = data[i] + deviation

    return data


def create_data(num_devices, n, dims, gamma, delta):
    def create_blueprint(size):
        bp = np.random.uniform(low=-1, high=1, size=size)
        return zscore(bp)

    def create_deviation(size, dev):
        return zscore(np.random.uniform(low=-1, high=1, size=size)) * dev

    def create_gaussian_noise(size, snr=0.2):
        return np.random.normal(size=size) * snr

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


def add_local_outliers(data, subspace_size, frac_outlying=0.03):
    outliers = np.empty(shape=data.shape, dtype=bool)
    outliers.fill(False)

    global_mean = np.mean(data, axis=(0, 1))

    mean_outlier_data = np.mean(data, axis=1)
    std_outlier_data = np.std(data, axis=1)

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
            # point, o = to_outlier(points[j], param, sign_point)
            points[j] = point
            data[i][j] = point
            outliers[i][j] = o

    return data, outliers


# def add_global_outliers(data, indices, subspace_size, frac_outlying=0.03):
#     def to_outlier(p, param):
#         subspace = np.random.choice(np.arange(len(p)), subspace_size, replace=False)
#         o = np.empty(shape=p.shape, dtype=bool)
#         o.fill(False)
#         sign = np.random.choice([-1, 1], subspace_size, replace=True)
#         a = 2.5 * sign
#         b = 3 * sign
#         ab = np.sort(np.vstack((a, b)).T)
#         a, b = ab.T[0], ab.T[1]
#         mean = param[0][subspace]
#         std = param[1][subspace]
#         out = mean + np.random.uniform(a, b, size=p[subspace].shape) * std
#         # out = truncnorm.rvs(a=a, b=b, loc=param[0][subspace], scale=param[1][subspace], size=p[subspace].shape)
#         p[subspace] = out
#         o[subspace] = True
#         return p, o
#
#     mean = np.mean(data, axis=(0, 1))
#     std = np.std(data, axis=(0, 1))
#     mean_param = np.array([mean, std])
#     outliers = np.empty(shape=data.shape, dtype=bool)
#     outliers.fill(False)
#
#     relevant_data = data[indices]
#     relevant_outliers = outliers[indices]
#     for i, points in enumerate(relevant_data):
#         num_out = int(len(points) * frac_outlying)
#         o_indices = np.random.choice(range(len(points)), num_out, replace=False)
#         for j in o_indices:
#             point, o = to_outlier(points[j], mean_param)
#             points[j] = point
#             relevant_outliers[i][j] = o
#         relevant_data[i] = points
#     data[indices] = relevant_data
#     outliers[indices] = relevant_outliers
#     return data, outliers


def add_abnormal_devices(data, indices, subspace_size, frac_outlying=0.9, mean_deviation=1.5):
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


def z_score_normalization_along_axis(data, axis):
    return zscore(data, axis=axis)


def trim_data(data, max_length=10000):
    min_length = min([len(d) for d in data])
    max_length = min(min_length, max_length)
    data = np.array([d[:max_length] for d in data])
    return data
