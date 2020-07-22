import numpy as np
from scipy.stats import truncnorm, zscore
import tensorflow.keras as keras


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
        noise = (1.0 - alpha) + alpha * np.random.uniform(size=len(subspace))
        noise = np.expand_dims(noise, axis=0)
        noise = np.repeat(noise, len(timeseries), axis=0)
        relevant_data[i][:, subspace] = timeseries[:, subspace] + noise * max_shift
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

    def to_outlier(p, device_index):
        o = np.empty(shape=p.shape, dtype=bool)
        o.fill(False)
        other_device = device_index
        while other_device == device_index:
            other_device = np.random.choice(np.arange(data.shape[0]))
            print(other_device)
        random_point_index = np.random.choice(np.arange(data.shape[1]))
        random_point = data[other_device][random_point_index]
        subspace = np.random.choice(np.arange(len(p)), subspace_size, replace=False)
        p[subspace] = random_point[subspace]
        o[subspace] = True
        return p, o

    # def to_outlier(p, param, sign):
    #     subspace = np.random.choice(np.arange(len(p)), subspace_size, replace=False)
    #     o = np.empty(shape=p.shape, dtype=bool)
    #     o.fill(False)
    #     a = np.sqrt(np.random.uniform(3.0, 3.1, size=sign[subspace].shape))*sign[subspace]
    #     b = np.sqrt(np.random.uniform(3.2, 3.3, size=sign[subspace].shape))*sign[subspace]
    #     ab = np.sort(np.vstack((a, b)).T)
    #     a, b = ab.T[0], ab.T[1]
    #     print(param[1][subspace])
    #     out = truncnorm.rvs(a=a, b=b, loc=param[0][subspace], scale=param[1][subspace], size=p[subspace].shape)
    #     p[subspace] = out
    #     o[subspace] = True
    #     return p, o

    for i, points in enumerate(data_with_outliers):
        param = np.array([mean_outlier_data[i], std_outlier_data[i]])
        sign_point = sign[i]
        num_out = int(len(points) * frac_outlying)
        o_indices = np.random.choice(range(len(points)), num_out, replace=False)

        for j in o_indices:
            point, o = to_outlier(points[j], indices[i])
            # point, o = to_outlier(points[j], param, sign_point)
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
        a = 3 * sign
        b = 3.5 * sign
        ab = np.sort(np.vstack((a, b)).T)
        a, b = ab.T[0], ab.T[1]
        mean = param[0][subspace]
        std = param[1][subspace]
        out = mean + np.random.uniform(a, b, size=p[subspace].shape) * std
        # out = truncnorm.rvs(a=a, b=b, loc=param[0][subspace], scale=param[1][subspace], size=p[subspace].shape)
        p[subspace] = out
        o[subspace] = True
        return p, o

    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    mean_param = np.array([mean, std])
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


def create_mnist_data(num_clients=100, contamination_local=0.005, contamination_global=0.005, num_outlying_devices=1):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = np.expand_dims(x_train, axis=-1)

    # remove global outlier labels values
    masked_labels = [0]
    for device_index in np.arange(1, num_outlying_devices + 1):
        masked_labels.append(device_index)
    inlier_mask = np.invert(np.isin(y_train, masked_labels)).flatten()
    x = x_train[inlier_mask]
    y = y_train[inlier_mask].flatten()

    x_out = x_train[np.invert(inlier_mask)]
    y_out = y_train[np.invert(inlier_mask)].flatten()
    global_outlier = 1
    x_out_global = x_out[y_out == global_outlier]
    y_out_global = y_out[y_out == global_outlier]
    x_out_part = x_out[y_out != global_outlier]
    y_out_part = y_out[y_out != global_outlier]

    sorted_indices = np.argsort(y)
    x = x[sorted_indices]
    y = y[sorted_indices]

    shards_per_user = 5

    def create_shards(shards_per_user):
        total_shards = shards_per_user * num_clients
        shard_size = int(len(y) / total_shards)
        shard_indices = np.arange(total_shards)
        x_shards = np.array([x[index * shard_size:index * shard_size + shard_size] for index in shard_indices])
        y_shards = np.array([y[index * shard_size:index * shard_size + shard_size] for index in shard_indices])
        return np.array(x_shards), np.array(y_shards)

    def add_global_outliers():
        num_outliers = int(len(y) * contamination_global)
        replaced_inlier_indices = np.random.choice(np.arange(len(y)), num_outliers, replace=False)
        outlier_indices = np.random.choice(np.arange(len(y_out_global)), num_outliers, replace=False)
        x[replaced_inlier_indices] = x_out_global[outlier_indices]
        y[replaced_inlier_indices] = y_out_global[outlier_indices]
        return x, y

    def add_local_outliers():
        num_outliers = int(len(y) * contamination_local)

    def assign_shards(x_shards, y_shards, shards_per_user):
        total_num_shards = len(y_shards)
        shard_indices = np.arange(total_num_shards)
        np.random.shuffle(shard_indices)
        part_indices = shard_indices.reshape((num_clients, shards_per_user))
        x_part = np.array([x_shards[shards] for shards in part_indices])
        y_part = np.array([y_shards[shards] for shards in part_indices])
        x_oldshape = x_part.shape
        y_oldshape = y_part.shape
        x_newshape = (x_oldshape[0], x_oldshape[1] * x_oldshape[2], x_oldshape[3], x_oldshape[4], x_oldshape[5])
        y_newshape = (y_oldshape[0], y_oldshape[1] * y_oldshape[2])
        x_part = x_part.reshape(x_newshape)
        y_part = y_part.reshape(y_newshape)
        return x_part, y_part

    def add_outlying_partitions(to_x_data, to_y_data):
        outlying_device_labels = np.arange(num_outlying_devices)
        for device_label in outlying_device_labels:
            num_data = to_x_data.shape[1]
            x_outliers = x_out_part[:num_data]
            y_outliers = y_out_part[:num_data]
            to_x_data[device_label] = x_outliers
            to_y_data[device_label] = y_outliers
        return to_x_data, to_y_data

    def shuffle(x_in, y_in):
        for i in np.arange(len(y_in)):
            y_this_part = y_in[i]
            x_this_part = x_in[i]
            indices = np.arange(len(y_this_part))
            np.random.shuffle(indices)
            x_in[i] = x_this_part[indices]
            y_in[i] = y_this_part[indices]
        return x_in, y_in

    add_global_outliers()
    add_local_outliers()

    x_shards, y_shards = create_shards(shards_per_user)
    x_part, y_part = assign_shards(x_shards, y_shards, shards_per_user)
    x_final, y_final = add_outlying_partitions(x_part, y_part)
    x_final, y_final = shuffle(x_final, y_final)
    return x_final, y_final
