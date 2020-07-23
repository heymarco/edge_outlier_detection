import os
import numpy as np
import tensorflow.keras as keras


def get_data(id, num_clients):
    if id == "mnist":
        return create_mnist_data(num_clients=num_clients)
    if id == "mvtec":
        return create_mvtec_data(num_clients=num_clients)


def create_shards(data, labels, num_clients, shards_per_client):
    total_shards = shards_per_client * num_clients
    shard_size = int(len(labels) / total_shards)
    shard_indices = np.arange(total_shards)
    x_shards = np.array([data[index * shard_size:index * shard_size + shard_size] for index in shard_indices])
    y_shards = np.array([labels[index * shard_size:index * shard_size + shard_size] for index in shard_indices])
    return np.array(x_shards), np.array(y_shards)


def assign_shards(x_shards, y_shards, num_clients, shards_per_user):
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


def shuffle(x_in, y_in):
    for i in np.arange(len(y_in)):
        y_this_part = y_in[i]
        x_this_part = x_in[i]
        indices = np.arange(len(y_this_part))
        np.random.shuffle(indices)
        x_in[i] = x_this_part[indices]
        y_in[i] = y_this_part[indices]
    return x_in, y_in


def add_outlying_partitions(to_x_data, to_y_data,
                            x_outliers, y_outliers,
                            num_outlying_devices):
    outlying_device_labels = np.arange(num_outlying_devices)
    print(len(y_outliers))
    for device_label in outlying_device_labels:
        num_data = to_x_data.shape[1]
        x_outliers = x_outliers[:num_data]
        y_outliers = y_outliers[:num_data]
        to_x_data[device_label] = x_outliers
        to_y_data[device_label] = y_outliers
    return to_x_data, to_y_data


def create_mnist_data(num_clients=100, contamination_local=0.005, contamination_global=0.005, num_outlying_devices=1):
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
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

    def add_global_outliers():
        num_outliers = int(len(y) * contamination_global)
        replaced_inlier_indices = np.random.choice(np.arange(len(y)), num_outliers, replace=False)
        outlier_indices = np.random.choice(np.arange(len(y_out_global)), num_outliers, replace=False)
        x[replaced_inlier_indices] = x_out_global[outlier_indices]
        y[replaced_inlier_indices] = y_out_global[outlier_indices]
        return x, y

    def add_local_outliers():
        num_outliers = int(len(y) * contamination_local)

    add_global_outliers()
    add_local_outliers()

    x_shards, y_shards = create_shards(x, y, num_clients, shards_per_user)
    x_part, y_part = assign_shards(x_shards, y_shards, num_clients, shards_per_user)
    x_final, y_final = add_outlying_partitions(x_part, y_part, x_out_part, y_out_part, num_outlying_devices)
    x_final, y_final = shuffle(x_final, y_final)
    return x_final, y_final


def create_mvtec_data(num_clients=10,
                      contamination_global=0.005, contamination_local=0.005,
                      num_outlying_devices=1, shards_per_client=5):
    x_inlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "inliers.npy"))
    y_inlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "labels_inliers.npy"))
    x_outlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "outliers.npy"))
    y_outlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "labels_outliers.npy"))

    # add outliers to data set
    num_outliers = int(contamination_global*len(y_inlier))
    assert num_outliers < len(y_outlier)
    shuffled_out_indices = np.arange(len(y_outlier))
    np.random.shuffle(shuffled_out_indices)
    x_outlier = x_outlier[shuffled_out_indices]
    y_outlier = y_outlier[shuffled_out_indices]
    outliers = (x_outlier[:num_outliers], y_outlier[:num_outliers])
    x = np.concatenate((x_inlier, outliers[0]))
    y = np.concatenate((y_inlier, outliers[1]))

    _, unique = np.unique(y, return_counts=True)
    print(unique)

    x = np.expand_dims(x, axis=-1)

    # remove global outlier labels values
    masked_labels = np.arange(num_outlying_devices)
    inlier_mask = np.invert(np.isin(y, masked_labels)).flatten()
    x_in_part = x[inlier_mask]
    y_in_part = y[inlier_mask].flatten()
    x_out_part = x[np.invert(inlier_mask)]
    y_out_part = y[np.invert(inlier_mask)].flatten()

    sorted_indices = np.argsort(y_in_part)
    x_in_part = x_in_part[sorted_indices]
    y_in_part = y_in_part[sorted_indices]

    x, y = create_shards(x_in_part, y_in_part, num_clients, shards_per_client)
    x, y = assign_shards(x, y, num_clients, shards_per_client)
    # x, y = add_outlying_partitions(x, y, x_out_part, y_out_part, num_outlying_devices)
    x, y = shuffle(x, y)

    return x, y
