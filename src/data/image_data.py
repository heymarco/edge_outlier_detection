import os
import numpy as np
import tensorflow.keras as keras

import matplotlib.pyplot as plt


def get_image_data(id, num_clients):
    if id == "mnist":
        return create_mnist_data(num_clients=num_clients)
    if id == "mvtec":
        return create_mvtec_data(num_clients=num_clients)


def create_shards(data, num_clients, shards_per_client):
    total_shards = shards_per_client * num_clients
    shard_size = int(len(data) / total_shards)
    shard_indices = np.arange(total_shards)
    data = np.array([data[index * shard_size:index * shard_size + shard_size] for index in shard_indices])
    return data


def assign_shards(x_shards, num_clients, shards_per_user):
    total_num_shards = len(x_shards)
    shard_indices = np.arange(total_num_shards)
    np.random.shuffle(shard_indices)
    part_indices = shard_indices.reshape((num_clients, shards_per_user))
    x_part = np.array([np.concatenate(x_shards[shards]) for shards in part_indices])
    return x_part


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
    for device_label in outlying_device_labels:
        num_data = to_x_data.shape[1]
        x_outliers = x_outliers[:num_data]
        y_outliers = y_outliers[:num_data]
        to_x_data[device_label] = x_outliers
        to_y_data[device_label] = y_outliers
    return to_x_data, to_y_data


def create_mnist_data(num_clients=100,
                      contamination_global=0.01, contamination_local=0.005,
                      num_outlying_devices=10, shards_per_client=5):
    # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x = x / 255.0

    # add outliers to data set
    num_outliers = int(contamination_global * len(y))
    outlier_indices = np.random.choice(np.arange(len(y)), num_outliers)
    for _ in outlier_indices:
        image_indices = np.random.choice(range(len(y)), 2, replace=False)
        image_a = x[image_indices[0]]
        image_b = x[image_indices[1]]
        outlier_image = np.maximum(image_a, image_b)

    labels = np.array([0 for i in range(len(y))], dtype=int)
    labels[outlier_indices] = 2

    shuffled_indices = np.arange(len(y))
    np.random.shuffle(shuffled_indices)

    data = np.array([[x[i], y[i], labels[i]] for i in range(len(y))])
    data = data[shuffled_indices]

    data = create_shards(data, num_clients, shards_per_client)
    data = assign_shards(data, num_clients, shards_per_client)
    np.random.shuffle(data)

    for i, client_data in enumerate(data):
        for j, d in enumerate(client_data):
            if i < num_outlying_devices:
                if np.random.uniform() < contamination_local:
                    data[i, j, 0] = 1.0 - d[0] # invert color
                    data[i, j, 2] = 1
            else:
                if np.random.uniform() < 0.5:
                    data[i, j, 0] = 1.0 - d[0]  # invert color
                    data[i, j, 2] = 1
                plt.imshow(data[i, j, 0].reshape((28, 28)))
                plt.show()

    x = np.array(data[:, :, 0])
    shape = list(x.shape) + list(x[0, 0].shape)
    x = np.vstack(x.flatten())
    x = x.reshape(shape).astype(float)
    y = data[:, :, 1].flatten().astype(float)
    labels = data[:, :, 2].flatten().astype(float)

    return x, y, labels


def create_mvtec_data(num_clients=10,
                      contamination_global=0.01, contamination_local=0.005,
                      num_outlying_devices=1, shards_per_client=5):
    x_inlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "inliers.npy"))
    y_inlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "labels_inliers.npy"))
    x_outlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "outliers.npy"))
    y_outlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "labels_outliers.npy"))

    x_inlier = np.expand_dims(x_inlier, axis=-1)
    x_outlier = np.expand_dims(x_outlier, axis=-1)

    inlier = np.array([[x_inlier[i], y, 0] for i, y in enumerate(y_inlier)])
    outlier = np.array([[x_outlier[i], y, 1] for i, y in enumerate(y_outlier)])

    # add outliers to data set
    num_outliers = int(contamination_global*len(y_inlier))
    assert num_outliers < len(outlier)
    shuffled_out_indices = np.arange(len(outlier))
    np.random.shuffle(shuffled_out_indices)
    outlier = outlier[shuffled_out_indices][:num_outliers]

    data = np.concatenate((inlier, outlier))

    # remove global outlier labels values
    masked_labels = np.arange(num_outlying_devices)
    inlier_mask = np.invert(np.isin(data[:, 1], masked_labels)).flatten()
    in_partition = data[inlier_mask]
    out_partition = data[np.invert(inlier_mask)]

    sorted_indices = np.argsort(in_partition[:, 1])
    in_partition = in_partition[sorted_indices]

    data = create_shards(in_partition, num_clients, shards_per_client)
    data = assign_shards(data, num_clients, shards_per_client)
    # x, y = add_outlying_partitions(x, y, x_out_part, y_out_part, num_outlying_devices)
    np.random.shuffle(data)

    x = np.array(data[:, :, 0])
    shape = list(x.shape) + list(x[0, 0].shape)
    x = np.vstack(x.flatten())
    x = x.reshape(shape).astype(float)
    y = data[:, :, 1].flatten().astype(float)
    labels = data[:, :, 2].flatten().astype(float)

    return x, y, labels


def plot_outlier(images, labels, index):
    out_images = images[labels == 0]
    plt.imshow(out_images[index])
    plt.show()
