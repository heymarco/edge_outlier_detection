import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt


def augment(image):
    original_shape = image.shape
    image = tf.image.resize_with_crop_or_pad(image, 34, 34) # Add 6 pixels of padding
    image = tf.image.random_crop(image, size=[28, 28, 1]) # Random crop back to 28x28
    image = tf.image.random_brightness(image, max_delta=0.2) # Random brightness
    image = tf.image.random_contrast(image, lower=0.2, upper=0.5)
    image = tf.image.random_flip_left_right(image)
    return image


def get_data(id, num_clients):
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


def create_mnist_data(num_clients=10,
                      contamination_global=0.01, contamination_local=0.005,
                      num_outlying_devices=1, shards_per_client=5):
    # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # add outliers to data set
    num_outliers = int(contamination_global * len(y))
    outlier_indices = np.random.choice(np.arange(len(y)), num_outliers)
    for index in outlier_indices:
        image_indices = np.random.choice(range(len(y)), 2, replace=False)
        image_a = x[image_indices[0]]
        image_b = x[image_indices[1]]
        outlier_image = np.maximum(image_a, image_b)
        print(outlier_image.shape)
        plt.imshow(outlier_image.reshape((28, 28)))
        plt.show()
        x[index] = outlier_image

    labels = np.array([0 for i in range(len(y))], dtype=int)
    labels[outlier_indices] = 1

    shuffled_indices = np.arange(len(y))
    np.random.shuffle(shuffled_indices)

    data = np.array([[x[i], y[i], labels[i]] for i in range(len(y))])
    data = data[shuffled_indices]

    data = create_shards(data, num_clients, shards_per_client)
    data = assign_shards(data, num_clients, shards_per_client)
    np.random.shuffle(data)

    x = np.array(data[:, :, 0])
    shape = list(x.shape) + list(x[0, 0].shape)
    x = np.vstack(x.flatten())
    x = x.reshape(shape).astype(float)
    y = data[:, :, 1].flatten().astype(float)
    labels = data[:, :, 2].flatten().astype(float)

    return x, y, labels


def create_mvtec_data(num_clients=10,
                      contamination_global=0.01, contamination_local=0.005,
                      num_outlying_devices=0, shards_per_client=5):
    x_inlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "inliers.npy"))
    y_inlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "labels_inliers.npy"))
    x_outlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "outliers.npy"))
    y_outlier = np.load(os.path.join(os.getcwd(), "data", "mvtec", "labels_outliers.npy"))

    x_inlier = np.expand_dims(x_inlier, axis=-1)
    x_outlier = np.expand_dims(x_outlier, axis=-1)

    for i in np.arange(len(x_inlier)):
        x_inlier[i] = augment(x_inlier[i])
    for i in np.arange(len(x_outlier)):
        x_outlier[i] = augment(x_outlier[i])

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
