import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def create_mnist_data(num_clients=100, contamination_local=0.005, contamination_global=0.005, num_outlying_devices=1):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = np.expand_dims(x_train, axis=-1)

    #remove global outlier labels values
    masked_labels = [0]
    for device_index in np.arange(1, num_outlying_devices+1):
        masked_labels.append(device_index)
    inlier_mask = np.invert(np.isin(y_train, masked_labels)).flatten()
    x = x_train[inlier_mask]
    y = y_train[inlier_mask].flatten()

    x_out = x_train[np.invert(inlier_mask)]
    y_out = y_train[np.invert(inlier_mask)].flatten()
    x_out_global = x_out[y_out==0]
    y_out_global = y_out[y_out==0]
    x_out_part = x_out[y_out!=0]
    y_out_part = y_out[y_out!=0]

    sorted_indices = np.argsort(y)
    x = x[sorted_indices]
    y = y[sorted_indices]

    shards_per_user = 50

    def create_shards(shards_per_user):
        total_shards = shards_per_user*num_clients
        shard_size = int(len(y)/total_shards)
        shard_indices = np.arange(total_shards)
        x_shards = np.array([x[index*shard_size:index*shard_size+shard_size] for index in shard_indices])
        y_shards = np.array([y[index*shard_size:index*shard_size+shard_size] for index in shard_indices])
        return np.array(x_shards), np.array(y_shards)

    def add_global_outliers():
        num_outliers = int(len(y)*contamination_global)
        replaced_inlier_indices = np.random.choice(np.arange(len(y)), num_outliers, replace=False)
        outlier_indices = np.random.choice(np.arange(len(y_out_global)), num_outliers, replace=False)
        x[replaced_inlier_indices] = x_out_global[outlier_indices]
        y[replaced_inlier_indices] = y_out_global[outlier_indices]
        return x, y

    def add_local_outliers():
        lo_label = "1"
        num_outliers = int(len(y)*contamination_local)

    def assign_shards(x_shards, y_shards, shards_per_user):
        total_num_shards = len(y_shards)
        shard_indices = np.arange(total_num_shards)
        np.random.shuffle(shard_indices)
        part_indices = shard_indices.reshape((num_clients, shards_per_user))
        x_part = np.array([x_shards[shards] for shards in part_indices])
        y_part = np.array([y_shards[shards] for shards in part_indices])
        x_oldshape = x_part.shape
        y_oldshape = y_part.shape
        x_newshape = (x_oldshape[0], x_oldshape[1]*x_oldshape[2], x_oldshape[3], x_oldshape[4], x_oldshape[5])
        y_newshape = (y_oldshape[0], y_oldshape[1]*y_oldshape[2])
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
