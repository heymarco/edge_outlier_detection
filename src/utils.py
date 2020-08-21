import json
import os

import tensorflow as tf
import numpy as np
import seaborn as sns

import numpy as np


color_palette = sns.color_palette("cubehelix", 4)


def load_json(filepath):
    with open(filepath) as file:

        return json.load(file)


def normalize(array_like):
    min_val = array_like.min()
    max_val = array_like.max()
    return (array_like - min_val) / (max_val - min_val)


def setup_machine(cuda_device, ram=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    CUDA_VISIBLE_DEVICE = cuda_device
    if gpus:
        try:
            if ram:
                tf.config.experimental.set_virtual_device_configuration(gpus[CUDA_VISIBLE_DEVICE],
                                                                        [tf.config.experimental.VirtualDeviceConfiguration(
                                                                            memory_limit=ram)])
            else:
                tf.config.experimental.set_visible_devices(gpus[CUDA_VISIBLE_DEVICE], 'GPU')

        except RuntimeError as e:
            print(e)


def average_weights(models):
    weights = [model.get_weights() for model in models]
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
    return new_weights


def load_all_in_dir(directory):
    all_files = {}
    for file in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, file)):
            continue
        if file.endswith(".npy"):
            filepath = os.path.join(directory, file)
            result_file = np.load(filepath)
            all_files[file] = result_file
    return all_files


def parse_filename(file):
    keys = [
        "num_devices",
        "num_data",
        "dims",
        "subspace_frac",
        "frac_outlying_devices",
        "frac_local",
        "frac_global",
        "sigma_l",
        "sigma_g",
        "data_type",
        "c_name",
        "l_name"
    ]
    components = file.split("_")
    assert len(components) == len(keys)
    parsed_args = {}
    for i in range(len(keys)):
        parsed_args[keys[i]] = components[i]
    return parsed_args