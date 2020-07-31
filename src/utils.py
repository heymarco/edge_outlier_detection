import json
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


def parse_filename(name):
    raw = name.split("_")
    params = {
        "num_devices": int(raw[0]),
        "num_data": int(raw[1]),
        "dims": int(raw[2]),
        "subspace_frac": float(raw[3]),
        "frac_outlying_devices": float(raw[4]),
        "frac_outlying_data": float(raw[5]),
        "gamma": float(raw[6]),
        "delta": float(raw[7]),
        "outlier_type": raw[8]
    }
    return params


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
