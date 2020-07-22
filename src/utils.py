import json
import tensorflow as tf
import numpy as np
import seaborn as sns


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


def sliding_window(index, window_size):
    assert index >= window_size, "start index must be >= than window_size"
    return np.arange(index - window_size, index)


def setup_machine(cuda_device, ram=4096):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    CUDA_VISIBLE_DEVICE = cuda_device
    print(gpus)
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[CUDA_VISIBLE_DEVICE], 'GPU')
            # tf.config.experimental.set_virtual_device_configuration(gpus[CUDA_VISIBLE_DEVICE],
            #                                                         [tf.config.experimental.VirtualDeviceConfiguration(
            #                                                             memory_limit=ram)])
        except RuntimeError as e:
            print(e)


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


color_palette = sns.color_palette("cubehelix", 4)


def average_weights(models):
    weights = [model.get_weights() for model in models]
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
    return new_weights