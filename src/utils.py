import json
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_json(filepath):
    with open(filepath) as file:
        return json.load(file)


def normalize(array_like):
    min_val = array_like.min()
    max_val = array_like.max()
    return (array_like-min_val)/(max_val-min_val)


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
    return np.arange(index-window_size, index)


def setup_machine(cuda_device, ram=4096):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    CUDA_VISIBLE_DEVICE = cuda_device
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[CUDA_VISIBLE_DEVICE], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[CUDA_VISIBLE_DEVICE], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=ram)])
        except RuntimeError as e:
            print(e)

