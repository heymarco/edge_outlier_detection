import os
import tensorflow as tf
import numpy as np


def setup_machine(cuda_device, ram=False):
    """
    If gpus exist, choose the gpu which is specified in 'cuda_device'
    :param cuda_device: The chosen GPU
    :param ram: The allocated ram. If not specified, memory growth is allowed.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    CUDA_VISIBLE_DEVICE = cuda_device
    if gpus:
        try:
            if ram:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[CUDA_VISIBLE_DEVICE],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=ram)])
            else:
                tf.config.experimental.set_memory_growth(gpus[CUDA_VISIBLE_DEVICE], True)
                tf.config.experimental.set_visible_devices(gpus[CUDA_VISIBLE_DEVICE], 'GPU')

        except RuntimeError as e:
            print(e)


def average_weights(models):
    weights = [model.get_weights() for model in models]
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))
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


def normalize_along_axis(data, axis, minimum=0.2, maximum=0.8):
    maxval = data.max(axis=axis, keepdims=True)
    minval = data.min(axis=axis, keepdims=True)
    data = (data - minval) / (maxval - minval)
    data = data * (maximum - minimum) + minimum
    return data
