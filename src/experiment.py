import os
import math
import numpy as np
from sklearn.metrics import roc_curve, auc
from src.data import create_blueprint, sample_distribution_parameters, add_gaussian_noise, normalize, generate_outliers
from src.data import average_weights
from src.data import generate_from_blueprint

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model


def train_federated(models, data, epochs, batch_size, frac_available=1.0):
    num_devices = len(models)
    active_devices = np.random.choice(range(num_devices), int(frac_available * num_devices))
    active_models = models[active_devices]
    relevant_data = data[active_devices]
    print("Train on {} devices".format(len(active_devices)))
    for i in range(len(active_models)):
        active_models[i].fit(relevant_data[i], relevant_data[i],
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=False)
    avg = average_weights(active_models)
    [model.set_weights(avg) for model in models]
    return models
