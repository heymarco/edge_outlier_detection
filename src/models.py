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


def create_model(dims, compression_factor):
    initializer = tf.keras.initializers.GlorotUniform()

    encoding_dim = int(dims*compression_factor)
    input_img = Input(shape=(dims,))
    encoded = Dense(encoding_dim,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=initializer)(input_img)
    decoded = Dense(dims, activation='sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    activity_regularizer=tf.keras.regularizers.l1(10e-5),
                    kernel_initializer=initializer)(encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


def create_models(num_devices, dims, compression_factor):
    models = []
    for _ in range(num_devices):
        ae = create_model(dims, compression_factor)
        models.append(ae)
    models = np.array(models)

    # SAME WEIGHT INITIALIZATION FOR ALL MODELS
    initial_weights = models[0].get_weights()
    [model.set_weights(initial_weights) for model in models]

    return models


def train_federated(models, data, epochs=1, batch_size=1, frac_available=1.0, verbose=1):
    num_devices = len(models)
    active_devices = np.random.choice(range(num_devices), int(frac_available * num_devices), replace=False)
    for i in active_devices:
        for point in data[i]:
            models[i].fit(np.array([point]), np.array([point]),
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=False,
                          verbose=verbose)
    avg = average_weights(models[active_devices])
    [model.set_weights(avg) for model in models]
    return models


def train_separated(models, data, epochs=1, batch_size=1, frac_available=1.0):
    num_devices = len(models)
    active_devices = np.random.choice(range(num_devices), int(frac_available * num_devices), replace=False)
    for i in active_devices:
        for point in data[i]:
            models[i].fit(np.array([point]), np.array([point]),
                          epochs=epochs,
                          batch_size=1,
                          shuffle=False,
                          verbose=0)
    return models


def train_central(models, data, epochs=1, batch_size=1, frac_available=1.0):
    d = np.reshape(data, newshape=(data.shape[0]*data.shape[1], data.shape[2]))
    models[0].fit(d, d,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=False,
                  verbose=0)
    return models
