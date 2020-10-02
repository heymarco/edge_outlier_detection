import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model


def create_model(dims, compression_factor):
    """
    Create an autoencoder with a single hidden layer
    :param dims: The number of input dimensions
    :param compression_factor: The compression factor
    :return: The autoencoder model
    """
    initializer = tf.keras.initializers.GlorotUniform()
    bias_initializer = tf.keras.initializers.Zeros()

    encoding_dim = int(dims * compression_factor)
    input_img = Input(shape=(dims,))
    encoded = Dense(encoding_dim,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=initializer,
                    bias_initializer=bias_initializer)(input_img)
    decoded = Dense(dims, activation='sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=initializer,
                    bias_initializer=bias_initializer)(encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
    return autoencoder


def create_models(num_devices, dims, compression_factor):
    """
    Create one model per network participant
    :param num_devices: The number of models to create
    :param dims: The number of input dimensions
    :param compression_factor: The compression factor of the autoencoder
    :return: A list of identical models
    """
    models = []
    for _ in range(num_devices):
        ae = create_model(dims, compression_factor)
        models.append(ae)
    models = np.array(models)

    # SAME WEIGHT INITIALIZATION FOR ALL MODELS
    initial_weights = models[0].get_weights()
    [model.set_weights(initial_weights) for model in models]

    return models
