import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model


def create_model(dims, compression_factor):
    initializer = tf.keras.initializers.GlorotUniform()
    bias_initializer = tf.keras.initializers.Zeros()

    encoding_dim = int(dims*compression_factor)
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


def create_deep_model(dims=(28, 28, 1)):
    input_img = Input(shape=dims)

    x = Conv2D(8, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=tf.keras.regularizers.l2())(input_img)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (128, 128, 8)

    x = Conv2D(8, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=tf.keras.regularizers.l2())(encoded)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
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


def create_deep_models(num_devices, dims, compression_factor):
    models = []
    for _ in range(num_devices):
        ae = create_deep_model(dims)
        models.append(ae)
    models = np.array(models)

    # SAME WEIGHT INITIALIZATION FOR ALL MODELS
    initial_weights = models[0].get_weights()
    [model.set_weights(initial_weights) for model in models]

    return models
