import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from src.data import *

# PARAMS
N = 100
n = 10
m = 28
div = 0.3
snr = 0.1
num_outliers = 6

# CREATE CORE MATRIX
core_matrix = symmetric_matrix(m)

# CREATE N COPIES
# (one for each device)
copies = [np.copy(core_matrix) for _ in range(N)]

# WEAKEN SIMILARITY BETWEEN COPIES
copies = [add_noise_to_matrix(mat, snr=div) for mat in copies]
copies = np.array(copies)

# ADD OUTLIERS
outliers = np.empty(shape=copies.shape)
for i in range(len(copies)):
    mat, out = add_outliers_to_matrix(copies[i], num_outliers)
    copies[i] = mat
    outliers[i] = out

# CREATE MULTIPLE COPIES OF EACH DEVICE MATRIX
copies = np.expand_dims(copies, axis=1)
copies = np.repeat(copies, axis=1, repeats=n)

# ADD NOISE TO EACH MATRIX
copies = np.array([[add_noise_to_matrix(mat, snr=snr) for mat in device_data] for device_data in copies])

# CREATE N FEDERATED MODELS
models = []
encoding_dim = 34
for i in range(N):
    input_img = Input(shape=(m*m,))
    encoded = Dense(encoding_dim,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2())(input_img)
    decoded = Dense(m*m, activation='sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    activity_regularizer=tf.keras.regularizers.l1(10e-5))(encoded)
    autoencoder = Model(input_img, decoded)
    models.append(autoencoder)

# SAME WEIGHT INITIALIZATION FOR ALL MODELS
initial_weights = models[0].get_weights()
[model.set_weights(initial_weights) for model in models]
[model.compile(optimizer='adam', loss='binary_crossentropy') for model in models]

# TRAIN NON-FEDERATED
# - sample 0.1*N clients
# - train for l local epochs with batch size bs
# - average results --> not done here
# distribute model to all clients

client_ids = range(N)
global_epochs = 5
local_epochs = 4
flattened = tf.reshape(copies, shape=(N, n, m*m, )).numpy()
for ge in range(global_epochs):
    active_clients = client_ids
    for i in active_clients:
        model = models[i]
        client_data = flattened[i]
        train = client_data[:-1]
        model.fit(train, train,
                        epochs=local_epochs,
                        batch_size=1,
                        shuffle=True)

# PREDICT FEDERATED
predictions = []
for i, model in enumerate(models):
    predicted = model.predict(flattened[i][-1:])
    predictions.append(predicted)


n = 10  # how many digits we will display
fig = plt.figure(figsize=(20, 5))
for i in range(n):
    test = copies[i][-1].reshape(m, m)
    out = outliers[i].reshape(m, m)
    prediction = predictions[i][0].reshape(m, m)

    # display original
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(test, cmap="Blues")
    ax.get_xaxis().set_visible(False)
    ax.set_title("Device {}".format(i))
    if i == 0:
        ax.set_ylabel("original")
    else:
        ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(prediction, cmap="Blues")
    ax.get_xaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("generated")
    else:
        ax.get_yaxis().set_visible(False)

        # display reconstruction
    diff = np.abs(test - prediction)
    ax = plt.subplot(4, n, i + 1 + 2 * n)
    plt.imshow(diff, cmap="Blues")
    ax.get_xaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("difference")
    else:
        ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + 3 * n)
    plt.imshow(out, cmap="Blues")
    ax.get_xaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("outliers")
    else:
        ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()





