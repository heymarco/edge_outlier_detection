import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# CREATE MATRICES
n_data = 1000
n_test = 200
n_outliers = 10
dims = 28
matrix = np.random.rand(28, 28)
matrix = np.array([matrix for _ in range(n_data)])
noise = np.random.rand(n_data, 28, 28) * 0.5
matrix = matrix + noise
matrix = np.array([np.maximum(mat, mat.transpose()) for mat in matrix])
for mat in matrix: np.fill_diagonal(mat, 1.0)
matrix = matrix / np.amax(matrix)
for mat in matrix: np.fill_diagonal(mat, 1.0)

def add_outliers(mat, n):
    rows = range(1, len(mat))
    cols = range(1, len(mat))
    outliers = np.copy(mat)
    outliers.fill(0.0)
    outlying_rows = np.random.choice(rows, n)
    outlying_cols = np.random.choice(cols, n)
    for i in range(n):
        row = outlying_rows[i]
        col = outlying_cols[i]
        val = mat[row][col]
        mat[row][col] = 0.0 if val > 0.5 else 1.0
        outliers[row][col] = 1.0
    return mat, outliers

print(matrix[0])
outliers = np.empty(shape=matrix.shape)
for i, mat in enumerate(matrix):
    modified, out = add_outliers(mat, n_outliers)
    matrix[i] = modified
    outliers[i] = out
print(matrix[0])

print(outliers.shape)

matrix = tf.reshape(matrix, shape=(n_data, 28*28, )).numpy()

train = matrix[:len(matrix)-n_test]
test = matrix[len(matrix)-n_test:]

# AUTOENCODER
# this is the size of our encoded representations
encoding_dim = 80  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(train, train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                )

encoded = encoder.predict(test)
decoded = decoder.predict(encoded)

diff = np.abs(test - decoded)

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(test[i].reshape(28, 28), cmap="Blues")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(decoded[i].reshape(28, 28), cmap="Blues")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + 2 * n)
    plt.imshow(outliers[-n_test:][i].reshape(28, 28), cmap="Blues")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(diff[i].reshape(28, 28), cmap="Blues")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()