import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num_outliers = 100
ratio_data_outlier = 50
num_valid = num_outliers*ratio_data_outlier
num_data = num_valid+num_outliers

dims = 3
mean = [1 for _ in range(dims)]
cov = [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]]

def create_outliers(n, mean, covariance):
    dim = len(mean)
    covariance[0][dim-1] = 0.0
    covariance[dim-1][0] = 0.0
    outliers = np.random.multivariate_normal(mean, covariance, size=n)
    return outliers

gaussian = np.random.multivariate_normal(mean, cov, size=num_valid)
gaussian = np.random.uniform(size=(num_valid, dims))
outlier = create_outliers(num_outliers, mean, cov)
print(outlier)
data = np.vstack((gaussian, outlier))
labels = [0 for _ in range(num_valid)] + [1 for _ in range(len(outlier))]
x1, y1, z1 = data.T
x2, y2, z2 = outlier.T

from tensorflow.keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

encoding_dim = 80
input = Input(shape=(num_data,))
print(input.shape)
encoded = Dense(encoding_dim, activation='relu')(input)
print(encoded.shape)
decoded = Dense(num_data, activation='sigmoid')(encoded)
print(decoded.shape)
autoencoder = Model(inputs=input, outputs=decoded)

encoder = Model(inputs=input, outputs=encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(data.T, data.T,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                verbose=0)

decoded = autoencoder.predict(data.T)

import numpy as np
dist = np.zeros(len(data))
for i, x in enumerate(data):
    dist[i] = np.linalg.norm(x-decoded.T[i])
print(dist)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(labels, dist)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC Autoencoder 100-80-100 ReLU/Sigmoid synth\_multidim\_100\_000')
plt.legend(loc="lower right")
plt.show()

x_dec, y_dec, z_dec = decoded
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=x1, ys=y1, zs=z1, color="gray", alpha=0.3)
ax.scatter(xs=x2, ys=y2, zs=z2, color="red", alpha=0.3)
ax.scatter(xs=x_dec, ys=y_dec, zs=z_dec, color="blue")
plt.show()

