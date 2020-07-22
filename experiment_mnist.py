import os
import numpy as np

from src.utils import setup_machine
from src.cifar10 import create_mnist_data
from src.models import create_models, create_deep_models, train_federated

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

setup_machine(cuda_device=0)

num_devices = 100
global_epochs = 5

x, y = create_mnist_data(num_clients=num_devices)

use_convolutional = True

newshape = (x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])

if not use_convolutional:
    x = x.reshape(newshape)

np.save("original.npy", x)
np.save("labels.npy", y)

if use_convolutional:
    models = create_deep_models(num_devices=num_devices, dims=(28, 28, 1), compression_factor=0.4)
else:
    models = create_models(num_devices=num_devices, dims=28 * 28, compression_factor=0.4)

for epoch in np.arange(global_epochs):
    models = train_federated(models, epochs=1, data=x)

# global scores
predicted = np.array([model.predict(x[i]) for i, model in enumerate(models)])
diff = predicted - x

if use_convolutional:
    diff = diff.reshape(newshape)

dist = np.linalg.norm(diff, axis=-1)
global_scores = dist.flatten()

np.save("predicted.npy", predicted)

labels = np.arange(100)
accumulated_result = []
for value in labels:
    mean_score = np.mean(global_scores[y.flatten() == value])
    accumulated_result.append(mean_score)

plt.bar(np.arange(len(accumulated_result)), accumulated_result)
plt.show()
