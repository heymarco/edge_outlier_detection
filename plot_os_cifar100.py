import os
import numpy as np

from src.cifar10 import create_mnist_data
from src.models import create_deep_models, train_federated

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


x = np.load("original.npy")
y = np.load("labels.npy")
predicted = np.load("predicted.npy")

newshape = (100, 450, 28, 28)

x = x.reshape(newshape)
predicted = predicted.reshape(newshape)

# global scores
diff = np.abs(predicted - x)

diff = diff.reshape((100, 450, 28*28))

dist = np.linalg.norm(diff, axis=-1)
print(dist.shape)
global_scores = dist.flatten()

print(y)

labels = np.arange(100)
accumulated_result = []
for value in labels:
    mean_score = np.mean(global_scores[y.flatten() == value])
    accumulated_result.append(mean_score)

plt.bar(np.arange(len(accumulated_result)), accumulated_result)
plt.show()