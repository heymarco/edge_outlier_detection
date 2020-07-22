import os
import numpy as np

from src.cifar10 import create_cifar10_data
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

# global scores
diff = predicted - x

newshape = (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]*x.shape[4])
diff = diff.reshape(newshape)

print(diff)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    i = -i if i > 12 else i
    plt.imshow(np.abs(diff).reshape((x.shape[0]*x.shape[1], 32, 32, 3))[i], cmap=plt.cm.binary)
plt.show()


dist = np.linalg.norm(diff, axis=-1)
global_scores = dist.flatten()

labels = np.arange(100)
accumulated_result = []
for value in labels:
    mean_score = np.mean(global_scores[y.flatten() == value])
    accumulated_result.append(mean_score)

plt.bar(np.arange(len(accumulated_result)), accumulated_result)
plt.show()