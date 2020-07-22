import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


y = np.load("labels.npy")

unique_values, unique_counts = np.unique(y, axis=1, return_counts=True)

result = np.zeros(shape=(100, 10))
labels_and_counts = [np.unique(arr, return_counts=True) for arr in y]

for i in range(len(result)):
    labels, counts = labels_and_counts[i]
    for j in range(len(labels)):
        result[i][labels[j]] = counts[j]

plt.imshow(result.T)
plt.show()