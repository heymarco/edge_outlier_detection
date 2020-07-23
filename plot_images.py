import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

x = np.load("original.npy")
y = np.load("labels.npy")
pred = np.load("predicted.npy")
label = np.load("outliers.npy")

oldshape = x.shape
newshape = (oldshape[0]*oldshape[1], 100, 100)

x = x.reshape(newshape)
pred = pred.reshape(newshape)
y = y.flatten()

plt.figure(figsize=(10, 4))
for i in range(1, 10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    y_true = y == i
    show = np.logical_and(y_true, label.astype(bool))
    if (np.any(show)):
        image = pred[show][0]
    plt.imshow(image)
plt.show()
