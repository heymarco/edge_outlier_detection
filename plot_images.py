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

oldshape = pred.shape
newshape = (oldshape[0]*oldshape[1], 128, 128)

x = x.reshape(newshape)
pred = pred.reshape(newshape)
y = y.flatten()

plt.figure(figsize=(10, 4))
i = 0
num_pics = 0
max_num_pics = 12
while num_pics < max_num_pics:
    y_true = y == i
    show = np.logical_and(y_true, label.astype(bool))
    if (np.any(show)):
        plt.subplot(3, max_num_pics, num_pics + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = x[show][0]
        plt.imshow(image)
        plt.subplot(3, max_num_pics, num_pics + max_num_pics + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = pred[show][0]
        plt.imshow(image)
        plt.subplot(3, max_num_pics, num_pics + 2*max_num_pics + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = np.abs(x[show][0] - pred[show][0])
        plt.imshow(image)
        num_pics += 1
    i += 1

plt.show()
