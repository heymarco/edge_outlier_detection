import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')
import seaborn as sns

x = np.load("original.npy")
y = np.load("labels.npy")
pred = np.load("predicted.npy")
label = np.load("outliers.npy")

oldshape = x.shape
newshape = (oldshape[0]*oldshape[1], 100, 100)

x = x.reshape(newshape)
pred = pred.reshape(newshape)
y = y.flatten()

diff = x - pred
dist = np.linalg.norm(diff, axis=(-1, -2))
global_scores = dist.flatten()

outliers = global_scores[label.astype(bool)]
inliers = global_scores[np.invert(label.astype(bool))]

means = []
for item in np.unique(y):
    relevant_indices = [i for i in range(len(y)) if y[i] == item]
    relevant_x = x[relevant_indices]
    relevant_labels = label[relevant_indices].astype(bool)
    mean = np.mean(relevant_x)
    mean_out = np.mean(relevant_x[relevant_labels])
    means.append((mean, mean_out))

plt.plot(np.arange(len(means)), [m[0] for m in means], "o", color="blue")
plt.plot(np.arange(len(means)), [m[1] for m in means], "o", color="red")
plt.show()




outlier_indices = []
for i in range(len(label)):
    if label[i]:
        outlier_indices.append(i)

inlier_indices = []
for i in range(len(label)):
    if not label[i]:
        inlier_indices.append(i)

plt.plot(outlier_indices, outliers, 'o', color="red")
plt.plot(inlier_indices, inliers, 'o', color="blue", alpha=0.1)
plt.show()
