import numpy as np
from sklearn.metrics import roc_curve, auc
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
newshape = (oldshape[0]*oldshape[1], 28, 28)

x = x.reshape(newshape)
pred = pred.reshape(newshape)
y = y.flatten()

diff = x - pred
dist = np.linalg.norm(diff, axis=(-1, -2))
global_scores = dist.flatten()

for val in np.unique(y):
    global_scores[y == val] = global_scores[y == val] / np.mean(global_scores[y == val])

outliers = global_scores[label == 1]
inliers = global_scores[label != 1]

fpr, tpr, thresholds = roc_curve(label.flatten() == 1, global_scores)
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

means = []
for item in np.unique(y):
    relevant_indices = [i for i in range(len(y)) if y[i] == item]
    relevant_x = x[relevant_indices]
    relevant_labels = label[relevant_indices].astype(bool)
    mean = np.mean(relevant_x)
    mean_out = np.mean(relevant_x[relevant_labels])
    mean_out = mean_out / mean
    mean = mean / mean
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
