import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

x = np.load("original.npy")
y = np.load("labels.npy")
predicted = np.load("predicted.npy")

oldshape = x.shape
newshape = (x.shape[0] * x.shape[1], 28, 28)

x = x.reshape(newshape)
predicted = predicted.reshape(newshape)
y = y.flatten()

# global scores
diff = np.abs(predicted - x)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image = diff[y == i][0]
    plt.imshow(image)
plt.show()

diff = diff.reshape((oldshape[0], oldshape[1], 28 * 28))
dist = np.linalg.norm(diff, axis=-1)
print(dist.shape)
global_scores = dist.flatten()

labels = np.arange(100)
accumulated_result = []
for value in labels:
    mean_score = np.mean(global_scores[y.flatten() == value])
    accumulated_result.append(mean_score)

plt.bar(np.arange(len(accumulated_result)), accumulated_result)
plt.show()