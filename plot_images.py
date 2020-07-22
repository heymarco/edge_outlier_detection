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

oldshape = x.shape
newshape = (oldshape[0]*oldshape[1], 28, 28)

x = x.reshape(newshape)
pred = pred.reshape(newshape)
y = y.flatten()


plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image = pred[y == i][0]
    plt.imshow(image)
plt.show()
