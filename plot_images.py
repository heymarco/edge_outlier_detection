import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

x = np.load("original.npy")
pred = np.load("predicted.npy")

newshape = (450*100, 32, 32, 3)

x = x.reshape(newshape)
pred = pred.reshape(newshape)


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    i = -i if i > 12 else i
    plt.imshow(pred[i], cmap=plt.cm.binary)
plt.show()
