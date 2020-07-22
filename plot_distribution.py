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

print(np.unique(y, return_counts=True))