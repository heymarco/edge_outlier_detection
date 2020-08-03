import os

import numpy as np
import argparse

from src.utils import setup_machine
from src.models import create_models, create_deep_models
from src.training import train_federated, train_ensembles
from src.data.image_data import get_image_data
from src.images.functions import create_ensembles

import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str)
parser.add_argument("-gpu", type=int)
parser.add_argument("-reps", type=int, default=1)
parser.add_argument("-conv", type=bool, default=True)
args = parser.parse_args()


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

num_devices = 100
global_epochs = 20

x, y, labels = get_image_data(args.data, num_clients=num_devices)

print("Fraction of outliers: {}".format(np.sum(labels)/len(labels)))

setup_machine(cuda_device=args.gpu)

use_convolutional = args.conv

oldshape = x.shape
newshape = (x.shape[0], x.shape[1], x.shape[-3] * x.shape[-2] * x.shape[-1])

if not use_convolutional:
    x = x.reshape(newshape)

# create ensembles
combinations = [("ae", "ae"),
                # ("ae", "lof8"),
                # ("ae", "if"),
                # ("ae", "xstream")
]
print("Executing combinations {}".format(combinations))

# run ensembles on each data set
contamination = np.sum(labels > 0)/len(labels.flatten())
for c_name, l_name in combinations:
    results = []
    for i in range(args.reps):
        ensembles = create_ensembles(x.shape, l_name, contamination=contamination, use_convolutional=use_convolutional)
        global_scores, local_scores = train_ensembles(x, ensembles,
                                                      global_epochs=global_epochs, l_name=l_name, convolutional=args.conv)
        result = np.vstack((global_scores, local_scores, labels))
        results.append(result)
    fname = "{}_{}_{}".format(args.data, c_name, l_name)
    np.save(os.path.join(os.getcwd(), "results", "numpy", "images", fname), results)
