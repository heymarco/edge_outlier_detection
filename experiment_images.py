import numpy as np
import argparse

from src.utils import setup_machine
from src.models import create_models, create_deep_models, train_federated
from src.data.image_data import get_data

import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str)
parser.add_argument("-gpu", type=int)
args = parser.parse_args()


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

setup_machine(cuda_device=0)

num_devices = 10
global_epochs = 20

use_convolutional = True

x, y, labels = get_data(args.data, num_clients=num_devices)

print("Fraction of outliers: {}".format(np.sum(labels)/len(labels)))

oldshape = x.shape
newshape = (x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])

if not use_convolutional:
    x = x.reshape(newshape)

np.save("original.npy", x)
np.save("labels.npy", y)
np.save("outliers.npy", labels)

if use_convolutional:
    print("Use convolutional network")
    models = create_deep_models(num_devices=num_devices, dims=(oldshape[-3], oldshape[-2], oldshape[-1]), compression_factor=0.4)
else:
    print("Use dense network")
    models = create_models(num_devices=num_devices, dims=oldshape[-3]*oldshape[-2], compression_factor=0.4)

for epoch in np.arange(global_epochs):
    models = train_federated(models, epochs=1, data=x)

# global scores
predicted = np.array([model.predict(x[i]) for i, model in enumerate(models)])
diff = predicted - x

if use_convolutional:
    diff = diff.reshape(newshape)

dist = np.linalg.norm(diff, axis=-1)
global_scores = dist.flatten()

np.save("predicted.npy", predicted)


print(x.shape)
print(y.shape)
print(labels.shape)
print(predicted.shape)

