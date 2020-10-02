import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from xstream.python.Chains import Chains
from src.models import create_model, create_models
from src.training import train_federated

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


def create_ensembles(shape, l_name, contamination=0.01):
    """
    Utility function for creating the ensembles
    :param shape: The input shape
    :param l_name: The idenfitier of the local outlier detector
    :param contamination: The contamination (for some models this is a parameter)
    :return: array(C), array(L)
    """
    num_clients = shape[0]
    c = create_models(num_clients, shape[-1], compression_factor=0.4)
    l = None
    if l_name == "lof1":
        l = [LocalOutlierFactor(n_neighbors=1, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof2":
        l = [LocalOutlierFactor(n_neighbors=2, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof4":
        l = [LocalOutlierFactor(n_neighbors=4, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof8":
        l = [LocalOutlierFactor(n_neighbors=8, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof16":
        l = [LocalOutlierFactor(n_neighbors=16, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof32":
        l = [LocalOutlierFactor(n_neighbors=32, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof64":
        l = [LocalOutlierFactor(n_neighbors=64, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof100":
        l = [LocalOutlierFactor(n_neighbors=100, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "xstream":
        l = [Chains(k=100, nchains=100, depth=15) for _ in range(num_clients)]
    if l_name == "ae":
        l = [create_model(shape[-1], compression_factor=0.4) for _ in range(num_clients)]
    if l_name == "if":
        l = [IsolationForest(contamination=contamination) for _ in range(num_clients)]
    if not l:
        raise KeyError("No valid local outlier detector name provided.")
    return np.array(c), np.array(l)


def train_global_detectors(data, collab_detectors, global_epochs):
    # federated training
    for _ in range(global_epochs):
        collab_detectors = train_federated(models=collab_detectors, data=data, epochs=1, batch_size=32,
                                           frac_available=1.0)

    # global scores
    predicted = np.array([model.predict(data[i]) for i, model in enumerate(collab_detectors)])
    diff = predicted - data
    global_scores = np.linalg.norm(diff, axis=-1)
    return global_scores


def train_local_detectors(data, local_detectors, global_epochs, l_name):
    print("Fitting {}".format(l_name))
    # local training
    if l_name == "lof" or l_name == "if" or l_name == "xstream":
        [l.fit(data[i]) for i, l in enumerate(local_detectors)]
    if l_name == "ae":
        [l.fit(data[i], data[i],
               batch_size=32, epochs=global_epochs) for i, l in enumerate(local_detectors)]

    # local scores
    if l_name == "lof":
        local_scores = - np.array([model.negative_outlier_factor_ for i, model in enumerate(local_detectors)])
    if l_name == "xstream":
        local_scores = np.array([-model.score(data[i]) for i, model in enumerate(local_detectors)])
    if l_name == "if":
        local_scores = -np.array([model.score_samples(data[i]) for i, model in enumerate(local_detectors)])
    if l_name == "ae":
        predicted = np.array([model.predict(data[i]) for i, model in enumerate(local_detectors)])
        diff = predicted - data
        dist = np.linalg.norm(diff, axis=-1)
        local_scores = np.reshape(dist, newshape=(data.shape[0], data.shape[1]))
    return local_scores
