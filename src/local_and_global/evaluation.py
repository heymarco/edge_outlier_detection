import os
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from src.metrics import kappa_m
from xstream.python.Chains import Chains
from src.models import create_model, train_federated
from src.local_outliers.evaluation import retrieve_labels


def load__rw(dirname):
    data = []
    directory = os.path.join(os.getcwd(), "data", dirname)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                print("Read {}".format(file))
                d = np.loadtxt(os.path.join(directory, file), skiprows=1, delimiter=",")
                if dirname == "xdk":
                    d = d[:, :-1]
                    data.append(d)
    return np.array(data)


def load_synth(filename):
    return np.load(os.path.join(os.getcwd(), "data", "synth", filename))


def create_ensemble(shape, l_name):
    c = create_model(shape[-1], compression_factor=0.4)
    l = None
    if l_name == "lof":
        l = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
    if l_name == "xstream":
        l = Chains(k=50, nchains=50, depth=10)
    if l_name == "ae":
        l = create_model(shape[-1], compression_factor=0.4)
    if l_name == "if":
        l = IsolationForest(contamination=0.01)
    if not l:
        raise KeyError("No valid local outlier detector name provided.")
    return c, l


def train_ensemble(data, ensembles, l_name, global_epochs=10):
    local_detectors = [ensemble[1] for ensemble in ensembles]
    collab_detectors = [ensemble[0] for ensemble in ensembles]

    # federated training
    for _ in range(global_epochs):
        collab_detectors = train_federated(models=collab_detectors, data=data, epochs=1, batch_size=32, frac_available=1.0)

    # global scores
    predicted = np.array([m.predict(data[i]) for i, m in enumerate(collab_detectors)])
    diff = predicted - data
    dist = np.linalg.norm(diff, axis=-1)
    global_scores = np.reshape(dist, newshape=(data.shape[0], data.shape[1]))

    # local training
    if l_name == "lof" or "if" or "xstream":
        [l.fit(data[i]) for i, l in enumerate(local_detectors)]
    if l_name == "ae":
        [l.fit(data[i], batch_size=32, epochs=10) for i, l in enumerate(local_detectors)]

    # local scores
    if l_name == "lof":
        local_scores = - np.array([model.negative_outlier_factor_ for i, model in enumerate(local_detectors)])
    if l_name == "xstream":
        local_scores = np.array([-model.score(data[i]) for i, model in enumerate(local_detectors)])
    if l_name == "if":
        local_scores = -np.array([model.score_samples(data[i]) for i, model in enumerate(local_detectors)])
    if l_name == "ae":
        predicted = np.array([m.predict(data[i]) for i, m in enumerate(local_detectors)])
        diff = predicted - data
        dist = np.linalg.norm(diff, axis=-1)
        local_scores = np.reshape(dist, newshape=(data.shape[0], data.shape[1]))

    return global_scores, local_scores


def classify(result_global, result_local, contamination=0.01):
    assert len(result_local) == len(result_global)
    labels = []
    for i in range(len(result_local)):
        labels_global = retrieve_labels(result_global[i], contamination).flatten()
        labels_local = retrieve_labels(result_local[i], contamination).flatten()
        classification = np.empty(shape=labels_global.shape)
        classification.fill(0)
        is_global_outlier = np.logical_and(labels_global, labels_local)
        classification[is_global_outlier] = 2
        is_local_outlier = np.logical_and(labels_local, np.invert(is_global_outlier))
        classification[is_local_outlier] = 1
        labels.append(classification)
    return np.array(labels)


def evaluate(labels, ground_truth, contamination):
    kappa = []
    for lbs in labels:
        kappa.append(kappa_m(lbs, ground_truth, contamination))
    return np.mean(kappa)

