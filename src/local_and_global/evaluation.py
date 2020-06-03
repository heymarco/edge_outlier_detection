import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from src.metrics import kappa_m, f1_score
from xstream.python.Chains import Chains
from src.models import create_model, create_models, train_federated
from src.local_outliers.evaluation import retrieve_labels

import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')
from matplotlib.offsetbox import AnchoredText


def load_rw(dirname):
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


def create_ensembles(shape, l_name, contamination=0.01):
    num_clients = shape[0]
    c = create_models(num_clients, shape[-1], compression_factor=0.4)
    l = None
    if l_name == "lof":
        l = [LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "xstream":
        l = [Chains(k=50, nchains=50, depth=10) for _ in range(num_clients)]
    if l_name == "ae":
        l = [create_model(shape[-1], compression_factor=0.4) for _ in range(num_clients)]
    if l_name == "if":
        l = [IsolationForest(contamination=contamination) for _ in range(num_clients)]
    if not l:
        raise KeyError("No valid local outlier detector name provided.")
    return np.array(c), np.array(l)


def train_ensembles(data, ensembles, l_name, global_epochs=10):
    collab_detectors = ensembles[0]
    local_detectors = ensembles[1]

    # federated training
    for _ in range(global_epochs):
        collab_detectors = train_federated(models=collab_detectors, data=data, epochs=1, batch_size=32, frac_available=1.0)

    # global scores
    predicted = np.array([m.predict(data[i]) for i, m in enumerate(collab_detectors)])
    diff = predicted - data
    dist = np.linalg.norm(diff, axis=-1)
    global_scores = dist.flatten()

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

    return global_scores, local_scores


def classify(result_global, result_local, contamination=0.01):
    assert len(result_local) == len(result_global)
    labels = []
    for i in range(len(result_local)):
        print("shape of global result: {}".format(result_global[i].shape))
        print("shape of local result: {}".format(result_local[i].shape))
        labels_global = retrieve_labels(result_global[i], contamination).flatten()
        labels_local = retrieve_labels(result_local[i], contamination).flatten()
        print(result_local[i].shape)
        # remove candidates for abnormal data partitions
        labels_global[np.logical_and(labels_global, np.invert(labels_local))] = 0
        classification = np.empty(shape=labels_global.shape)
        classification.fill(0)
        is_global_outlier = np.logical_and(labels_global, labels_local)
        classification[is_global_outlier] = 2
        is_local_outlier = np.logical_and(labels_local, np.invert(is_global_outlier))
        classification[is_local_outlier] = 1
        labels.append(classification)
    return np.array(labels)


def evaluate(labels, ground_truth, contamination):
    ground_truth = ground_truth.flatten()
    kappa = []
    f1_local = []
    f1_global = []
    for lbs in labels:
        lbs = lbs.flatten()
        kappa.append(kappa_m(lbs, ground_truth, 1-contamination))
        f1_local.append(f1_score(lbs, ground_truth, relevant_label=1))
        f1_global.append(f1_score(lbs, ground_truth, relevant_label=2))
    return np.mean(kappa), np.mean(f1_global), np.mean(f1_local)


def plot_result():
    # read from dir
    directory = os.path.join(os.getcwd(), "results", "numpy", "local_and_global")

    def parse_filename(file):
        components = file.split("_")
        c_name = components[-2]
        l_name = components[-1]
        num_devices = components[0]
        return num_devices, c_name, l_name

    results_kappa = {}
    results_f1_global = {}
    results_f1_local = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                num_devices, c_name, l_name = parse_filename(file[:-4])
                result = np.load(os.path.join(directory, file))
                key = "{}, {}".format(c_name, l_name)
                if key not in results_kappa:
                    results_kappa[key] = {}
                if key not in results_f1_global:
                    results_f1_global[key] = {}
                if key not in results_f1_local:
                    results_f1_local[key] = {}
                results_kappa[key][int(num_devices)] = result[0]
                results_f1_global[key][int(num_devices)] = result[1]
                results_f1_local[key][int(num_devices)] = result[2]

    df_kappa = pd.DataFrame(results_kappa)
    df_kappa.sort_index(inplace=True)
    df_f1_global = pd.DataFrame(results_f1_global).sort_index()
    df_f1_local = pd.DataFrame(results_f1_local).sort_index()

    fig, axes = plt.subplots(1, 3)
    sns.lineplot(data=df_kappa, ax=axes[0])
    sns.lineplot(data=df_f1_global, ax=axes[1])
    sns.lineplot(data=df_f1_local, ax=axes[2])
    plt.xlabel("k")
    axes[0].set_ylabel("$\kappa_m$")
    axes[1].set_ylabel("$f1_{global}$")
    axes[2].set_ylabel("$f1_{local}$")
    plt.tight_layout()
    plt.show()
