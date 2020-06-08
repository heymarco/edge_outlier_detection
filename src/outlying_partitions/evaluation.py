import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr, zscore

from xstream.python.Chains import Chains
from src.models import create_model, create_models, train_federated
from src.utils import color_palette

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


def create_ensembles(shape, l_name, contamination=0.01):
    num_clients = shape[0]
    c = create_models(num_clients, shape[-1], compression_factor=0.4)
    l = None
    if l_name == "lof":
        l = [LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "xstream":
        l = [Chains(k=50, nchains=10, depth=10) for _ in range(num_clients)]
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
    predicted = np.array([model.predict(data[i]) for i, model in enumerate(collab_detectors)])
    diff = predicted - data
    dist = np.linalg.norm(diff, axis=-1)
    global_scores = dist.flatten()

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

    return global_scores, local_scores


def score(result_global, result_local):
    assert len(result_local) == len(result_global)
    scores = []
    for i in range(len(result_local)):
        rl = result_local[i]
        rg = np.reshape(result_global[i], newshape=rl.shape)
        s = np.array([spearmanr(rl[i], rg[i])[0] for i in range(len(rl))])
        scores.append(s)
    return np.mean(np.array(scores), axis=0)


def evaluate(scores, ground_truth):
    is_candidate = ground_truth.any(axis=1)
    mean_score_normal = np.mean(scores[np.invert(is_candidate)])
    mean_score_outlying = np.mean(scores[is_candidate])
    return mean_score_normal, mean_score_outlying


def plot_result():
    # read from dir
    directory = os.path.join(os.getcwd(), "results", "numpy", "outlying_partitions")

    def parse_filename(file):
        components = file.split("_")
        c_name = components[-2]
        l_name = components[-1]
        num_devices = components[0]
        frac = components[3]
        contamination = components[5]
        return num_devices, frac, c_name, l_name, contamination

    names = {
        "ae": "AE",
        "if": "IF",
        "xstream": "xStream",
        "lof": "LOF"
    }
    res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                num_devices, frac, c_name, l_name, contamination = parse_filename(file[:-4])
                result = np.load(os.path.join(directory, file))
                c = names[c_name]
                l = names[l_name]
                new_res = [int(num_devices),
                           float(frac),
                           float(contamination),
                           "{}/{}".format(c, l),
                           result[0],
                           "All Partitions"]
                res.append(new_res)
                new_res = [int(num_devices),
                           float(frac),
                           float(contamination),
                           "{}/{}".format(c, l),
                           result[1],
                           "Partition w/ inliers"]
                res.append(new_res)
                new_res = [int(num_devices),
                           float(frac),
                           float(contamination),
                           "{}/{}".format(c, l),
                           result[2],
                           "Partition w/ outliers"]
                res.append(new_res)

    mpl.rc('font', **{"size": 12})
    d = {'color': color_palette, "marker": ["o", "*", "v", "x"]}
    df = pd.DataFrame(res,
                      columns=["\# Devices", "Subspace frac", "Contamination", "Ensemble",
                                    "mean $OS^{C}$", "Type"])
    df = df.sort_values(by=["Ensemble", "Contamination"])
    g = sns.FacetGrid(df, col="Ensemble", row="\# Devices", hue="Type", hue_kws=d, margin_titles=True)
    g.map(plt.plot, "Contamination", "mean $OS^{C}$").add_legend()

    # plt.tight_layout()
    plt.show()
