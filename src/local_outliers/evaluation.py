import os
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from xstream.python.Chains import Chains
from src.models import create_model


def load_data(dirname):
    if not dirname == "synth":
        data = []
        directory = os.path.join(os.getcwd(), "data", dirname)
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    print("Read {}".format(file))
                    d = np.loadtxt(os.path.join(directory, file), skiprows=1, delimiter=",")
                    if dirname == "xdk" or "mhealth":
                        d = d[:, :-1]
                    data.append(d)
        data = np.array(data)
    else:
        data = np.load(os.path.join(os.getcwd(), "data", "synth", "10_1000_100_1.0_1.0_0.01_0.5_0.2_local_d.npy"))
    return data


# XSTREAM
def fit_predict_xstream_global(data, k=50, nchains=50, depth=10):
    assert data.ndim == 3, "Error, data must have 3 dimensions but has {}".format(data.ndim)
    data = np.reshape(data,
                      newshape=(data.shape[0] * data.shape[1], data.shape[2]))
    cf = Chains(k=k, nchains=nchains, depth=depth)
    cf.fit(data)
    predictions = -cf.score(data)
    return predictions


def fit_predict_xstream_local(data, k=50, nchains=50, depth=10):
    assert data.ndim == 3, "Error, data must have 3 dimensions but has {}".format(data.ndim)
    models = [Chains(k=k, nchains=nchains, depth=depth) for _ in range(len(data))]
    [model.fit(data[i]) for i, model in enumerate(models)]
    predicted = np.array([-model.score(data[i]) for i, model in enumerate(models)])
    return predicted


# AUTOENCODER
def fit_predict_autoencoder_local(data, compression_factor=0.8):
    assert data.ndim == 3, "Error, data must have 3 dimensions but has {}".format(data.ndim)
    models = [create_model(data.shape[-1], compression_factor) for i in range(len(data))]
    [m.fit(data[i], data[i],
           batch_size=32,
           epochs=10, shuffle=True) for i, m in enumerate(models)]
    predicted = np.array([m.predict(data[i]) for i, m in enumerate(models)])
    diff = predicted - data
    dist = np.linalg.norm(diff, axis=-1)
    dist = np.reshape(dist, newshape=(data.shape[0], data.shape[1]))
    return dist


def fit_predict_autoencoder_global(data, compression_factor=0.8):
    assert data.ndim == 3, "Error, data must have 3 dimensions but has {}".format(data.ndim)
    data = np.reshape(data,
                      newshape=(data.shape[0] * data.shape[1], data.shape[2]))
    model = create_model(data.shape[-1], compression_factor)
    model.fit(data, data,
              batch_size=32,
              epochs=10, shuffle=True)
    predicted = model.predict(data)
    diff = predicted - data
    dist = np.linalg.norm(diff, axis=-1)
    return dist


# Local Outlier Factor
def fit_predict_lof_global(data, contamination=0.01):
    data = np.reshape(data,
                      newshape=(data.shape[0] * data.shape[1], data.shape[2]))
    print(data)
    clf = LocalOutlierFactor(n_neighbors=10, contamination=contamination, novelty=False)
    clf.fit(data)
    return -clf.negative_outlier_factor_


def fit_predict_lof_local(data, contamination=0.01):
    models = [LocalOutlierFactor(n_neighbors=10, contamination=contamination) for _ in
              range(data.shape[0])]
    [model.fit(data[i]) for i, model in enumerate(models)]
    return -np.array([model.negative_outlier_factor_ for i, model in enumerate(models)])


# Isolation Forests
def fit_predict_if_global(data, contamination=0.01):
    data = np.reshape(data,
                      newshape=(data.shape[0] * data.shape[1], data.shape[2]))
    print(data)
    forest = IsolationForest(contamination=contamination)
    forest.fit(data)
    return -forest.score_samples(data)


def fit_predict_if_local(data, contamination=0.01):
    models = [IsolationForest(contamination=contamination) for _ in range(data.shape[0])]
    [model.fit(data[i]) for i, model in enumerate(models)]
    return -np.array([model.score_samples(data[i]) for i, model in enumerate(models)])


def retrieve_labels(results, contamination=0.01):
    percent = (1-contamination)*100
    if results.ndim == 1:
        # global
        thresh = np.percentile(results, percent)
        return results > thresh
    else:
        # local
        thresh = np.percentile(results, percent, axis=1, keepdims=True)
        thresh = np.repeat(thresh, results.shape[-1], axis=-1)
        return (results > thresh).flatten()


def save_result(result, name):
    np.save(file=os.path.join(os.getcwd(), "results", "numpy", "local_outliers", "{}".format(name)),
            arr=result)


def load_result(filename):
    fp = os.path.join(os.getcwd(), "results", "numpy", "local_outliers", filename)
    return np.load(fp)


def get_frac_local(result_global, result_local, contamination=0.01):
    fracs = []
    for i in range(len(result_local)):
        labels_global = retrieve_labels(result_global[i], contamination=contamination)
        labels_local = retrieve_labels(result_local[i], contamination=contamination)
        global_outliers = np.logical_and(labels_global, labels_local)
        frac = 1 - np.sum(global_outliers)/np.sum(labels_local)
        fracs.append(frac)
    return np.mean(fracs)




