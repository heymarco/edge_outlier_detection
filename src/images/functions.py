import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from src.models import create_model, create_models, create_deep_models, create_deep_model
from xstream.python.Chains import Chains


def create_ensembles(shape, l_name, contamination=0.01, use_convolutional=True):
    num_clients = shape[0]
    if use_convolutional:
        print(shape)
        c = create_deep_models(num_clients, dims=(shape[-3], shape[-2], shape[-1]), compression_factor=0.4)
    else:
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
    if l_name == "lof10":
        l = [LocalOutlierFactor(n_neighbors=10, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof20":
        l = [LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof16":
        l = [LocalOutlierFactor(n_neighbors=16, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof32":
        l = [LocalOutlierFactor(n_neighbors=32, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof64":
        l = [LocalOutlierFactor(n_neighbors=64, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof100":
        l = [LocalOutlierFactor(n_neighbors=100, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "xstream":
        l = [Chains(k=50, nchains=50, depth=10) for _ in range(num_clients)]
    if l_name == "ae":
        if use_convolutional:
            l = [create_deep_model((shape[-3], shape[-2], shape[-1])) for _ in range(num_clients)]
        else:
            l = [create_model(shape[-1], compression_factor=0.4) for _ in range(num_clients)]
    if l_name == "if":
        l = [IsolationForest(contamination=contamination) for _ in range(num_clients)]
    if not l:
        raise KeyError("No valid local outlier detector name provided, Got {}.".format(l_name))
    return np.array(c), np.array(l)