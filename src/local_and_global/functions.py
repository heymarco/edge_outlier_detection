import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from xstream.python.Chains import Chains
from src.models import create_model, create_models


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
        l = [Chains(k=100, nchains=100, depth=15) for _ in range(num_clients)]
    if l_name == "ae":
        l = [create_model(shape[-1], compression_factor=0.4) for _ in range(num_clients)]
    if l_name == "if":
        l = [IsolationForest(contamination=contamination) for _ in range(num_clients)]
    if not l:
        raise KeyError("No valid local outlier detector name provided.")
    return np.array(c), np.array(l)
