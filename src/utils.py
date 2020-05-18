import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_json(filepath):
    with open(filepath) as file:
        return json.load(file)


def matrix(json_file, key, diagonal_value=0):
    num_pairs = len(json_file[0])
    num_attributes = num_pairs + 1
    mat = np.zeros((num_attributes, num_attributes)) if diagonal_value == 0 else np.ones((num_attributes, num_attributes))
    for i, row in enumerate(json_file):
        for j, item in enumerate(row):
            mat[i][i+j+1] = item[key]
            mat[i+j+1][i] = item[key]
    return mat


def normalize(array_like):
    min_val = array_like.min()
    max_val = array_like.max()
    return (array_like-min_val)/(max_val-min_val)


def parse_filename(name):
    raw = name.split("_")
    params = {
        "num_devices": int(raw[0]),
        "n": int(raw[1]),
        "dims": int(raw[2]),
        "subspace_frac": float(raw[3]),
        "frac_outlying_devices": float(raw[4]),
        "frac_outlying_data": float(raw[5]),
        "gamma": float(raw[6]),
        "delta": float(raw[7]),
    }
    return params

