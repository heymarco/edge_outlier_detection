import os
import pandas as pd
import numpy as np


def create_federated_festool_dataset(num_clients: int=10, sample_length: int = 30, flatten: bool = True):
    magic_df = pd.read_csv(os.path.join("..", "..", "data", "magic.csv"), sep=";", header=0)

    ts_columns = ["current_PDC", "moment_PDC", "omega_PDC"]
    label_columns = [col for col in list(magic_df.columns) if col not in ts_columns]

    screw_types = {type: i for i, type in enumerate(np.sort(np.unique(magic_df["schraubenlaengeFloat"])))}
    screw_type_labels = [screw_types[type] for type in magic_df["schraubenlaengeFloat"]]
    magic_df["screw_type"] = screw_type_labels

    magic_df = magic_df[magic_df["kopflabel"] != "#inactive"].reset_index()

    ts_data = magic_df[ts_columns].to_numpy()
    all_labels = magic_df[label_columns]

    outlier_indices = list(magic_df[magic_df["screw_type"] == 1].index) + list(magic_df[magic_df["screw_type"] == 2].index)

    outlier_data = ts_data[outlier_indices]
    inlier_data = np.delete(ts_data, outlier_indices, axis=0)
    outlier_data = outlier_data[:len(outlier_data)-(len(outlier_data)%sample_length)]
    inlier_data = inlier_data[:len(inlier_data) - (len(inlier_data) % (sample_length*(num_clients-1)))]
    outlier_data = np.reshape(outlier_data, newshape=(int(len(outlier_data)/sample_length),
                                                      sample_length,
                                                      3))
    inlier_data = np.reshape(inlier_data, newshape=(num_clients-1,
                                                    int(len(inlier_data) / ((num_clients-1)*sample_length)),
                                                    sample_length,
                                                    inlier_data.shape[-1]))

    datasets = [outlier_data]
    datasets += [data for data in inlier_data]

    if flatten:
        [data.reshape((data.shape[0], data.shape[1]*data.shape[2])) for data in datasets]

    return datasets
