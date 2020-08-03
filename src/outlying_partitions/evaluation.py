import os
import numpy as np
import pandas as pd

from .t_test import t_statistic, evaluate_array_t_statistic

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


def parse_filename(file):
    components = file.split("_")
    c_name = components[-2]
    l_name = components[-1]
    num_devices = components[0]
    frac = components[3]
    return num_devices, frac, c_name, l_name


def load_all_in_dir(directory):
    all_files = {}
    all_labels = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                filepath = os.path.join(directory, file)
                labels_suffix = "_labels.npy"
                if file.endswith(labels_suffix):
                    all_labels[file[:-len(labels_suffix)]] = np.load(filepath)
                result_file = np.load(filepath)
                all_files[file] = result_file
    return all_files, all_labels


def plot_os_star_hist(from_dir):

    def create_plots(file_dict):
        def get_os_star(f):
            return np.mean(f[0], axis=-1)

        def create_hist(os_stars, label):
            plt.hist(os_stars, label=label, bins=7)

        for i, key in enumerate(file_dict):
            file = file_dict[key]
            os_star = get_os_star(file[0])
            ax = plt.subplot(3, 2, i+1)
            create_hist(os_star, "$sf=$")
        plt.show()

    fs = load_all_in_dir(from_dir)
    create_plots(file_dict=fs)


def plot_t_test_over(x, directory):

    file_dict, label_dict = load_all_in_dir(directory)

    def over_frac():
        fracs = []
        means_t = []
        means_p = []
        for key in file_dict:
            num_devices, frac, c_name, l_name = parse_filename(key)
            fracs.append(frac)
            f = file_dict[key]
            labels = label_dict[key].astype(np.bool)
            results = evaluate_array_t_statistic(f)
            t_values = results.T[0][labels]
            p_values = results.T[1][labels]
            means_t.append(np.mean(t_values))
            means_p.append(np.mean(p_values))
        plt.plot(fracs, means_p, label="p-value")
        plt.plot(fracs, means_t, linestyle="--", label="t-value")
        plt.xlabel("Subspace fraction")
        plt.ylabel("t, p")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def over_devices():
        devices = []
        means_t = []
        means_p = []
        for key in file_dict:
            num_devices, frac, c_name, l_name = parse_filename(key)
            devices.append(frac)
            f = file_dict[key]
            labels = label_dict[key].astype(np.bool)
            results = evaluate_array_t_statistic(f)
            t_values = results.T[0][labels]
            p_values = results.T[1][labels]
            means_t.append(np.mean(t_values))
            means_p.append(np.mean(p_values))
        plt.plot(devices, means_p, label="p-value")
        plt.plot(devices, means_t, linestyle="--", label="t-value")
        plt.xlabel("Subspace fraction")
        plt.ylabel("t, p")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if x == "frac":
        over_frac()
    if x == "devices":
        over_devices()
