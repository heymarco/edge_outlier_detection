import os
import numpy as np
import pandas as pd

from .t_test import t_statistic, evaluate_array_t_statistic

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


labels_suffix = "_labels.npy"
ensemble_suffix = "_aa_bb.npy"


def load_all_in_dir(directory):
    all_files = {}
    all_labels = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                filepath = os.path.join(directory, file)
                if file.endswith(labels_suffix):
                    all_labels[file[:-len(labels_suffix)]] = np.load(filepath)
                    continue
                result_file = np.load(filepath)
                all_files[file[:-(len(ensemble_suffix))]] = result_file
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

    x_axis_vals = []
    means_t = []
    means_p = []
    for key in file_dict:
        num_devices, frac, c_name, l_name, cont = parse_filename(key)
        if x == "frac":
            x_axis_vals.append(frac)
        elif x == "devices":
            x_axis_vals.append(num_devices)
        elif x == "cont":
            x_axis_vals.append(cont)
        else:
            print("No valid x-identifier provided")
        f = file_dict[key]
        labels = label_dict[key]
        these_results_p = []
        these_results_t = []
        for rep in f:
            results = evaluate_array_t_statistic(rep)
            t_values = results.T[0][labels]
            p_values = results.T[1][labels]
            these_results_t.append(np.mean(t_values))
            these_results_p.append(np.mean(p_values))
        means_t.append(np.mean(these_results_t))
        means_p.append(np.mean(these_results_p))
    sorted_indices = np.argsort(x_axis_vals)
    x_axis_vals = np.sort(x_axis_vals)
    means_t = np.array(means_t)[sorted_indices]
    means_p = np.array(means_p)[sorted_indices]
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(x_axis_vals, means_p, label="p-value")
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line2 = ax2.plot(x_axis_vals, means_t, linestyle="--", label="t-value")
    if x == "frac":
        ax1.set_xlabel("Subspace fraction")
    elif x == "cont":
        ax1.set_xlabel("Fraction of outlying observations")
    elif x == "devices":
        ax1.set_xlabel("Total number of devices")
    ax1.set_ylabel("$p$-value")
    ax2.set_ylabel("$t$-value")
    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    plt.legend(lines, labs)
    plt.tight_layout()
    plt.show()


def parse_filename(file):
    keys = [
        "num_devices",
        "num_data",
        "dims",
        "subspace_frac",
        "frac_outlying_devices",
        "sigma_l",
        "shift"
    ]
    components = file.split("_")
    assert len(components) == len(keys)
    parsed_args = {}
    for i in range(len(keys)):
        parsed_args[keys[i]] = components[i]
    return parsed_args
