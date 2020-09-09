import os
import numpy as np
import pandas as pd

from .t_test import t_statistic, evaluate_array_t_statistic

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


ensemble_suffix = "_aa_bb.npy"


def load_all_in_dir(directory):
    all_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                filepath = os.path.join(directory, file)
                result_file = np.load(filepath)
                all_files[file[:-(len(ensemble_suffix))]] = result_file
    print(all_files.keys())
    return all_files


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

    file_dict = load_all_in_dir(directory)

    x_axis_vals = []
    means_t_outlier = []
    means_p_outlier = []
    means_t_inlier = []
    means_p_inlier = []

    for key in file_dict:
        params = parse_filename(key)
        if x == "frac":
            x_axis_vals.append(params["subspace_frac"])
        elif x == "devices":
            x_axis_vals.append(params["num_devices"])
        elif x == "shift":
            shift = float(params["shift"])
            affected_dims = float(params["subspace_frac"])*float(params["dims"])
            avg_dist_from_mean = np.sqrt(affected_dims*(shift**2))
            print(avg_dist_from_mean)
            x_axis_vals.append(avg_dist_from_mean)
        else:
            print("No valid x-identifier provided")
        f = file_dict[key]
        mean_results_p_outlier = []
        mean_results_t_outlier = []
        mean_results_p_inlier = []
        mean_results_t_inlier = []
        for rep in f:
            scores = rep[0]
            labels = rep[1]
            distributed_shape = (int(params["num_devices"]), int(params["num_data"]))
            scores = scores.reshape(distributed_shape)
            labels = labels.reshape(distributed_shape)
            labels = np.any(labels, axis=-1)
            results = evaluate_array_t_statistic(scores)
            t_values_outlier = results.T[0][labels]
            p_values_outlier = results.T[1][labels]
            t_values_inlier = results.T[0][np.invert(labels)]
            p_values_inlier = results.T[1][np.invert(labels)]
            mean_results_t_outlier.append(np.mean(t_values_outlier))
            mean_results_p_outlier.append(np.mean(p_values_outlier))
            mean_results_t_inlier.append(np.mean(t_values_inlier))
            mean_results_p_inlier.append(np.mean(p_values_inlier))
        means_t_outlier.append(np.mean(mean_results_t_outlier))
        means_p_outlier.append(np.mean(mean_results_p_outlier))
        means_t_inlier.append(np.mean(mean_results_t_inlier))
        means_p_inlier.append(np.mean(mean_results_p_inlier))
    sorted_indices = np.argsort(x_axis_vals)
    x_axis_vals = np.sort(x_axis_vals)
    means_t_outlier = np.array(means_t_outlier)[sorted_indices]
    means_p_outlier = np.array(means_p_outlier)[sorted_indices]
    means_t_inlier = np.array(means_t_inlier)[sorted_indices]
    means_p_inlier = np.array(means_p_inlier)[sorted_indices]
    fig, axes = plt.subplots(1, 2)
    ax1 = axes[0]
    ax2 = axes[1]
    ax1.plot(x_axis_vals, means_t_outlier, linestyle="--", label="outliers")
    ax1.plot(x_axis_vals, means_t_inlier, linestyle="--", label="inliers")
    ax2.plot(x_axis_vals, means_p_outlier, label="outliers")
    ax2.plot(x_axis_vals, means_p_inlier, label="inlier")
    ax2.set_yscale("log")
    if x == "frac":
        ax1.set_xlabel("Subspace fraction")
        ax2.set_xlabel("Subspace fraction")
    elif x == "devices":
        ax1.set_xlabel("Total number of devices")
        ax2.set_xlabel("Total number of devices")
    elif x == "shift":
        ax1.set_xlabel("Deviation of outlying partition")
        ax2.set_xlabel("Deviation of outlying partition")
    ax1.set_ylabel("$t$-value")
    ax1.legend()
    ax2.set_ylabel("$p$-value")
    ax2.legend()
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
    parsed_args = {}
    for i in range(len(keys)):
        parsed_args[keys[i]] = components[i]
    return parsed_args
