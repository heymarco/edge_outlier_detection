import os
import numpy as np
import pandas as pd
import seaborn as sns

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
    result_df = []

    for key in file_dict:
        params = parse_filename(key)
        if x == "frac":
            x_axis_vals.append(float(params["subspace_frac"]))
        elif x == "devices":
            x_axis_vals.append(float(params["num_devices"]))
        elif x == "shift":
            shift = float(params["shift"])
            affected_dims = float(params["subspace_frac"])*float(params["dims"])
            avg_dist_from_mean = np.sqrt(affected_dims*(shift**2))
            x_axis_vals.append(avg_dist_from_mean)
        else:
            print("No valid x-identifier provided")
        f = file_dict[key]
        for rep in f:
            scores = rep[0]
            labels = rep[1]
            distributed_shape = (int(params["num_devices"]), int(params["num_data"]))
            scores = scores.reshape(distributed_shape)
            labels = labels.reshape(distributed_shape)
            labels = np.any(labels, axis=-1)
            results = evaluate_array_t_statistic(scores)
            for i, res in enumerate(results):
                result_df.append([x_axis_vals[-1], res[0], res[1], labels[i]])
    result_df = pd.DataFrame(result_df, columns=["x", "t", "p", "outlier"])
    result_df = result_df
    fig, axes = plt.subplots(1, 2)
    ax1 = axes[0]
    ax2 = axes[1]
    sns.lineplot(data=result_df, x="x", y="t", hue="outlier", ax=ax1)
    sns.lineplot(data=result_df, x="x", y="p", hue="outlier", ax=ax2)

    # ax2.set_yscale("log")
    if x == "frac":
        ax1.set_xlabel("Subspace fraction")
        ax2.set_xlabel("Subspace fraction")
    elif x == "devices":
        ax1.set_xlabel("Total number of devices")
        ax2.set_xlabel("Total number of devices")
    elif x == "shift":
        ax1.set_xlabel("$dist(mean(X_{out})-mean(X^G))\ [Std]$")
        ax2.set_xlabel("$dist(mean(X_{out})-mean(X^G))\ [Std]$")
    ax1.set_ylabel("$t$-value")
    ax1.legend()
    ax2.set_ylabel("$p$-value")

    ax_alpha = ax2.twinx()
    # ax_alpha.set_yscale("log")
    ax_alpha.set_ylim(ax2.get_ylim())
    alpha_vals = [0.001, 0.01, 0.05]
    for val in alpha_vals:
        ax_alpha.axhline(val, c="black", lw=0.7, ls="dotted")
    ax_alpha.set_yticks(alpha_vals)
    ax_alpha.set_yticklabels([r"$\alpha={}$".format(val) for val in alpha_vals])

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
