import os
import numpy as np
from scipy.stats import tmean, gmean
import pandas as pd
import seaborn as sns

from .t_test import t_statistic, evaluate_array_t_statistic
from src.utils import load_all_in_dir

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


ensemble_suffix = "_aa_bb.npy"
qualitative_cp = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


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


def plot_t_test_shift(directory):
    sns.set_palette(sns.color_palette(qualitative_cp))
    file_dict = load_all_in_dir(directory)

    x_axis_vals = []
    result_df = []

    for key in file_dict:
        params = parse_filename(key)
        shift = float(params["shift"])
        affected_dims = float(params["subspace_frac"]) * float(params["dims"])
        avg_dist_from_mean = np.sqrt(affected_dims * (shift ** 2))
        x_axis_vals.append(avg_dist_from_mean)
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
    fig, axes = plt.subplots(2, 1, sharex="all")
    ax1 = axes[0]
    ax2 = axes[1]

    def estimator(x):
        low = np.quantile(x, 0.2)
        high = np.quantile(x, 0.8)
        return tmean(x, [low, high])

    sns.lineplot(data=result_df, x="x", y="t", hue="outlier", ax=ax1, ci=90, estimator=estimator)
    sns.lineplot(data=result_df, x="x", y="p", hue="outlier", ax=ax2, ci=90, estimator=estimator)

    ax1.set_xlabel("Shift [Std]")
    ax2.set_xlabel("Shift [Std]")
    ax1.set_ylabel("$t$-value")
    ax1.legend()
    ax2.set_ylabel("$p$-value")

    ax_alpha = ax2.twinx()
    ax_alpha.set_ylim(ax2.get_ylim())
    alpha_vals = [0.05]
    for val in alpha_vals:
        ax_alpha.axhline(val, c="black", lw=0.7, ls="dotted")
    ax_alpha.set_yticks(alpha_vals)
    ax_alpha.set_yticklabels([r"$\alpha={}$".format(val) for val in alpha_vals])

    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=len(handles), title="Outlier")

    for ax in [ax1, ax2]:
        ax.get_legend().remove()

    plt.show()


def plot_t_test_frac(directory):
    sns.set_palette(sns.color_palette(qualitative_cp))
    file_dict = load_all_in_dir(directory)

    x_axis_vals = []
    result_df = []

    for key in file_dict:
        params = parse_filename(key)
        frac = float(params["subspace_frac"])
        x_axis_vals.append(frac)
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
                result_df.append([x_axis_vals[-1], res[0], res[1], labels[i], params["shift"], params["subspace_frac"]])

    result_df = pd.DataFrame(result_df, columns=["x", "t", "p", "outlier", "shift", "sf"])
    tested_shift = np.unique(result_df["shift"])

    fig, axs = plt.subplots(len(tested_shift), sharex="all", sharey="all")

    axs[0].set_xlabel("Subspace fraction")
    axs[0].set_ylabel("$p$-value")

    for ax, shift in zip(axs, tested_shift):
        selection = result_df["shift"] == shift
        sns.violinplot(data=result_df[selection], x="x", y="p", hue="outlier",
                       ax=ax, cut=0, split=True, scale="width")

    for ax in axs:
        ax_alpha = ax.twinx()
        ax_alpha.set_ylim(ax.get_ylim())
        alpha_vals = [0.05]
        for val in alpha_vals:
            ax_alpha.axhline(val, c="black", lw=0.7, ls="dotted")
        ax_alpha.set_yticks(alpha_vals)
        ax_alpha.set_yticklabels([r"$\alpha={}$".format(val) for val in alpha_vals])

    handles, labels = axs[0].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=len(handles), title="Outlier")

    pad = 5
    for ax, shift in zip(axs, tested_shift):
        ax.set_ylabel("AUPR")
        ax.annotate(shift, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    plt.show()

def plot_t_test_over(x, directory):
    sns.set_palette(sns.color_palette(qualitative_cp))
    file_dict = load_all_in_dir(directory)

    x_axis_vals = []
    result_df = []

    for key in file_dict:
        params = parse_filename(key)
        if x == "frac":
            frac = float(params["subspace_frac"])
            if (round(frac, 2) * 100) % 10 != 0:
                continue
            x_axis_vals.append(frac)
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
    fig, axes = plt.subplots(2, 1, sharex="all")
    ax1 = axes[0]
    ax2 = axes[1]

    def estimator(x):
        low = np.quantile(x, 0.2)
        high = np.quantile(x, 0.8)
        return tmean(x, [low, high])

    sns.violinplot(data=result_df, split=True, x="x", y="t", hue="outlier", ax=ax1, scale="width",
                   palette="Set2")
    sns.violinplot(data=result_df, x="x", y="p", hue="outlier", ax=ax2,
                   cut=0, palette="Set2", split=True, scale="width")

    if x == "frac":
        ax1.set_xlabel("")
        ax2.set_xlabel("Subspace fraction")
    elif x == "devices":
        ax1.set_xlabel("Total number of devices")
        ax2.set_xlabel("Total number of devices")
    elif x == "shift":
        ax1.set_xlabel("Shift [Std]")
        ax2.set_xlabel("Shift [Std]")
    ax1.set_ylabel("$t$-value")
    ax1.legend()
    ax2.set_ylabel("$p$-value")

    ax_alpha = ax2.twinx()
    ax_alpha.set_ylim(ax2.get_ylim())
    alpha_vals = [0.05]
    for val in alpha_vals:
        ax_alpha.axhline(val, c="black", lw=0.7, ls="dotted")
    ax_alpha.set_yticks(alpha_vals)
    ax_alpha.set_yticklabels([r"$\alpha={}$".format(val) for val in alpha_vals])

    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=len(handles), title="Outlier")

    for ax in [ax1, ax2]:
        ax.get_legend().remove()

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
