import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import load_all_in_dir

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


ensemble_suffix = "_aa_bb.npy"
qualitative_cp = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def plot_evaluation_shift(directory):
    sns.set_palette(sns.color_palette(qualitative_cp))
    file_dict = load_all_in_dir(directory)

    x_axis_vals = []
    result_df = []

    for key in file_dict:
        params = parse_filename(key)
        shift = float(params["shift"])
        x_axis_vals.append(round(shift, 2))
        f = file_dict[key]
        for rep in f:
            scores = rep[0]
            labels = rep[1]
            distributed_shape = (int(params["num_devices"]), int(params["num_data"]))
            scores = scores.reshape(distributed_shape)
            labels = labels.reshape(distributed_shape)
            labels = np.any(labels, axis=-1)
            results = get_probabilities(scores)
            for i, res in enumerate(results):
                result_df.append([x_axis_vals[-1], res[0], res[1], labels[i]])

    result_df = pd.DataFrame(result_df, columns=["x", "t", "p", "Outlier"])
    fig, axes = plt.subplots(2, 1, sharex="all")
    ax1 = axes[0]
    ax2 = axes[1]

    def estimator(x):
        return np.median(x)

    summary = result_df.groupby(["x", "Outlier"]).describe()
    p_low = summary["p", "25%"]
    p_high = summary["p", "75%"]
    os_low = summary["t", "25%"]
    os_high = summary["t", "75%"]
    sns.lineplot(data=result_df, x="x", y="t", hue="Outlier", ax=ax1, ci=None, estimator=estimator, style="Outlier")
    sns.lineplot(data=result_df, x="x", y="p", hue="Outlier", ax=ax2, ci=None, estimator=estimator, style="Outlier")
    is_outlier = np.array(summary.index.get_level_values(1), dtype=bool)
    is_inlier = np.invert(is_outlier)
    print(is_outlier)
    x = np.unique(summary.index.get_level_values(0))
    ax1.fill_between(x, os_low[is_inlier], os_high[is_inlier], color=qualitative_cp[0], alpha=0.3)
    ax2.fill_between(x, p_low[is_inlier], p_high[is_inlier], color=qualitative_cp[0], alpha=0.3)
    ax1.fill_between(x, os_low[is_outlier], os_high[is_outlier], color=qualitative_cp[1], alpha=0.3)
    ax2.fill_between(x, p_low[is_outlier], p_high[is_outlier], color=qualitative_cp[1], alpha=0.3)

    ax1.set_xlabel("$\sigma_p$")
    ax2.set_xlabel("$\sigma_p$")
    ax1.set_ylabel("median $os^*_i$")
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


def plot_evaluation_frac(directory):
    sns.set_palette(sns.color_palette(["#1b9e77", "#fda968"]))
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
            results = get_probabilities(scores)
            for i, res in enumerate(results):
                result_df.append([x_axis_vals[-1], res[0], res[1], labels[i], params["shift"], params["subspace_frac"]])

    result_df = pd.DataFrame(result_df, columns=["Subspace fraction", "$os^*_i$", "$p$-value", "Outlier", "shift", "sf"])
    tested_shift = np.unique(result_df["shift"])

    fig, axs = plt.subplots(len(tested_shift), 2, sharex="col", sharey="col")

    axs[0, 0].set_xlabel("Subspace fraction")
    axs[0, 0].set_ylabel("Outlier score")

    axs[0, 1].set_xlabel("Subspace fraction")
    axs[0, 1].set_ylabel("$p$-value")

    for ax, shift in zip(axs[:, -1], tested_shift):
        selection = result_df["shift"] == shift
        sns.violinplot(data=result_df[selection], x="Subspace fraction", y="$p$-value", hue="Outlier",
                       ax=ax, cut=0, split=True, scale="width", saturation=0.8)

    for ax, shift in zip(axs[:, 0], tested_shift):
        selection = result_df["shift"] == shift
        sns.violinplot(data=result_df[selection], x="Subspace fraction", y="$os^*_i$", hue="Outlier",
                       ax=ax, cut=0, split=True, scale="width", saturation=0.8)

    for ax in axs[:, -1]:
        ax_alpha = ax.twinx()
        ax_alpha.set_ylim(ax.get_ylim())
        alpha_vals = [0.05]
        for val in alpha_vals:
            ax_alpha.axhline(val, c="black", lw=0.7, ls="dotted")
        ax_alpha.set_yticks(alpha_vals)
        ax_alpha.set_yticklabels([r"$\alpha={}$".format(val) for val in alpha_vals])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=len(handles), title="Outlier")

    pad = 5
    for ax, shift in zip(axs[:, 0], tested_shift):
        ax.annotate("$\sigma_p={}$".format(shift), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    for ax in axs.flatten():
        ax.get_legend().remove()

    for ax in axs[:-1, :].flatten():
        ax.set_xlabel("")

    plt.show()


def get_scores(directory):
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
            results = get_probabilities(scores)
            for i, res in enumerate(results):
                result_df.append([x_axis_vals[-1], res[0], res[1], labels[i], params["shift"], params["subspace_frac"]])

    result_df = pd.DataFrame(result_df, columns=["Subspace fraction", "t", "p", "Outlier", "shift", "sf"])
    tested_shift = np.unique(result_df["shift"])
    tested_sf = np.unique(result_df["sf"])

    alpha = 0.05
    scores = []
    i = 0
    for shift in tested_shift:
        for sf in tested_sf:
            i += 1
            selection = np.logical_and(result_df["shift"] == shift, result_df["sf"] == sf)
            is_outlier = np.logical_and(selection, result_df["Outlier"])
            tp = result_df[np.logical_and(is_outlier, result_df["p"] < alpha)]
            precision = len(tp) / np.sum((result_df[selection]["p"] < alpha))
            recall = len(tp) / len(result_df[is_outlier])
            f1 = 2*(precision*recall)/(precision+recall)
            scores.append([shift, sf, round(precision, 2), round(recall, 2), round(f1, 2)])

    scores = pd.DataFrame(scores, columns=["Shift", "Subspace fraction", "Precision", "Recall", "F1-Score"])\
        .sort_values(by=["Shift", "Subspace fraction"])
    print(scores.to_latex(index=False))


def get_scores_cont(directory):
    file_dict = load_all_in_dir(directory)

    x_axis_vals = []
    result_df = []

    for key in file_dict:
        params = parse_filename(key)
        sigma = float(params["shift"])
        x_axis_vals.append(sigma)
        f = file_dict[key]
        for rep in f:
            scores = rep[0]
            labels = rep[1]
            distributed_shape = (int(params["num_devices"]), int(params["num_data"]))
            scores = scores.reshape(distributed_shape)
            labels = labels.reshape(distributed_shape)
            labels = np.any(labels, axis=-1)
            results = get_probabilities(scores)
            for i, res in enumerate(results):
                result_df.append([x_axis_vals[-1], res[0], res[1], labels[i]])

    result_df = pd.DataFrame(result_df, columns=["shift", "t", "p", "Outlier"])
    tested_shift = np.unique(result_df["shift"])

    alpha = 0.05
    scores = []
    i = 0
    for shift in tested_shift:
        i += 1
        selection = result_df["shift"] == shift
        is_outlier = np.logical_and(selection, result_df["Outlier"])
        tp = result_df[np.logical_and(is_outlier, result_df["p"] < alpha)]
        precision = len(tp) / np.sum((result_df[selection]["p"] < alpha))
        recall = len(tp) / len(result_df[is_outlier])
        f1 = 2*(precision*recall)/(precision+recall)
        if i % 3 == 1:
            scores.append([shift, round(precision, 2), round(recall, 2), round(f1, 2)])

    scores = pd.DataFrame(scores, columns=["$\sigma_p$", "Precision", "Recall", "F1-Score"])\
        .sort_values(by=["$\sigma_p$"])
    print(scores.to_latex(index=False, escape=False))


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


def get_probabilities(arr):
    os_star = np.mean(arr, axis=-1)
    os_star = zscore(os_star)
    mean = np.mean(os_star)
    std = np.std(os_star)
    props = [norm(mean, std).sf(val) for val in os_star]
    results = [os_star, props]
    return np.array(results).T