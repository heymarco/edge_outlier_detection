import os
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd

import numpy as np
from src.data.synthetic_data import create_raw_data, add_global_outliers, add_local_outliers, add_deviation

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.utils import load_all_in_dir

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')

qualitative_cp = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
sequential_cp = ["#000000", "#0b4131", "#168362", "#1eae83", "#3bdead", "#7ce9c8", "#bef4e4"]


# sns.set_palette(sns.cubehelix_palette(8, start=.5, rot=-.75))


def get_rank_distance(os_c, os_l, labels, beta):
    os_c = os_c.flatten()
    os_l = os_l.flatten()
    labels = labels.flatten()

    # argsort osl
    # argsort osc
    si_osc = np.argsort(os_c)
    si_osl = np.argsort(os_l)
    # Indices, die die thresholds sortieren

    # a = argsort prev_res_l
    # b = argsort prev_res_c
    sorted_indices_si_osc = np.argsort(si_osc)
    sorted_indices_si_osl = np.argsort(si_osl)
    # --> Position der indices im Array

    # diff = a - b
    diff = sorted_indices_si_osl - sorted_indices_si_osc

    beta_abs = beta * len(labels)

    dist = diff - beta_abs
    return dist


def prc_ranks(os_c, os_l, labels, pos_label, beta=0.05, dist=None):
    if dist is None:
        dist = get_rank_distance(os_c, os_l, labels, beta)
    precisions = []
    recalls = []

    val_range = np.argsort(os_l)
    sorted_labels = labels[val_range]
    val_range = np.array([i for i in range(len(val_range)) if sorted_labels[i] > 0])
    val_range = val_range / len(labels)
    print(labels.shape)

    is_pos_label = labels == pos_label

    for p in val_range:
        l_thresh = np.quantile(os_l, p)
        classification = np.zeros(labels.shape)
        classification[os_l > l_thresh] = 1
        classification[np.logical_and(os_l > l_thresh, dist <= 0)] = 2
        true_positives = np.logical_and(classification == labels, is_pos_label)
        precision = np.sum(true_positives) / np.sum(classification == pos_label)
        recall = np.sum(true_positives) / np.sum(is_pos_label)
        if not np.isnan(recall) and not np.isnan(precision):
            recalls.append(recall)
            precisions.append(precision)
    recalls.append(1.0)  # left end
    precisions.append(np.sum(labels > 0) / len(labels))  # right end
    return sorted(precisions, reverse=True), sorted(recalls)


def roc_ranks(os_c, os_l, labels, pos_label, beta=0.01, dist=None):
    if dist is None:
        dist = get_rank_distance(os_c, os_l, labels, beta)
    tprs = []
    fprs = []

    val_range = np.argsort(os_l)
    sorted_labels = labels[val_range]
    val_range = np.array([i for i in range(len(val_range)) if sorted_labels[i] > 0])
    val_range = val_range / len(labels)

    is_pos_label = labels == pos_label

    for p in val_range:
        l_thresh = np.quantile(os_l, p)
        classification = np.zeros(labels.shape)
        classification[os_l > l_thresh] = 1
        classification[np.logical_and(os_l > l_thresh, dist <= 0)] = 2
        true_positives = np.logical_and(classification == labels, is_pos_label)
        false_positives = np.logical_and(classification == pos_label, labels != pos_label)
        tpr = np.sum(true_positives) / np.sum(is_pos_label)
        fpr = np.sum(false_positives) / np.sum(labels != pos_label)
        if not np.isnan(tpr) and not np.isnan(fpr):
            tprs.append(tpr)
            fprs.append(fpr)
    # fprs.append(0.0)
    # tprs.append(np.sum(labels > 0)/len(labels))
    return sorted(tprs, reverse=True), sorted(fprs)


def kappa_ranks(os_c, os_l, labels, beta=0.01, dist=None):
    if dist is None:
        dist = get_rank_distance(os_c, os_l, labels, beta)
    kappas = []

    val_range = np.argsort(os_l)
    sorted_labels = labels[val_range]
    val_range = np.array([i for i in range(len(val_range)) if sorted_labels[i] > 0])
    val_range = val_range / len(labels)

    for p in val_range:
        l_thresh = np.quantile(os_l, p)
        classification = np.zeros(labels.shape)
        classification[os_l > l_thresh] = 1
        classification[np.logical_and(os_l > l_thresh, dist <= 0)] = 2
        accuracy = np.sum(classification == labels) / len(labels)
        majority_chance = np.sum(labels == 0) / len(labels)
        kappa_m = (accuracy - majority_chance) / (1 - majority_chance)
        kappas.append(kappa_m)
    return kappas


def evaluate_vary_ratio(from_dir):
    files = load_all_in_dir(from_dir)
    beta_range = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) / float(parse_filename(key)["frac_global"])
                           for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    for i, file in enumerate(file_keys[sorted_key_indices]):
        cached_filename_pr1 = os.path.join(from_dir, "cache", file[:-4] + "pr1" + ".npy")
        cached_filename_pr2 = os.path.join(from_dir, "cache", file[:-4] + "pr2" + ".npy")

        if not (os.path.exists(cached_filename_pr1) and os.path.exists(cached_filename_pr2)):
            result = files[file]
            final_pr1 = []
            final_pr2 = []
            for j, res in enumerate(result):
                os_c = res[0]
                os_l = res[1]
                labels = res[2].astype(int).flatten()
                os_c = os_c.flatten()
                os_l = os_l.flatten()
                results_au_pr_1 = []
                results_au_pr_2 = []
                for beta in beta_range:
                    distance = get_rank_distance(os_c, os_l, labels, beta)
                    prec1, rec1 = prc_ranks(os_c, os_l, labels, pos_label=1, beta=beta, dist=distance)
                    prec2, rec2 = prc_ranks(os_c, os_l, labels, pos_label=2, beta=beta, dist=distance)

                    au_pr_1 = auc(rec1, prec1)
                    au_pr_2 = auc(rec2, prec2)

                    results_au_pr_1.append(au_pr_1)
                    results_au_pr_2.append(au_pr_2)

                final_pr1.append(results_au_pr_1)
                final_pr2.append(results_au_pr_2)

            final_pr1 = np.mean(final_pr1, axis=0)
            final_pr2 = np.mean(final_pr2, axis=0)

            np.save(os.path.join(from_dir, "cache", file[:-4] + "pr1"), final_pr1)
            np.save(os.path.join(from_dir, "cache", file[:-4] + "pr2"), final_pr2)


def plot_vary_ratio(from_dir):
    sns.set_palette(sns.diverging_palette(162, 26, n=8))
    files = load_all_in_dir(from_dir)
    beta_range = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    fig, axs = plt.subplots(4, 2, sharex="all", sharey="all")
    pad = 5
    rows = ["AE / AE", "AE / LOF", "AE / IF", "AE / xStream"]
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel("AUPR")
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)
    for ax in axs[-1, :]:
        ax.set_xlabel(r"$\beta$")
    axs[0, 0].set_title("Local")
    axs[0, 1].set_title("Global")
    for ax in axs[:-1, 1:].flatten():
        ax.set_xlim(0.0, 0.04)

    def get_row(l_name):
        if l_name.startswith("ae"): return 0
        if l_name.startswith("lof"): return 1
        if l_name.startswith("if"): return 2
        if l_name.startswith("xstream"): return 3

    for i, file in enumerate(file_keys[sorted_key_indices]):
        cached_filename_pr1 = os.path.join(from_dir, "cache", file[:-4] + "pr1" + ".npy")
        cached_filename_pr2 = os.path.join(from_dir, "cache", file[:-4] + "pr2" + ".npy")
        params = parse_filename(file)
        row = get_row(params["l_name"])

        if os.path.exists(cached_filename_pr1) and os.path.exists(cached_filename_pr2):
            final_pr1 = np.load(cached_filename_pr1)
            final_pr2 = np.load(cached_filename_pr2)

        fl = round(float(params["frac_local"]), 3)
        fg = round(float(params["frac_global"]), 3)

        print(cached_filename_pr1)
        print(len(final_pr1))

        p1 = axs[row, 0].plot(beta_range, final_pr1)
        p2 = axs[row, 1].plot(beta_range, final_pr2,
                              label=r"$ratio = {}$".format(round(fg / fl, 1)))

        axs[row, 0].axvline(beta_range[np.argmax(final_pr1)], zorder=0, c=p1[-1].get_color(), alpha=0.9, lw=0.5)
        axs[row, 1].axvline(beta_range[np.argmax(final_pr2)], zorder=0, c=p2[-1].get_color(), alpha=0.9, lw=0.5)

    handles, labels = axs[0, -1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=2)
    plt.show()


def get_scores_vary_ratio(from_dir):
    sns.set_palette(sns.diverging_palette(162, 26, n=8))
    files = load_all_in_dir(from_dir)
    beta_range = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.013, 0.021, 0.034, 0.055, 0.089, 0.144, 0.233, 0.377]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    def get_row(l_name):
        if l_name.startswith("ae"): return 0
        if l_name.startswith("lof"): return 1
        if l_name.startswith("if"): return 2
        if l_name.startswith("xstream"): return 3

    result = []


    for i, file in enumerate(file_keys[sorted_key_indices]):
        cached_filename_pr1 = os.path.join(from_dir, "cache", file[:-4] + "pr1" + ".npy")
        cached_filename_pr2 = os.path.join(from_dir, "cache", file[:-4] + "pr2" + ".npy")
        params = parse_filename(file)
        row = get_row(params["l_name"])

        if os.path.exists(cached_filename_pr1) and os.path.exists(cached_filename_pr2):
            final_pr1 = np.load(cached_filename_pr1)
            final_pr2 = np.load(cached_filename_pr2)

        fl = round(float(params["frac_local"]), 3)
        fg = round(float(params["frac_global"]), 3)

        max_index_pr1 = np.argmax(final_pr1)
        max_index_pr2 = np.argmax(final_pr2)
        ratio = round(fg/fl, 2)
        beta_pr1 = beta_range[max_index_pr1]
        beta_pr2 = beta_range[max_index_pr2]
        aupr1 = final_pr1[max_index_pr1]
        aupr2 = final_pr2[max_index_pr2]
        result.append(["{}/{}".format(params["c_name"], params["l_name"]).upper(),
                       ratio, beta_pr2, aupr2, beta_pr1, aupr1])

    result = pd.DataFrame(result, columns=["Ensemble", "Ratio", "$\beta_{opt} (global)$", "$AUPR (global)$",
                                           "$\beta_{opt} (local)$", "$AUPR (local)$"])

    print(result.groupby("Ensemble").to_latex(index=False))


def evaluate_vary_cont(from_dir):
    files = load_all_in_dir(from_dir)
    beta_range = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.25]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    for i, file in enumerate(file_keys[sorted_key_indices]):
        cached_filename_pr1 = os.path.join(from_dir, "cache", file[:-4] + "pr1" + ".npy")
        cached_filename_pr2 = os.path.join(from_dir, "cache", file[:-4] + "pr2" + ".npy")

        if not (os.path.exists(cached_filename_pr1) and os.path.exists(cached_filename_pr2)):
            result = files[file]
            final_pr1 = []
            final_pr2 = []
            for j, res in enumerate(result):
                os_c = res[0]
                os_l = res[1]
                labels = res[2].astype(int).flatten()
                os_c = os_c.flatten()
                os_l = os_l.flatten()
                results_au_pr_1 = []
                results_au_pr_2 = []
                for beta in beta_range:
                    distance = get_rank_distance(os_c, os_l, labels, beta)
                    prec1, rec1 = prc_ranks(os_c, os_l, labels, pos_label=1, beta=beta, dist=distance)
                    prec2, rec2 = prc_ranks(os_c, os_l, labels, pos_label=2, beta=beta, dist=distance)

                    au_pr_1 = auc(rec1, prec1)
                    au_pr_2 = auc(rec2, prec2)

                    results_au_pr_1.append(au_pr_1)
                    results_au_pr_2.append(au_pr_2)

                final_pr1.append(results_au_pr_1)
                final_pr2.append(results_au_pr_2)

            final_pr1 = np.mean(final_pr1, axis=0)
            final_pr2 = np.mean(final_pr2, axis=0)

            np.save(os.path.join(from_dir, "cache", file[:-4] + "pr1"), final_pr1)
            np.save(os.path.join(from_dir, "cache", file[:-4] + "pr2"), final_pr2)


def plot_vary_cont(from_dir):
    sns.set_palette(sns.color_palette(qualitative_cp))
    files = load_all_in_dir(from_dir)
    beta_range = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.013, 0.021, 0.034, 0.055]
    line_styles = ["solid", "dotted", "dashed", "dashdot"]
    line_styles = ["solid", (0, (3, 1, 1, 1, 1, 1)), "dashed", "dashdot", (0, (3, 1, 1, 1, 1, 1)), "dotted"]

    fig, axs = plt.subplots(4, 2, sharex="all", sharey="all")
    pad = 5
    rows = ["AE / AE", "AE / LOF", "AE / IF", "AE / xStream"]
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel("AUPR")
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)
    # for ax in axs.flatten():
        # ax.set_ylim(bottom=0.6, top=1.0)
    for ax in axs[-1, :]:
        ax.set_xlabel(r"$\beta$")
    axs[0, 0].set_title("Local")
    axs[0, 1].set_title("Global")
    for ax in axs[:-1, 1:].flatten():
        ax.set_xlim(0.0, 0.03)

    def get_row(l_name):
        if l_name.startswith("ae"): return 0
        if l_name.startswith("lof"): return 1
        if l_name.startswith("if"): return 2
        if l_name.startswith("xstream"): return 3

    def get_linestyle(frac_local):
        if frac_local == "0.005": return line_styles[0]
        if frac_local == "0.01": return line_styles[1]
        if frac_local == "0.015": return line_styles[2]
        if frac_local == "0.025":
            return line_styles[3]
        else:
            return line_styles[0]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    for i, file in enumerate(file_keys[sorted_key_indices]):
        cached_filename_pr1 = os.path.join(from_dir, "cache", file[:-4] + "pr1" + ".npy")
        cached_filename_pr2 = os.path.join(from_dir, "cache", file[:-4] + "pr2" + ".npy")
        params = parse_filename(file)
        row = get_row(params["l_name"])

        final_pr1 = np.load(cached_filename_pr1)
        final_pr2 = np.load(cached_filename_pr2)

        fl = round(float(params["frac_local"]), 3)
        fg = round(float(params["frac_global"]), 3)
        print("max for local: {}".format(np.max(final_pr1)))
        print("max for global: {}".format(np.max(final_pr2)))
        lines1 = axs[row, 0].plot(beta_range, final_pr1, ls=get_linestyle(params["frac_local"]))
        lines2 = axs[row, 1].plot(beta_range, final_pr2, ls=get_linestyle(params["frac_local"]),
                         label=r"$cont={}$".format(fl + fg))

        axs[row, 0].axvline(beta_range[np.argmax(final_pr1)], color=lines1[0].get_color(), lw=0.5, alpha=0.7)
        axs[row, 1].axvline(beta_range[np.argmax(final_pr2)], color=lines2[0].get_color(), lw=0.5, alpha=0.7)



    handles, labels = axs[0, -1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=2)
    plt.show()


def get_scores_vary_cont(from_dir):
    sns.set_palette(sns.diverging_palette(162, 26, n=8))
    files = load_all_in_dir(from_dir)
    beta_range = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.013, 0.021, 0.034, 0.055]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    result = []

    for i, file in enumerate(file_keys[sorted_key_indices]):
        cached_filename_pr1 = os.path.join(from_dir, "cache", file[:-4] + "pr1" + ".npy")
        cached_filename_pr2 = os.path.join(from_dir, "cache", file[:-4] + "pr2" + ".npy")
        params = parse_filename(file)

        if os.path.exists(cached_filename_pr1) and os.path.exists(cached_filename_pr2):
            final_pr1 = np.load(cached_filename_pr1)
            final_pr2 = np.load(cached_filename_pr2)

        fl = round(float(params["frac_local"]), 3)
        fg = round(float(params["frac_global"]), 3)

        max_index_pr1 = np.argmax(final_pr1)
        max_index_pr2 = np.argmax(final_pr2)
        ratio = round(fg/fl, 2)
        beta_pr1 = beta_range[max_index_pr1]
        beta_pr2 = beta_range[max_index_pr2]
        aupr1 = round(final_pr1[max_index_pr1], 2)
        aupr2 = round(final_pr2[max_index_pr2], 2)
        result.append(["{}/{}".format(params["c_name"], params["l_name"]).upper(),
                       round(fg+fl, 2), beta_pr2, aupr2, beta_pr1, aupr1])

    result = pd.DataFrame(result, columns=["Ensemble", "$cont$", r"$\beta_{opt}$ (global)", "$AUPR$ (global)",
                                           r"$\beta_{opt}$ (local)", "$AUPR$ (local)"])

    print(result.sort_values(by=["Ensemble", "$cont$"]).to_latex(index=False, escape=False))


def evaluate_results(from_dir):
    sns.set_palette(sns.color_palette(qualitative_cp))

    def plot_roc(precision, recall, label, hline_y, axis):
        styles = ["solid", "dotted", "dashed", "dashdot", (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
        roc_auc = auc(recall, precision)
        num_lines = len(axis.get_lines())
        print(num_lines)
        axis.plot(recall, precision, label='$PR_{auc} = %0.2f)$' % roc_auc, ls=styles[num_lines])

    def create_subplots(results):

        def add_f1_iso_curve(axis):
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = axis.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                # plt.annotate('$f_1={0:0.1f}$'.format(f_score), xy=(x[45] + 0.02, 0.2), fontsize=6)
            return l

        def average_result(result):
            p_c1_arr = []
            p_l1_arr = []
            r_c1_arr = []
            r_l1_arr = []
            p_c2_arr = []
            p_l2_arr = []
            r_c2_arr = []
            r_l2_arr = []
            p_comb1_arr = []
            r_comb1_arr = []
            p_comb2_arr = []
            r_comb2_arr = []
            for rep in result:
                os_c = rep[0]
                os_l = rep[1]
                labels = rep[2].astype(int).flatten()
                os_c = os_c.flatten()
                os_l = os_l.flatten()
                p_c1, r_c1, _ = precision_recall_curve(labels, os_c, pos_label=1)
                p_l1, r_l1, _ = precision_recall_curve(labels, os_l, pos_label=1)
                p_c2, r_c2, _ = precision_recall_curve(labels, os_c, pos_label=2)
                p_l2, r_l2, _ = precision_recall_curve(labels, os_l, pos_label=2)
                beta_opt = 0.5*np.sum(labels == 2)/len(labels)
                print(beta_opt)
                p_comb1, r_comb1 = prc_ranks(os_c, os_l, labels, pos_label=1, beta=beta_opt)
                p_comb2, r_comb2 = prc_ranks(os_c, os_l, labels, pos_label=2, beta=beta_opt)
                p_c1_arr.append(p_c1)
                p_l1_arr.append(p_l1)
                p_c2_arr.append(p_c2)
                p_l2_arr.append(p_l2)
                r_c1_arr.append(r_c1)
                r_l1_arr.append(r_l1)
                r_c2_arr.append(r_c2)
                r_l2_arr.append(r_l2)
                p_comb1_arr.append(p_comb1)
                p_comb2_arr.append(p_comb2)
                r_comb1_arr.append(r_comb1)
                r_comb2_arr.append(r_comb2)

            shortest_length = np.min([len(item) for item in p_c1_arr])
            for r in range(len(p_c1_arr)):
                diff_lengths = len(p_c1_arr[r]) - shortest_length
                excluded_indices = np.random.choice(range(len(p_c1_arr[r])), diff_lengths, replace=False)
                p_c1_arr[r] = np.delete(p_c1_arr[r], excluded_indices)
                r_c1_arr[r] = np.delete(r_c1_arr[r], excluded_indices)

            shortest_length = np.min([len(item) for item in p_c2_arr])
            for r in range(len(p_c2_arr)):
                diff_lengths = len(p_c2_arr[r]) - shortest_length
                excluded_indices = np.random.choice(range(len(p_c2_arr[r])), diff_lengths, replace=False)
                p_c2_arr[r] = np.delete(p_c2_arr[r], excluded_indices)
                r_c2_arr[r] = np.delete(r_c2_arr[r], excluded_indices)

            shortest_length = np.min([len(item) for item in p_l1_arr])
            for r in range(len(p_l1_arr)):
                diff_lengths = len(p_l1_arr[r]) - shortest_length
                excluded_indices = np.random.choice(range(len(p_l1_arr[r])), diff_lengths, replace=False)
                p_l1_arr[r] = np.delete(p_l1_arr[r], excluded_indices)
                r_l1_arr[r] = np.delete(r_l1_arr[r], excluded_indices)

            shortest_length = np.min([len(item) for item in p_l2_arr])
            for r in range(len(p_l2_arr)):
                diff_lengths = len(p_l2_arr[r]) - shortest_length
                excluded_indices = np.random.choice(range(len(p_l2_arr[r])), diff_lengths, replace=False)
                p_l2_arr[r] = np.delete(p_l2_arr[r], excluded_indices)
                r_l2_arr[r] = np.delete(r_l2_arr[r], excluded_indices)

            shortest_length = np.min([len(item) for item in p_comb1_arr])
            for r in range(len(p_comb1_arr)):
                diff_lengths = len(p_comb1_arr[r]) - shortest_length
                excluded_indices = np.random.choice(range(len(p_comb1_arr[r])), diff_lengths, replace=False)
                p_comb1_arr[r] = np.delete(p_comb1_arr[r], excluded_indices)
                r_comb1_arr[r] = np.delete(r_comb1_arr[r], excluded_indices)

            shortest_length = np.min([len(item) for item in p_comb2_arr])
            for r in range(len(p_comb2_arr)):
                diff_lengths = len(p_comb2_arr[r]) - shortest_length
                excluded_indices = np.random.choice(range(len(p_comb2_arr[r])), diff_lengths, replace=False)
                p_comb2_arr[r] = np.delete(p_comb2_arr[r], excluded_indices)
                r_comb2_arr[r] = np.delete(r_comb2_arr[r], excluded_indices)

            res1 = (np.mean(p_c1_arr, axis=0), np.mean(r_c1_arr, axis=0))
            res2 = (np.mean(p_c2_arr, axis=0), np.mean(r_c2_arr, axis=0))
            res3 = (np.mean(p_l1_arr, axis=0), np.mean(r_l1_arr, axis=0))
            res4 = (np.mean(p_l2_arr, axis=0), np.mean(r_l2_arr, axis=0))
            res5 = (np.mean(p_comb1_arr, axis=0), np.mean(r_comb1_arr, axis=0))
            res6 = (np.mean(p_comb2_arr, axis=0), np.mean(r_comb2_arr, axis=0))

            return res1, res2, res3, res4, res5, res6

        mainlegend_labels = []

        fig, axs = plt.subplots(2, 3, sharey="all", sharex="all")

        for i, key in enumerate(sorted(results)):
            result = results[key]
            params = parse_filename(key)
            frac = params["subspace_frac"]
            l_name = params["l_name"]
            if not l_name.startswith("ae"):
                print(l_name)
                continue
            res_1, res_2, res_3, res_4, res_5, res_6 = average_result(result)

            mainlegend_labels.append("sf={}".format(frac))

            plot_roc(res_1[0], res_1[1], label="sf={}".format(frac), hline_y=0.005, axis=axs[0, 0])
            plot_roc(res_3[0], res_3[1], label="sf={}".format(frac), hline_y=0.005, axis=axs[0, 1])
            plot_roc(res_5[0], res_5[1], label="sf={}".format(frac), hline_y=0.005, axis=axs[0, 2])
            plot_roc(res_2[0], res_2[1], label="sf={}".format(frac), hline_y=0.005, axis=axs[1, 0])
            plot_roc(res_4[0], res_4[1], label="sf={}".format(frac), hline_y=0.005, axis=axs[1, 1])
            plot_roc(res_6[0], res_6[1], label="sf={}".format(frac), hline_y=0.005, axis=axs[1, 2])

        f1_legend = None
        for ax in axs.flatten():
            f1_legend = add_f1_iso_curve(ax)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        handles.append(f1_legend)
        mainlegend_labels.append("$f_1 = [0.2, 0.4, 0.6, 0.8]$")
        plt.figlegend(handles, mainlegend_labels, loc='lower center', frameon=False, ncol=len(handles))

        pad = 5
        rows = ["Local outliers", "Global outliers"]
        for ax in axs[-1, :]:
            ax.set_xlabel("Recall")
        for ax in axs[:, 0]:
            ax.set_ylabel("Precision")
        axs[0, 0].set_title("$C$")
        axs[0, 1].set_title("$L$")
        axs[0, 2].set_title("$C+L$")

        for ax, row in zip(axs[:, 0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)

        axs[0, 0].set_xlim([0, 1])
        axs[0, 0].set_ylim([0, 1])

    files = load_all_in_dir(from_dir)
    create_subplots(files)
    # plt.tight_layout()
    plt.show()


def plot_outlier_scores(file_dir):
    file = np.load(file_dir)
    osc = file[0][0].flatten()
    osl = file[0][1].flatten()
    labels = file[0][2].flatten()
    print(labels.shape)
    indices = np.arange(len(osc))
    plt.subplot(121)
    plt.scatter(indices[labels == 0], osc[labels == 0], alpha=0.05, label="inlier")
    plt.scatter(indices[labels == 1], osc[labels == 1], label="local outlier")
    plt.scatter(indices[labels == 2], osc[labels == 2], label="global outlier")
    plt.title("$os_c$")
    plt.subplot(122)
    plt.scatter(indices[labels == 0], osl[labels == 0], alpha=0.05, label="inlier")
    plt.scatter(indices[labels == 1], osl[labels == 1], label="local outlier")
    plt.scatter(indices[labels == 2], osl[labels == 2], label="global outlier")
    plt.title("$os_l$")
    plt.legend()
    plt.show()


def plot_2d_dataset(dev):
    palette = sns.color_palette()
    alpha = 1.0

    point_marker = "."
    marker_size = 0.25

    num_outliers = 1

    def remove_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([""])
        ax.set_yticklabels([""])

    fig, axs = plt.subplots(1, 5, sharex="all", sharey="all")
    current_ax = 0

    for ax in axs:
        remove_ticks(ax)

    data = create_raw_data(4, 200, 2)
    axs[current_ax].set_title("$(1)$")
    for i, d in enumerate(data):
        axs[current_ax].scatter(d.T[0], d.T[1], marker=point_marker, s=3 * marker_size, color=palette[i], alpha=1.0)
        axs[current_ax].autoscale(enable=False)
        axs[current_ax].autoscale()

    current_ax += 1

    data, labels_global = add_global_outliers(data, 2, frac_outlying=0.005, sigma=3)
    labels_global = np.any(labels_global, axis=-1)
    axs[current_ax].set_title("$(2)$")
    for i, d in enumerate(data):
        axs[current_ax].scatter(d[np.invert(labels_global[i])].T[0], d[np.invert(labels_global[i])].T[1],
                                marker=point_marker,
                                s=marker_size, color=palette[i], alpha=alpha)
        axs[current_ax].scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], color=palette[i], marker="x",
                                zorder=2, alpha=0.5)
        axs[current_ax].autoscale(enable=False)
        axs[current_ax].autoscale()

    current_ax += 1

    data = add_deviation(data, dev)
    axs[current_ax].set_title("$(3)$")
    for i, d in enumerate(data):
        axs[current_ax].scatter(d[np.invert(labels_global[i])].T[0], d[np.invert(labels_global[i])].T[1], marker=point_marker,
                       s=marker_size, color=palette[i], alpha=alpha)
        axs[current_ax].scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], marker="x", zorder=2, color=palette[i], alpha=0.5)
        axs[current_ax].autoscale(enable=False)
        axs[current_ax].autoscale()
    mean3 = np.mean(data, axis=(0, 1))

    current_ax += 1

    data, labels_local = add_local_outliers(data, 2, 0.005001)
    labels_local = np.any(labels_local, axis=-1)
    is_inlier = np.invert(np.logical_or(labels_local, labels_global))
    axs[current_ax].set_title("$(4)$")
    remove_ticks(ax)
    for i, d in enumerate(data):
        axs[current_ax].scatter(d[is_inlier[i]].T[0], d[is_inlier[i]].T[1], marker=point_marker,
                                label="$db_{}$".format(i + 1),
                                s=marker_size, color=palette[i], alpha=alpha)
        axs[current_ax].scatter(d[labels_local[i]].T[0], d[labels_local[i]].T[1], marker="d", zorder=3,
                                label="$o^L_{}$".format(i + 1), color=palette[i])
        axs[current_ax].scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], marker="x", zorder=2,
                                label="$o^C_{}$".format(i + 1), color=palette[i])
        axs[current_ax].autoscale(enable=False)
        axs[current_ax].autoscale()

    mean4 = np.mean(data, axis=(0, 1))

    current_ax += 1

    data = add_2d_correlation(data)
    axs[current_ax].set_title("$(5)$")
    remove_ticks(ax)
    for i, d in enumerate(data):
        axs[current_ax].scatter(d[np.invert(labels_global[i])].T[0], d[np.invert(labels_global[i])].T[1],
                                marker=point_marker,
                                s=marker_size, color=palette[i], alpha=alpha)
        axs[current_ax].scatter(d[labels_local[i]].T[0], d[labels_local[i]].T[1], marker="d", zorder=3,
                                label="$o^L_{}$".format(i + 1), color=palette[i])
        axs[current_ax].scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], marker="x", zorder=2,
                                color=palette[i],
                                label="$o^C_{}$")
        axs[current_ax].autoscale(enable=False)
        axs[current_ax].autoscale()

    handles, labels = axs[-1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=len(handles))

    mean5 = np.mean(data, axis=(0, 1))
    deviation = 20
    for i, ax in enumerate(axs):
        if i == 2:
            ax.set_xlim([mean3[0]-deviation, mean3[0]+deviation])
            ax.set_ylim([mean3[1]-deviation, mean3[1]+deviation])
        if i == 3:
            ax.set_xlim([mean4[0]-deviation, mean4[0]+deviation])
            ax.set_ylim([mean4[1]-deviation, mean4[1]+deviation])
        if i == 4:
            ax.set_xlim([mean5[0]-deviation, mean5[0]+deviation])
            ax.set_ylim([mean5[1]-deviation, mean5[1]+deviation])
        else:
            ax.set_xlim([-deviation, deviation])
            ax.set_ylim([-deviation, deviation])
    plt.show()


def add_2d_correlation(data):
    dims = data.shape[-1]
    evs = np.random.uniform(0.01, 1, size=dims)
    evs = evs / np.sum(evs) * dims
    random_corr_matrix = np.array([[1, 0.8],
                                   [0.8, 1]])
    cholesky_transform = np.linalg.cholesky(random_corr_matrix)
    for i in range(data.shape[0]):
        normal_eq_mean = cholesky_transform.dot(data[i].T)  # Generating random MVN (0, cov_matrix)
        normal_eq_mean = normal_eq_mean.transpose()
        normal_eq_mean = normal_eq_mean.transpose()  # Transposing back
        data[i] = normal_eq_mean.T
    # plt.scatter(data[0].T[0], data[0].T[1])
    # plt.show()
    return data


def parse_filename(file):
    keys = [
        "num_devices",
        "num_data",
        "dims",
        "subspace_frac",
        "frac_outlying_devices",
        "frac_local",
        "frac_global",
        "sigma_l",
        "sigma_g",
        "data_type",
        "c_name",
        "l_name"
    ]
    components = file.split("_")
    assert len(components) == len(keys)
    parsed_args = {}
    for i in range(len(keys)):
        if components[i].endswith(".npy"):
            components[i] = components[i][:-4]
        parsed_args[keys[i]] = components[i]
    return parsed_args
