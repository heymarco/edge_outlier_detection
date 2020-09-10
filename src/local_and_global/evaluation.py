import os
from sklearn.metrics import precision_recall_curve, auc

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

qualitative_cp = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
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


def prc_ranks(os_c, os_l, labels, pos_label, beta=0.01, dist=None):
    if dist is None:
        dist = get_rank_distance(os_c, os_l, labels, beta)
    precisions = []
    recalls = []

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
        precision = np.sum(true_positives) / np.sum(classification == pos_label)
        recall = np.sum(true_positives) / np.sum(is_pos_label)
        if not np.isnan(recall) and not np.isnan(precision):
            recalls.append(recall)
            precisions.append(precision)
    recalls.append(1.0)
    precisions.append(np.sum(labels > 0)/len(labels))
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
    beta_range = [0.0, 0.005, 0.1, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.25]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    for i, file in enumerate(file_keys[sorted_key_indices]):
        cached_filename_pr1 = os.path.join(from_dir, "cache", file[:-4] + "pr1" + ".npy")
        cached_filename_pr2 = os.path.join(from_dir, "cache", file[:-4] + "pr2" + ".npy")

        if not (os.path.exists(cached_filename_pr1) and os.path.exists(cached_filename_pr2)):
            result = files[file]
            print(file)
            print(result.shape)
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
    files = load_all_in_dir(from_dir)
    beta_range = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.25]

    file_keys = np.array(list(files.keys()))
    contamination_fracs = [float(parse_filename(key)["frac_local"]) for key in files]
    sorted_key_indices = np.argsort(contamination_fracs)

    fig, axs = plt.subplots(4, 2)
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
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs[:-1, 0]:
        ax.set_xticklabels([])
        ax.set_xticks([])
    for ax in axs[-1, 1:]:
        ax.set_yticklabels([])
        ax.set_yticks([])

    def get_row(l_name):
        if l_name.startswith("ae"): return 0
        if l_name.startswith("lof"): return 1
        if l_name.startswith("if"): return 2
        if l_name.startswith("xstream"): return 3

    def get_linestyle(frac_local):
        line_styles = ["solid", "dotted", "dashed", "dashdot"]
        if frac_local == "0.005": return line_styles[0]
        if frac_local == "0.025": return line_styles[1]
        if frac_local == "0.05":
            return line_styles[2]
        else:
            return line_styles[0]

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
        p1 = axs[row, 0].plot(beta_range, final_pr1)
        p2 = axs[row, 1].plot(beta_range, final_pr2, label=r"$|\vec{o}_g|={}, |\vec{o}_l|={}$".format(round(fg/fl, 1)))
        axs[row, 0].axvline(beta_range[np.argmax(final_pr1)], zorder=0, c=p1[-1].get_color(), ls="dotted")
        axs[row, 1].axvline(beta_range[np.argmax(final_pr2)], zorder=0, c=p2[-1].get_color(), ls="dotted")

    handles, labels = axs[0, -1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=2)
    plt.show()


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
            print(result.shape)
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
    beta_range = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.25]
    line_styles = ["solid", "dotted", "dashed", "dashdot"]

    fig, axs = plt.subplots(4, 2, sharex="all")
    pad = 5
    rows = ["AE / AE", "AE / LOF", "AE / IF", "AE / xStream"]
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel("AUPR")
        ax.set_ylim(top=1)
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)
    for ax in axs[-1, :]:
        ax.set_xlabel(r"$\beta$")
    axs[0, 0].set_title("Local")
    axs[0, 1].set_title("Global")
    for ax in axs[:-1, 1:].flatten():
        ax.set_xlim(0.0, 0.2)
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # for ax in axs[:-1, 0]:
    #     ax.set_xticklabels([])
    #     ax.set_xticks([])
    # for ax in axs[-1, 1:]:
    #     ax.set_yticklabels([])
    #     ax.set_yticks([])
    # for ax in axs.flatten():
    #     ax.axvline(0.005, ls=line_styles[0], color="gray")
    #     ax.axvline(0.015, ls=line_styles[1], color="gray")
    #     ax.axvline(0.01, ls=line_styles[2], color="gray")
    #     ax.axvline(0.025, ls=line_styles[3], color="gray")

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
        print(final_pr1)
        print(final_pr2)
        axs[row, 0].plot(beta_range, final_pr1, ls=get_linestyle(params["frac_local"]))
        axs[row, 1].plot(beta_range, final_pr2, ls=get_linestyle(params["frac_local"]),
                         label=r"$|\vec{o}_g|={}, |\vec{o}_l|={}$".format(fl, fg))

    handles, labels = axs[0, -1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=2)
    plt.show()


def evaluate_results(from_dir):

    def plot_roc(precision, recall, label, hline_y):
        roc_auc = auc(recall, precision)
        plt.plot(recall, precision, label='$PR_{auc} = %0.2f)$' % roc_auc)
        plt.xlim((0, 1))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.axhline(hline_y, color='navy', linestyle='--')

    def create_subplots(results):
        sns.set_palette(sns.color_palette("PuBuGn_d"))

        def add_f1_iso_curve():
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
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
                rep = result[0]
                os_c = rep[0]
                os_l = rep[1]
                labels = rep[2].astype(int).flatten()
                os_c = os_c.flatten()
                os_l = os_l.flatten()
                p_c1, r_c1, _ = precision_recall_curve(labels, os_c, pos_label=1)
                p_l1, r_l1, _ = precision_recall_curve(labels, os_l, pos_label=1)
                p_c2, r_c2, _ = precision_recall_curve(labels, os_c, pos_label=2)
                p_l2, r_l2, _ = precision_recall_curve(labels, os_l, pos_label=2)
                p_comb1, r_comb1 = prc_ranks(os_c, os_l, labels, pos_label=1)
                p_comb2, r_comb2 = prc_ranks(os_c, os_l, labels, pos_label=2)
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

        for i, key in enumerate(sorted(results)):
            result = results[key]
            _, frac, _, _ = parse_filename(key)
            res_1, res_2, res_3, res_4, res_5, res_6 = average_result(result)

            mainlegend_labels.append("sf={}".format(frac))

            ax1 = plt.subplot(2, 3, 1)
            ax1.get_xaxis().set_visible(False)
            plot_roc(res_1[0], res_1[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, 0), loc="upper center", frameon=False, ncol=2)
            plt.title('$C$ identifies LO')

            if i == 0:
                plt.ylim((0, 1))
                f1_legend = add_f1_iso_curve()

            ax2 = plt.subplot(2, 3, 4, sharey=ax1, sharex=ax1)
            plot_roc(res_2[0], res_2[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", frameon=False, ncol=2)
            plt.title('$C$ identifies GO')

            if i == 0:
                plt.ylim((0, 1))
                add_f1_iso_curve()

            ax3 = plt.subplot(2, 3, 2, sharex=ax1, sharey=ax1)
            ax3.get_yaxis().set_visible(False)
            ax3.get_xaxis().set_visible(False)
            plot_roc(res_3[0], res_3[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, 0), loc="upper center", frameon=False, ncol=2)
            plt.title('$L$ identifies LO')

            if i == 0:
                plt.ylim((0, 1))
                add_f1_iso_curve()

            ax4 = plt.subplot(2, 3, 5, sharey=ax1, sharex=ax1)
            ax4.get_yaxis().set_visible(False)
            plot_roc(res_4[0], res_4[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", frameon=False, ncol=2)
            plt.title('$L$ identifies GO')

            if i == 0:
                plt.ylim((0, 1))
                add_f1_iso_curve()

            ax5 = plt.subplot(2, 3, 3, sharey=ax1, sharex=ax1)
            ax5.get_yaxis().set_visible(False)
            ax5.get_xaxis().set_visible(False)
            plot_roc(res_5[0], res_5[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, 0), loc="upper center", frameon=False, ncol=2)
            plt.title('$L + C$ identify LO')

            if i == 0:
                plt.ylim((0, 1))
                add_f1_iso_curve()

            ax6 = plt.subplot(2, 3, 6, sharey=ax1, sharex=ax1)
            ax6.get_yaxis().set_visible(False)
            plot_roc(res_6[0], res_6[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", frameon=False, ncol=2)
            plt.title('$L + C$ identify GO')

            if i == 0:
                plt.ylim((0, 1))
                add_f1_iso_curve()

        handles, labels = ax1.get_legend_handles_labels()
        handles.append(f1_legend)
        mainlegend_labels.append("$f_1 = [0.2, 0.4, 0.6, 0.8]$")
        plt.figlegend(handles, mainlegend_labels, loc='lower center', frameon=False, ncol=len(handles))

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
    alpha = 0.3
    def remove_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([""])
        ax.set_yticklabels([""])

    data = create_raw_data(2, 50, 2)
    ax = plt.subplot(151)
    plt.title("$(1)$")
    remove_ticks(ax)
    for i, d in enumerate(data):
        plt.scatter(d.T[0], d.T[1], marker=".", color=palette[i], alpha=1.0)

    data, labels_global = add_global_outliers(data, 2, frac_outlying=0.02)
    labels_global = np.any(labels_global, axis=-1)
    ax = plt.subplot(152)
    plt.title("$(2)$")
    remove_ticks(ax)
    for i, d in enumerate(data):
        plt.scatter(d[np.invert(labels_global[i])].T[0], d[np.invert(labels_global[i])].T[1], marker=".", color=palette[i], alpha=alpha)
        plt.scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], color=palette[i], marker="x", zorder=2)

    data = add_deviation(data, dev, 0)
    ax = plt.subplot(153)
    plt.title("$(3)$")
    remove_ticks(ax)
    for i, d in enumerate(data):
        plt.scatter(d[np.invert(labels_global[i])].T[0], d[np.invert(labels_global[i])].T[1], marker=".", color=palette[i], alpha=alpha)
        plt.scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], marker="x", zorder=2, color=palette[i])

    data = add_2d_correlation(data)
    ax = plt.subplot(154)
    plt.title("$(4)$")
    remove_ticks(ax)
    for i, d in enumerate(data):
        plt.scatter(d[np.invert(labels_global[i])].T[0], d[np.invert(labels_global[i])].T[1], marker=".", color=palette[i], alpha=alpha)
        plt.scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], marker="x", zorder=2, color=palette[i])

    data, labels_local = add_local_outliers(data, 2, 0.02)
    labels_local = np.any(labels_local, axis=-1)
    is_inlier = np.invert(np.logical_or(labels_local, labels_global))
    ax = plt.subplot(155)
    plt.title("$(5)$")
    remove_ticks(ax)
    for i, d in enumerate(data):
        plt.scatter(d[is_inlier[i]].T[0], d[is_inlier[i]].T[1], marker=".", label="$db_{}$".format(i+1), color=palette[i], alpha=alpha)
        plt.scatter(d[labels_local[i]].T[0], d[labels_local[i]].T[1], marker="d", zorder=3, label="$o^L_{}$".format(i+1), color=palette[i])
        plt.scatter(d[labels_global[i]].T[0], d[labels_global[i]].T[1], marker="x", zorder=2, label="$o^C_{}$".format(i+1), color=palette[i])

    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=len(handles))
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
        parsed_args[keys[i]] = components[i]
    return parsed_args