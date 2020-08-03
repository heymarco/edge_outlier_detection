import os
import numpy as np

from sklearn.metrics import auc, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


def prc_combined(os_c, os_l, labels, pos_label):
    os_c = os_c.flatten()
    os_l = os_l.flatten()
    labels = labels.flatten()
    recalls = []
    precisions = []

    val_range = np.argsort(os_l)
    sorted_labels = labels[val_range]
    val_range = np.array([i for i in range(len(val_range)) if sorted_labels[i] > 0])
    val_range = val_range / len(labels)

    for p in val_range:
        print(p)
        l_thresh = np.quantile(os_l, p)
        c_thresh = np.quantile(os_c, p)
        classification = np.zeros(labels.shape)
        classification[os_l >= l_thresh] = 1
        classification[np.logical_and(os_l >= l_thresh, os_c >= c_thresh)] = 2
        true_positives = np.logical_and(classification == labels, labels == pos_label)
        precision = np.sum(true_positives) / np.sum(classification == pos_label)
        recall = np.sum(true_positives) / np.sum(labels == pos_label)
        if not np.isnan(recall) and not np.isnan(precision):
            recalls.append(recall)
            precisions.append(precision)
    recalls.append(1.0)
    precisions.append(np.sum(labels > 0)/len(labels))
    return sorted(precisions, reverse=True), sorted(recalls)


def prc_ranks(os_c, os_l, labels, pos_label):
    os_c = os_c.flatten()
    os_l = os_l.flatten()
    labels = labels.flatten()
    recalls = []
    precisions = []

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
    #

    beta = 0.1
    beta_abs = beta*len(labels)

    dist = diff - beta_abs

    # if beta_abs = 0 --> local outlier

    val_range = np.argsort(os_l)
    sorted_labels = labels[val_range]
    val_range = np.array([i for i in range(len(val_range)) if sorted_labels[i] > 0])
    val_range = val_range / len(labels)

    for p in val_range:
        l_thresh = np.quantile(os_l, p)
        classification = np.zeros(labels.shape)
        classification[os_l > l_thresh] = 1
        classification[np.logical_and(os_l > l_thresh, dist <= 0)] = 2
        true_positives = np.logical_and(classification == labels, labels == pos_label)
        precision = np.sum(true_positives) / np.sum(classification == pos_label)
        recall = np.sum(true_positives) / np.sum(labels == pos_label)
        if not np.isnan(recall) and not np.isnan(precision):
            recalls.append(recall)
            precisions.append(precision)
    recalls.append(1.0)
    precisions.append(np.sum(labels > 0)/len(labels))
    return sorted(precisions, reverse=True), sorted(recalls)


def plot_os_star_hist(from_dir):
    def load_all_in_dir(dir):
        all_files = {}
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".npy"):
                    filepath = os.path.join(from_dir, file)
                    result_file = np.load(filepath)
                    all_files[file] = result_file
        return all_files

    def parse_filename(file):
        components = file.split("_")
        c_name = components[-2]
        l_name = components[-1]
        num_devices = components[0]
        frac = components[3]
        return num_devices, frac, c_name, l_name

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


def evaluate_results(from_dir):
    def load_all_in_dir(dir):
        all_files = {}
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".npy"):
                    filepath = os.path.join(from_dir, file)
                    result_file = np.load(filepath)
                    all_files[file] = result_file
        return all_files

    def plot_roc(precision, recall, label, hline_y):
        roc_auc = auc(recall, precision)
        plt.plot(recall, precision, label='$PR_{auc} = %0.2f)$' % roc_auc)
        # plt.plot(recall, precision, label=label)
        plt.xlim((0, 1))
        # plt.ylim((0, 1))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axhline(hline_y, color='navy', linestyle='--')

    def create_subplots(results):
        sns.set_palette(sns.color_palette("PuBuGn_d"))
        def parse_filename(file):
            components = file.split("_")
            c_name = components[-2]
            l_name = components[-1]
            num_devices = components[0]
            frac = components[3] if len(components) > 3 else None
            return num_devices, frac, c_name, l_name

        def add_f1_iso_curve():
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('$f_1={0:0.1f}$'.format(f_score), xy=(0.8, y[45] + 0.02), fontsize=6)

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
            rep = result[0]
            # for rep in result:
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
                add_f1_iso_curve()

            ax2 = plt.subplot(2, 3, 4, sharey=ax1, sharex=ax1)
            plot_roc(res_2[0], res_2[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", frameon=False, ncol=2)
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
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", frameon=False, ncol=2)
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
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", frameon=False, ncol=2)
            plt.title('$L + C$ identify GO')

            if i == 0:
                plt.ylim((0, 1))
                add_f1_iso_curve()

        handles, labels = ax1.get_legend_handles_labels()
        plt.figlegend(handles, mainlegend_labels, loc='lower center', frameon=False, ncol=len(handles))

    files = load_all_in_dir(from_dir)
    create_subplots(files)
    # plt.tight_layout()
    plt.show()