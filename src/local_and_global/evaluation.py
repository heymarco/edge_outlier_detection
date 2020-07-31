import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from src.metrics import kappa_m, f1_score
from xstream.python.Chains import Chains
from src.models import create_model, create_models, train_federated
from src.local_outliers.evaluation import retrieve_labels
from src.utils import color_palette

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rc('font', family='serif')


def create_ensembles(shape, l_name, contamination=0.01):
    num_clients = shape[0]
    c = create_models(num_clients, shape[-1], compression_factor=0.4)
    l = None
    if l_name == "lof1":
        l = [LocalOutlierFactor(n_neighbors=1, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof2":
        l = [LocalOutlierFactor(n_neighbors=2, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof4":
        l = [LocalOutlierFactor(n_neighbors=4, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof8":
        l = [LocalOutlierFactor(n_neighbors=8, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof10":
        l = [LocalOutlierFactor(n_neighbors=10, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof20":
        l = [LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof16":
        l = [LocalOutlierFactor(n_neighbors=16, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof32":
        l = [LocalOutlierFactor(n_neighbors=32, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof64":
        l = [LocalOutlierFactor(n_neighbors=64, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "lof100":
        l = [LocalOutlierFactor(n_neighbors=100, contamination=contamination, novelty=True) for _ in range(num_clients)]
    if l_name == "xstream":
        l = [Chains(k=50, nchains=50, depth=10) for _ in range(num_clients)]
    if l_name == "ae":
        l = [create_model(shape[-1], compression_factor=0.4) for _ in range(num_clients)]
    if l_name == "if":
        l = [IsolationForest(contamination=contamination) for _ in range(num_clients)]
    if not l:
        raise KeyError("No valid local outlier detector name provided.")
    return np.array(c), np.array(l)


def train_ensembles(data, ensembles, l_name, global_epochs=10):
    collab_detectors = ensembles[0]
    local_detectors = ensembles[1]

    # federated training
    for _ in range(global_epochs):
        collab_detectors = train_federated(models=collab_detectors, data=data, epochs=1, batch_size=32,
                                           frac_available=1.0)

    # global scores
    predicted = np.array([model.predict(data[i]) for i, model in enumerate(collab_detectors)])
    diff = predicted - data
    dist = np.linalg.norm(diff, axis=-1)
    global_scores = dist.flatten()

    print("Fitting {}".format(l_name))
    # local training
    if l_name.startswith("lof") or l_name == "if" or l_name == "xstream":
        [l.fit(data[i]) for i, l in enumerate(local_detectors)]
    if l_name == "ae":
        [l.fit(data[i], data[i],
               batch_size=32, epochs=global_epochs) for i, l in enumerate(local_detectors)]

    # local scores
    if l_name.startswith("lof"):
        local_scores = - np.array([model.negative_outlier_factor_ for i, model in enumerate(local_detectors)])
    if l_name == "xstream":
        local_scores = np.array([-model.score(data[i]) for i, model in enumerate(local_detectors)])
    if l_name == "if":
        local_scores = -np.array([model.score_samples(data[i]) for i, model in enumerate(local_detectors)])
    if l_name == "ae":
        predicted = np.array([model.predict(data[i]) for i, model in enumerate(local_detectors)])
        diff = predicted - data
        dist = np.linalg.norm(diff, axis=-1)
        local_scores = np.reshape(dist, newshape=(data.shape[0], data.shape[1]))

    return global_scores, local_scores


def roc_global_combined(os_c, os_l, labels):
    pos_label = 2
    os_c = os_c.flatten()
    os_l = os_l.flatten()
    labels = labels.flatten()
    tpr = []
    fpr = []
    percentages = np.arange(1001) / 10
    for thresh in percentages:
        l_thresh = np.percentile(os_l, thresh)
        c_thresh = np.percentile(os_c, thresh)
        classification = np.zeros(labels.shape)
        classification[np.logical_and(os_l >= l_thresh, os_c >= c_thresh)] = pos_label
        tp = np.logical_and(labels == pos_label, classification == pos_label)
        fp = np.logical_and(labels != pos_label, classification == pos_label)
        positives = np.sum(labels == pos_label)
        negatives = np.sum(labels != pos_label)
        tp = np.sum(tp) / positives
        fp = np.sum(fp) / negatives
        tpr.append(tp)
        fpr.append(fp)
    return tpr, fpr


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


def prc_local(os_l, labels):
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
        classification = np.zeros(labels.shape)
        classification[os_l >= l_thresh] = 1
        true_positives = np.logical_and(classification == labels, labels > 0)
        precision = np.sum(true_positives) / np.sum(classification > 0)
        recall = np.sum(true_positives) / np.sum(labels > 0)
        if not np.isnan(recall) and not np.isnan(precision):
            recalls.append(recall)
            precisions.append(precision)
    recalls.append(1.0)
    precisions.append(np.sum(labels > 0)/len(labels))
    return sorted(precisions, reverse=True), sorted(recalls)


def prc_global(os_c, labels):
    os_c = os_c.flatten()
    labels = labels.flatten()
    recalls = []
    precisions = []
    percentages = np.arange(0, 10010, 10) / 10000.0
    for p in percentages:
        l_thresh = np.min(os_c) + (np.max(os_c)-np.min(os_c)) * p
        # l_thresh = np.percentile(os_l, p)
        # c_thresh = np.percentile(os_c, 100-p)
        classification = np.zeros(labels.shape)
        classification[os_c >= l_thresh] = 2
        true_positives = np.logical_and(classification == labels, labels > 0)
        precision = np.sum(true_positives) / np.sum(classification > 0)
        recall = np.sum(true_positives) / np.sum(labels > 0)
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
            frac = components[3]
            return num_devices, frac, c_name, l_name

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
                os_c = os_c.flatten()  # (os_c / np.mean(os_c, axis=-1, keepdims=True)).flatten()
                os_l = os_l.flatten()  # (os_l / np.mean(os_l, axis=-1, keepdims=True)).flatten()
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

        for key in sorted(results):
            result = results[key]
            _, frac, _, _ = parse_filename(key)
            res_1, res_2, res_3, res_4, res_5, res_6 = average_result(result)

            mainlegend_labels.append("sf={}".format(frac))

            ax1 = plt.subplot(2, 3, 1)
            ax1.get_xaxis().set_visible(False)
            plot_roc(res_1[0], res_1[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, 0), loc="upper center", frameon=False, ncol=2)
            plt.title('$C$ identifies LO')

            ax2 = plt.subplot(2, 3, 4, sharey=ax1, sharex=ax1)
            plot_roc(res_2[0], res_2[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", frameon=False, ncol=2)
            plt.title('$C$ identifies GO')

            ax3 = plt.subplot(2, 3, 2, sharex=ax1, sharey=ax1)
            ax3.get_yaxis().set_visible(False)
            ax3.get_xaxis().set_visible(False)
            plot_roc(res_3[0], res_3[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, 0), loc="upper center", frameon=False, ncol=2)
            plt.title('$L$ identifies LO')

            ax4 = plt.subplot(2, 3, 5, sharey=ax1, sharex=ax1)
            ax4.get_yaxis().set_visible(False)
            plot_roc(res_4[0], res_4[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", frameon=False, ncol=2)
            plt.title('$L$ identifies GO')

            ax5 = plt.subplot(2, 3, 3, sharey=ax1, sharex=ax1)
            ax5.get_yaxis().set_visible(False)
            ax5.get_xaxis().set_visible(False)
            plot_roc(res_5[0], res_5[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, 0), loc="upper center", frameon=False, ncol=2)
            plt.title('$L + C$ identify LO')

            ax6 = plt.subplot(2, 3, 6, sharey=ax1, sharex=ax1)
            ax6.get_yaxis().set_visible(False)
            plot_roc(res_6[0], res_6[1], label="sf={}".format(frac), hline_y=0.005)
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", frameon=False, ncol=2)
            plt.title('$L + C$ identify GO')

        handles, labels = ax1.get_legend_handles_labels()
        plt.figlegend(handles, mainlegend_labels, loc='lower center', frameon=False, ncol=len(handles))

    files = load_all_in_dir(from_dir)
    create_subplots(files)
    # plt.tight_layout()
    plt.show()


def classify(result_global, result_local, contamination=0.01):
    assert len(result_local) == len(result_global)
    labels = []
    for i in range(len(result_local)):
        labels_global = retrieve_labels(result_global[i], contamination).flatten()
        labels_local = retrieve_labels(result_local[i], contamination).flatten()
        # remove candidates for abnormal data partitions
        labels_global[np.logical_and(labels_global, np.invert(labels_local))] = 0
        print("Number of global outliers: {}".format(np.sum(np.logical_and(labels_global, labels_local))))
        print("Number of local outliers: {}".format(np.sum(np.logical_and(labels_local, np.invert(labels_global)))))
        classification = np.empty(shape=labels_global.shape)
        classification.fill(0)
        is_global_outlier = np.logical_and(labels_global, labels_local)
        classification[is_global_outlier] = 2
        is_local_outlier = np.logical_and(labels_local, np.invert(is_global_outlier))
        classification[is_local_outlier] = 1
        labels.append(classification)
    return np.array(labels)


def evaluate(labels, ground_truth, contamination):
    ground_truth = ground_truth.flatten()
    print("Sum of correct: {}".format(np.sum(np.logical_and(labels > 0, ground_truth > 0))))
    kappa = []
    f1_local = []
    f1_global = []
    for lbs in labels:
        lbs = lbs.flatten()
        kappa.append(kappa_m(lbs, ground_truth, 1 - contamination))
        f1_local.append(f1_score(lbs, ground_truth, relevant_label=1))
        f1_global.append(f1_score(lbs, ground_truth, relevant_label=2))
    return np.nanmean(kappa), np.nanmean(f1_global), np.nanmean(f1_local)


def plot_result():
    # read from dir
    directory = os.path.join(os.getcwd(), "results", "numpy", "local_and_global")

    def parse_filename(file):
        components = file.split("_")
        c_name = components[-2]
        l_name = components[-1]
        num_devices = components[0]
        frac = components[3]
        return num_devices, frac, c_name, l_name

    names = {
        "ae": "AE",
        "if": "IF",
        "xstream": "xStream",
        "lof8": "LOF"
    }
    res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                num_devices, frac, c_name, l_name = parse_filename(file[:-4])
                result = np.load(os.path.join(directory, file))
                l = l_name
                if l_name not in ["lof1", "lof2", "lof4", "lof20", "lof10"]:
                    c = names[c_name]
                    l = names[l_name]
                    new_res = [float(num_devices),
                               float(frac), "{}/{}".format(c, l),
                               result[0],
                               "$\kappa_m$"]
                    res.append(new_res)
                    new_res = [float(num_devices),
                               float(frac), "{}/{}".format(c, l),
                               result[1],
                               "f1$_{global}$"]
                    res.append(new_res)
                    new_res = [float(num_devices),
                               float(frac), "{}/{}".format(c, l),
                               result[2],
                               "f1$_{local}$"]
                    res.append(new_res)

    d = {'color': color_palette, "marker": ["o", "*", "v", "x"]}
    df = pd.DataFrame(res, columns=["\# Devices", "Subspace fraction", "Ensemble", "Value", "Measure"])
    df = df.sort_values(by=["Ensemble", "Measure", "Subspace fraction"])
    mpl.rc('font', **{"size": 14})
    g = sns.FacetGrid(df, col="Measure", hue="Ensemble", hue_kws=d)
    g.map(plt.plot, "Subspace fraction", "Value").add_legend()

    plt.show()
