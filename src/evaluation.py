import numpy as np

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


