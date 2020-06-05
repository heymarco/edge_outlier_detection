import numpy as np


def kappa_m(labels, ground_truth, p_c):
    p_0 = np.sum(labels == ground_truth)/len(labels.flatten())
    return (p_0-p_c)/(1-p_c)


def precision(labels, ground_truth, relevant_label):
    labels = labels == relevant_label
    ground_truth = ground_truth == relevant_label
    tp = np.logical_and(labels, ground_truth)
    fp = np.logical_and(labels, np.invert(ground_truth))
    return np.sum(tp)/(np.sum(tp)+np.sum(fp))


def recall(labels, ground_truth, relevant_label):
    labels = labels == relevant_label
    ground_truth = ground_truth == relevant_label
    tp = np.logical_and(labels, ground_truth)
    sum_outliers = np.sum(ground_truth)
    return np.sum(tp)/sum_outliers


def f1_score(labels, ground_truth, relevant_label):
    precision_ = precision(labels, ground_truth, relevant_label)
    recall_ = recall(labels, ground_truth, relevant_label)
    return 2*(precision_*recall_)/(precision_+recall_)

