import numpy as np


def kappa_m(labels, ground_truth, p_c):
    print("Num outliers in labels: {}".format(np.sum(labels > 0)))
    print("Num outliers in ground truth: {}".format(np.sum(ground_truth > 0)))
    p_0 = np.sum(labels == ground_truth)/len(labels)
    return (p_0-p_c)/(1-p_c)


def precision(labels, ground_truth, relevant_label):
    labels = labels == relevant_label
    ground_truth = ground_truth == relevant_label
    tp = labels == ground_truth
    fp = np.logical_and(labels, np.invert(ground_truth))
    return np.sum(tp)/(np.sum(tp)+np.sum(fp))


def recall(labels, ground_truth, relevant_label):
    labels = labels == relevant_label
    ground_truth = ground_truth == relevant_label
    tp = labels == ground_truth
    fn = np.logical_and(ground_truth, np.invert(labels))
    return np.sum(tp)/(np.sum(tp)+np.sum(fn))


def f1_score(labels, ground_truth, relevant_label):
    precision_ = precision(labels, ground_truth, relevant_label)
    recall_ = recall(labels, ground_truth, relevant_label)
    return 2*(precision_*recall_)/(precision_+recall_)

