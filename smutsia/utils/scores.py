import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, jaccard_score, confusion_matrix
from smutsia.utils.arrays import cartesian_product


def compute_scores(y_true, y_pred, threshold=0.0, print_info=False, sample_name=None):
    """Auxiliary Function that compute binary prediction scores.

    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndrarray
        Prediction. It can also be a probability vector

    threshold: float
        if greater than 0.0 this value will be used as threshold value for prediction.

    print_info: bool
        if true program flushes out obtained scores

    sample_name: str
        print out name of sample in flush out
    """

    if threshold > 0.0:
        y_pred = (y_pred > threshold).flatten()
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    if print_info:
        print("Scores {}: \n"
              "F1 -> {},\n"
              "Recall -> {},\n"
              "Precision -> {},\n"
              "Accuracy -> {},\n"
              "Jaccard -> {}.".format(sample_name, f1, recall, precision, acc, jaccard))

        print('------------------------------------------------')

    scores = {'f1': f1,
              'recall': recall,
              'precision': precision,
              'acc': acc,
              'jaccard': jaccard}

    return scores


def mat_renorm_rows(M):
    sr = M.astype(float).sum(axis=1)[:, np.newaxis]
    return np.divide(M, sr, where=sr != 0.0)


def get_confusion_matrix(y_true, y_pred, selectedId):
    conf_mat = confusion_matrix(y_true, y_pred, labels=selectedId)
    conf_mat_norm = mat_renorm_rows(conf_mat)
    return conf_mat, conf_mat_norm


def normalize_confusion_matrix(conf_mat):
    conf_mat_norm = np.zeros(conf_mat.shape)
    mySum = conf_mat.astype(np.float).sum(axis=1)
    myLen = mySum.shape[0]

    for i in range(myLen):
        currentSum = mySum[i]
        if currentSum > 0:
            for j in range(myLen):
                conf_mat_norm[i, j] = conf_mat[i, j] / mySum[i]

    conf_mat_norm = np.around(conf_mat_norm, decimals=2)

    return conf_mat_norm


def condense_confusion_matrix(conf_mat, input_labels, condense_list):
    condensed_mat = np.zeros((len(condense_list), len(condense_list)))
    ranges = np.arange(len(condense_list))
    ext_to_condense = []
    cond = []
    for condense in condense_list:
        cond_i = []
        for el in condense:
            cond_i.append(input_labels.index(el))
        cond.append(cond_i)

    for i in ranges:
        for j in ranges:
            vec0, vec1 = np.array(cond[i]), np.array(cond[j])
            c = cartesian_product([vec0, vec1])
            ii = np.repeat(i, len(c))
            jj = np.repeat(j, len(c))
            ext_to_condense.append(np.c_[ii, jj, c])

    ext_to_condense = np.concatenate(ext_to_condense, axis=0)
    for coords in ext_to_condense:
        condensed_mat[coords[0], coords[1]] += conf_mat[coords[2], coords[3]]

    assert condensed_mat.sum() == conf_mat.sum()

    return condensed_mat
