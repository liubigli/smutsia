import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, jaccard_score, confusion_matrix


def compute_scores(y_true, y_pred, threshold=0.0, print_info=False):
    """Auxiliary Function that compute binary prediction scores.

    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndrarray
        Prediction. It can also be a probability vector

    threshold: float
        if greater than 0.0 this value will be used as threshold value for prediction.
    """

    if threshold > 0.0:
        y_pred = (y_pred > threshold).flatten()
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_pred, y_true)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    if print_info:
        print("Scores: \n"
              "F1 -> {},\n"
              "Recall -> {},\n"
              "Precision -> {},\n"
              "Accuracy -> {},\n"
              "Jaccard -> {}.".format(f1, recall, precision, acc, jaccard))

        print('------------------------------------------------')

    scores = {'f1': f1,
              'recall': recall,
              'precision': precision,
              'acc': acc,
              'jaccard': jaccard}

    return scores

def mat_renorm_rows(M):
    sr = M.astype(float).sum(axis=1)[:, np.newaxis]
    return np.divide(M, sr, where=sr!=0.0)

def get_confusion_matrix(y_true, y_pred, selectedId):

    conf_mat = confusion_matrix(y_true,y_pred, labels=selectedId)
    conf_mat_norm = mat_renorm_rows(conf_mat)


    return(conf_mat,conf_mat_norm)


def normalize_confusion_matrix(conf_mat):
    conf_mat_norm = np.zeros(conf_mat.shape)
    mySum = conf_mat.astype(np.float).sum(axis=1)
    myLen = mySum.shape[0]

    for i in range(myLen):
        currentSum = mySum[i]
        if(currentSum > 0):
            for j in range(myLen):
                conf_mat_norm[i,j] = conf_mat[i,j] / mySum[i]

    conf_mat_norm = np.around(conf_mat_norm, decimals=2)

    return conf_mat_norm

def condense_confusion_matrix(conf_mat, inputID, outputID):
    tmp_conf_mat = np.zeros(conf_mat.shape)
    tmp_conf_mat[:,:]=conf_mat[:,:]
    deleted_index = []
    for elem in outputID:
        if(type(elem) != int):
            first = 1
            for id in elem:
                if (not first):
                    myindex = inputID.index(id)
                    tmp_conf_mat[:,first_index]=tmp_conf_mat[:,first_index]+tmp_conf_mat[:,myindex]
                    tmp_conf_mat[first_index,:]=tmp_conf_mat[first_index,:]+tmp_conf_mat[myindex,:]
                    deleted_index.append(myindex)
                else:
                    first_index = inputID.index(id)
                first = 0

    myshape = conf_mat.shape

    new_conf_mat = np.ndarray((len(outputID),len(outputID)))
    ii = 0
    for i in range(myshape[0]):
        jj = 0
        if not (i in deleted_index):
            for j in range(myshape[0]):
                if not(j in deleted_index):
                    new_conf_mat[ii,jj]=tmp_conf_mat[i,j]
                    jj = jj + 1
            ii = ii + 1

    return(new_conf_mat)