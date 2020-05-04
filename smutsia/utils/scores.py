from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, jaccard_score


def compute_scores(y_true, y_pred, threshold=0.0):
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

    print("Scores: "
          "F1 -> {},"
          " Recall -> {},"
          " Precision -> {},"
          " Accuracy -> {},"
          " Jaccard -> {}".format(f1, recall, precision, acc, jaccard))

    print('------------------------------------------------')

    scores = {'f1': f1,
              'recall': recall,
              'precision': precision,
              'acc': acc,
              'jaccard': jaccard}

    return scores
