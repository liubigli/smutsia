import numpy as np


def _logistic_function(x, x0, L, k):
    """
    General logistic function. It returns
        y = L / (1 + np.exp(-k * (x - x0)))

    Parameters
    ----------
    x: ndarray
        input values
    x0: float
        shift value
    L: float
        Max value of the logistic function
    k: float
        power to the exponential function

    Returns
    -------
    y: ndarray
    """
    return L / (1 + np.exp(-k * (x - x0)))


def z_nz_dist(xyz, normals, src, dst, aggr='diff'):
    """
    Custom metric used for Ground segmentation. Basically the weight between two points is the difference of the
    ratios elevation / vertical orientation.

    Parameters
    ----------
    xyz: ndarray
        input points

    normals: ndarray
        input normals

    src: ndarray
        array of source ids

    dst: ndarray
        array of destination ids

    aggr: optional {'diff', 'max'}
        if 'diff' take diff between ratios
        if 'max' take max between ratios

    Returns
    -------
    weights: ndarray
        array of weights
    """
    z = xyz[:, 2] - xyz[:, 2].min()
    nz = np.abs(normals[:, 2])

    # apply logistic function to normals
    log_nz = _logistic_function(nz, x0=0.7, L=1, k=32)
    if aggr == 'max':
        weights = np.c_[(z[src] / log_nz[src]), (z[dst] / log_nz[dst])].max(axis=1)
    elif aggr == 'min':
        weights = np.c_[(z[src] / log_nz[src]), (z[dst] / log_nz[dst])].min(axis=1)
    else:
        weights = np.abs((z[src] / log_nz[src]) - (z[dst] / log_nz[dst]))

    return weights
