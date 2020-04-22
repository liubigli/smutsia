import numpy as np
from skimage.morphology import opening, rectangle


def retrieve_layers(points, max_layers=64):
    """
    Function that retrieve the layer for each point. We do the hypothesis that layer are stocked one after the other.
    And each layer is stocked in a clockwise (or anticlockwise) fashion.

    """
    x = points[:, 0]
    y = points[:, 1]

    # compute the theta angles
    thetas = np.arctan2(y, x)
    op_thetas = opening(thetas.reshape(-1, 1), rectangle(20, 1))
    thetas = op_thetas.flatten()
    idx = np.ones(len(thetas))

    idx_pos = idx.copy()
    idx_pos[thetas < 0] = 0

    # since each layer is stocked in a clockwise fashion each time we do a 2*pi angle we can change layer
    # so we identify each time we do a round
    changes = np.arange(len(thetas) - 1)[np.ediff1d(idx_pos) == 1]
    changes += 1  # we add one for indexes reason

    # Stocking intervals. Each element of intervals contains min index and max index of points in the same layer
    intervals = []
    for i in range(len(changes)):
        if i == 0:
            intervals.append([0, changes[i]])
        else:
            intervals.append([changes[i - 1], changes[i]])

    intervals.append([changes[-1], len(thetas)])

    # check if we have retrieved all the layers
    if len(intervals) < max_layers:
        el = intervals.pop(0)
        # in case not we are going to explore again the vector of thetas on the initial part
        thex = np.copy(thetas[:el[1]])
        # we compute again the diffs between consecutive angles and we mark each time we have a negative difference
        diffs = np.ediff1d(thex)
        idx = diffs < 0
        ints = np.arange(len(idx))[idx]
        # the negative differences mark the end of a layer and the beginning of another
        new_intervals = []
        max_new_ints = min(len(ints), max_layers - len(intervals))
        for i in range(max_new_ints):
            if i == 0:
                new_intervals.append([0, ints[i]])
            elif i == max_new_ints - 1:
                new_intervals.append([ints[i], el[1]])
            else:
                new_intervals.append([ints[i], ints[i + 1]])
        intervals = new_intervals + intervals

    # for each element in interval we assign a label that identifies the layer
    layers = np.zeros(len(thetas), dtype=np.uint8)

    for n, el in enumerate(intervals[::-1]):
        layers[el[0]:el[1]] = max_layers - (n + 1)

    return layers


def add_layers(points):
    layers = retrieve_layers(points)

    new_points = np.c_[points, layers]

    return new_points
