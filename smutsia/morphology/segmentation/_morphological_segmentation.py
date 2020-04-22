from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from smutsia.utils import shuffle_labels


def threshold_edges(graph, threshold, metric='distance'):
    """
    Functions that thresholds edges from a graph according to the value of the parameter threshold

    Parameters
    ----------
    graph: NxN csr_matrix
        csr matrix representing adj matrix of a graph

    threshold: float
        value used to remove all the edges bigger than threshold

    metric: {'distance, 'similarity'} optional
        'distance' means that the weights in the graph represent a distance between the node.
            In this case we remove edges whose weights are above the threshold.
        'similarity' means that the weights in the graph represent a similarity between the node.
            In this case we remove edges whose weights are under the threshold.

        default: 'distance'

    Returns
    -------
    graph: NxN csr_matrix
        csr matrix of the remaining edges in the graph
    """

    if metric == 'distance':
        return graph - graph.multiply(graph >= threshold)

    elif metric == 'similarity':
        return graph.multiply(graph >= threshold)
    else:
        raise ValueError("Accepted values for metric parameter are only 'distance' or 'similarity'")


def quasi_flat_zones(graph, threshold, debug_info=False, return_ncc=False):
    """
    Method that given a graph and a given threshold lambda it returns all the lambda quasi flat zones of the graph

    Parameters
    ----------
    graph: N, N csr_matrix
        graph represented as adjacency matrix

    threshold: float
        value used to compute quasi flat zones

    debug_info: bool (default: False)
        print further information on the obtained connected components

    return_ncc: bool (default: False)
        if True it returns also the number of lambda flat zones obtained

    Returns
    -------
    labels: N numpy.ndarray
        vector that at each node of the graph associate a label corresponding to its lambda quasi flat zone
    """

    # first of all we compute the minimum spanning tree of the graph
    mst = minimum_spanning_tree(graph)

    # second we remove all the edges heavier than threshold
    seg_graph = threshold_edges(mst, threshold)

    # the resulting segmentation is made by the connected components of the segmented graph
    ncc, labels = connected_components(seg_graph, directed=False)

    if return_ncc:
        return ncc, shuffle_labels(labels)

    if debug_info:  # printing the number of resulting connected components
        print("Number of connected components: ", ncc)

    return shuffle_labels(labels)


# def alpha_omega_constrained_connectivity(minimum_spanning_tree, pixels_values, alpha, omega):
#     """
#     Function that computes (alpha,omega)-constrained connectivity of a given image.
#     See paper of P. Soille as reference ( https://doi.org/10.1109/TPAMI.2007.70817 )
#
#     Parameters
#     ----------
#     minimum_spanning_tree: NxN csr_matrix
#         Sparse adjacent matrix of a minimum spanning tree of the graph associated to image
#
#     pixels_values: Nxd ndarray
#         Array containing pixels values. For gray-scale images d=1, for multichannel images d is equal to the number
#         of channels
#
#     alpha: float
#         Threshold value for edge weights
#
#     omega: float
#         Threshold value for max range in each connected component
#
#     Returns
#     -------
#     labels: N ndarray
#         Array that associate at each pixel of the image a label in the segmentation.
#     """
#     from SST.hierarchy._mst_based_hierarchy import hierarchy_from_mst, COL_W, COL_RNG, COL_V1, COL_V2
#     from scipy.sparse import csr_matrix
#
#     if len(pixels_values.shape) > 1:
#         pixels_values = pixels_values.flatten()
#
#     hierarchy = hierarchy_from_mst(minimum_spanning_tree, pixels_values)
#
#     alpha_omega = hierarchy[np.logical_and(hierarchy[:,COL_W] < alpha, hierarchy[:, COL_RNG] < omega)]
#
#     seg_graph = csr_matrix((alpha_omega[:,COL_W], (alpha_omega[:,COL_V1], alpha_omega[:, COL_V2])), shape=minimum_spanning_tree.shape)
#
#     # the resulting segmentation is made by the connected components of the segmented graph
#     ncc, labels = connected_components(seg_graph, directed=False)
#
#     return shuffle_labels(labels).astype(int)
