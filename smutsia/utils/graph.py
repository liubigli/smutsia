from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse import find, csr_matrix
from scipy.spatial import cKDTree
from smutsia.point_cloud.projection import Projection

def resize_graph(graph, gshape, labels=None):
    """
    Change size of a graph

    Parameters
    ----------
    graph: csr_matrix
        Graph to resize

    gshape: tuple
        New size of the resized graph

    labels: ndarray
        Optional remapping to apply to nodes of the initial graph

    Returns
    -------
    graph: csr_matrix
        Representing the graph with new shape

    """
    [src, dest, weights] = find(graph)

    if labels is None:
        return csr_matrix((weights, (src, dest)), shape=gshape)

    return csr_matrix((weights, (labels[src], labels[dest])), shape=gshape)


def shuffle_labels(labels):
    """
    Function that shuffle labels of a segmentation

    Parameters
    ----------
    labels: N ndarray
        input labels

    Returns
    -------
    labels: N ndarray
        vector of shuffled labels

    """
    unique = np.unique(labels)
    np.random.seed(1)
    np.random.shuffle(unique)

    return unique[labels]


# auxiliary function that reconstruct the ith mst from the sequences T,E
def reconstruct_ith_mst(stable_forest, unstable_forest, ith=0):
    # easy case
    if ith == 0:
        return stable_forest[0] + unstable_forest[0]

    if ith > len(stable_forest) - 1 or ith < 0:
        ith = len(stable_forest) - 1

    max_shape = max(t.shape for t in stable_forest[:ith + 1])

    # resizing all the graphs in the lists
    for i in range(ith + 1):
        stable_forest[i] = resize_graph(stable_forest[i], max_shape)

    mst = sum(stable_forest[:ith + 1])

    if unstable_forest[ith] is not None:
        mst += unstable_forest[ith]

    return mst


def get_positive_degree_nodes(graph):
    """
    Function that given an adjacent matrix associated to a graph returns the nodes that has positive degree in the graph

    Parameters
    ----------
    graph: NxN csr_matrix
        Adjacent matrix representing graph

    Returns
    -------
    nodes: M ndarray
        Array containing id of nodes with positive degree

    """

    src, dst, _ = find(graph)

    nodes = np.unique(np.concatenate((src, dst)))

    return nodes


def get_subgraph(graph, nodes, return_map=False):
    """
    Function that given a graph and a list of nodes returns the sub graph restricted only to those nodes

    Parameters
    ----------
    graph: NxN csr_matrix
        Adjacent sparse matrix representing graph

    nodes: M ndarray
        Nodes of the restricted graph

    return_map: bool
        If True returns an object that allows to map any node in nodes to a node in the new matrix

    Returns
    -------
    subgraph: MxM csr_matrix
        Graph restricted only to nodes in nodes

    backmap: dict
        Dictionary that maps initial nodes to new nodes
    """
    # remark that the id of the nodes in this subgraph are remapped
    if return_map:
        cnodes = nodes.copy()
        cnodes.sort()
        backmap = {cnodes[i]: i for i in range(len(cnodes))}

        return graph[cnodes, :][:, cnodes], backmap

    else:
        return graph[nodes, :][:, nodes]


def merge_graphs(graphs):
    """
    Function that takes two graphs and returns its union.

    Parameters
    ----------
    graphs: tuple of NxN csr_matrix
        A tuple of graphs

    Returns
    -------
    graph: NxN csr_matrix
        Union of graph1 and graph2
    """
    graph = None

    for n, g in enumerate(graphs):
        if n == 0:
            graph = g
        else:
            graph = graph.multiply(graph > 0).maximum(g.multiply(g > 0)) + \
                        graph.multiply(graph < 0).minimum(g.multiply(g < 0))

    return graph



def cloud_knn_graph(xyz, k=10, metric=None):
    """
    Parameters
    ----------
    xyz: PyntCloud
        input point cloud

    k: int
        number of neighbors to consider

    metric: func
        function used as a metric for the graph
    Returns
    -------
    graph: csr_matrix
        k-nn graph
    """
    n_points = len(xyz)
    tree = cKDTree(xyz)
    dist, neighs = tree.query(xyz, k=k + 1)
    knn = neighs[:, 1:]
    src = np.repeat(np.arange(n_points), k)
    dst = knn.flatten()

    if metric is None:
        weights = np.ones(len(src))
    else:
        weights = metric(xyz[src], xyz[dst])

    # initialise graph
    graph = csr_matrix((weights, (src, dst)), shape=(n_points, n_points))

    # make sparse matrix symmetric
    graph = graph.maximum(graph.T)

    return graph


def cloud_spherical_graph(xyz, res_pitch=64, res_yaw=2048, metric=None):
    """
    Parameters
    ----------
    xyz: ndarray

    res_pitch: int

    res_yaw: int

    metric: function

    Returns
    -------
    graph: csr_matrix
    """
    n_points = len(xyz)
    # img_edges = _make_edges_spherical_images(nr=res_pitch, nc=res_yaw)
    fov_pitch = [85.0 * np.pi / 180.0, 115.0 * np.pi / 180.0]
    fov_yaw = [0.0, 2 * np.pi]
    proj = Projection(proj_type='spherical', res_yaw=res_yaw, res_pitch=res_pitch, fov_yaw=fov_yaw, fov_pitch=fov_pitch)
    lidx, i_img, j_img = proj.projector.project_point(xyz)

    # todo: improve this code
    unique, inverse = np.unique(lidx, return_inverse=True)
    acc_map = np.zeros(res_pitch*res_yaw, dtype=np.bool)
    acc_map[unique[1:]] = True
    pxl2points = {u: [] for u in unique[1:]}
    for n, l in enumerate(lidx):
        if l > 0:
            pxl2points[l].append(n)

    edges = []

    for k in pxl2points:
        v = pxl2points[k]
        i, j = k // res_yaw, k % res_yaw

        if len(v) > 1:
            vv = np.array(np.meshgrid(v, v)).T.reshape(-1, 2)
            edges.append(vv[vv[:, 0] != vv[:, 1]])

        l_n, b_n = i * res_yaw + (j + 1) % res_yaw, k + res_yaw

        if acc_map[l_n]:
            w = pxl2points[l_n]
            edges.append(np.array(np.meshgrid(v, w)).T.reshape(-1, 2))

        if b_n < res_yaw * res_pitch and acc_map[b_n]:
            w = pxl2points[b_n]
            edges.append(np.array(np.meshgrid(v, w)).T.reshape(-1, 2))

    edges = np.concatenate(edges)
    src = edges[:, 0]
    dst = edges[:, 1]

    if metric is not None:
        w = metric(xyz[src], xyz[dst])
    else:
        w = np.ones_like(src)

    graph = csr_matrix((w, (src, dst)), shape=(n_points, n_points))

    graph = graph.maximum(graph.T)

    return graph
