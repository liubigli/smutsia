import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from smutsia.point_cloud.projection import Projection
from smutsia.utils import merge_graphs, set_distance, cartesian_product


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


def cloud_spherical_graph(xyz, nb_layers=64, res_yaw=2048, metric=None):
    """
    Parameters
    ----------
    xyz: ndarray
        input point cloud

    nb_layers: int
        number of layers in the scanner used for acquiring point cloud

    res_yaw: int
        yaw resolution used for projection by layer

    metric: function
        custom metric function used to assign weights to graph.

    Returns
    -------
    graph: csr_matrix
    """
    from smutsia.graph.spherical_edges import build_spherical_edges

    n_points = len(xyz)
    # img_edges = _make_edges_spherical_images(nr=res_pitch, nc=res_yaw)

    proj = Projection(proj_type='layers', res_yaw=res_yaw, nb_layers=nb_layers)
    lidx, i_img, j_img = proj.projector.project_point(xyz)

    # using c++ code to build spherical graph
    edges = build_spherical_edges(lidx, xyz)

    src = edges[:, 0]
    dst = edges[:, 1]

    if metric is not None:
        w = metric(xyz[src], xyz[dst])
    else:
        w = np.ones_like(src)

    graph = csr_matrix((w, (src, dst)), shape=(n_points, n_points))

    graph = graph.maximum(graph.T)

    return graph


def cloud_3d_graph(xyz, k=10, nb_layers=64, res_yaw=2048, metric=None, return_connected=False):
    """
    Auxiliary function that build a graph on 3D point cloud as the sum of a spherical graph and knn graph

    Parameters
    ----------
    xyz: input point cloud

    k: int
        number of nearest neighbors to consider to build knn graph

    nb_layers: int
        Number of layers used by the scanners. Used for projection by layers

    res_yaw: int
        horizontal resolution to use for projection by layers

    metric: func
        custom function to use to weights edges

    return_connected: bool
        if true add edges to the graph until the graph is connected

    Returns
    -------
    graph: csr_matrix
        output graph
    """
    knn_graph = cloud_knn_graph(xyz, k=k, metric=metric)
    spherical_graph = cloud_spherical_graph(xyz, nb_layers=nb_layers, res_yaw=res_yaw, metric=metric)

    graph = merge_graphs([knn_graph, spherical_graph])

    if return_connected:
        """
        to return a connected graph we deploy the following strategy
        For each connected component we find the 3 closest connected components. 
        For each couple of connected componets
        We add connections between closest points among them. 
        """

        n_cc, cc = connected_components(graph, directed=False)
        if n_cc == 1:
            return graph

        sub_ids = [np.arange(len(xyz))[cc == i] for i in range(n_cc)]
        trees = [cKDTree(xyz[sid]) for sid in sub_ids]
        # matrix of distances between connected components
        dij = np.zeros((n_cc, n_cc))
        aminij = np.zeros((n_cc, n_cc, 2))
        for i in range(n_cc):
            for j in range(i+1, n_cc):
                dij[i, j], (ii, jj) = set_distance(xyz[cc == i], xyz[cc == j], return_amin=True)
                dij[j, i] = dij[i, j]
                ii = sub_ids[i][ii]
                jj = sub_ids[j][jj]
                aminij[i, j] = ii, jj
                aminij[j, i] = jj, ii

        # for each cc we retrieve the first 3 closest cc
        closest_cc = dij.argsort(axis=1)[:, 1:min(4, n_cc)]

        src = []
        dst = []
        for i in range(n_cc):
            for j in closest_cc[i]:
                # x_i in cc_i and x_j in cc_j are the closest points between
                x_i, x_j = aminij[i, j]
                # we find the 30 closest points to x_i  in cc_j
                _, neigh_i = trees[i].query(xyz[x_j], k=min(30, len(sub_ids[i])-1))
                # find the 30 closest points to x_j in cc_i
                _, neigh_j = trees[j].query(xyz[x_i], k=min(30, len(sub_ids[j])-1))
                # remapping neighs
                neigh_i = sub_ids[i][neigh_i]
                neigh_j = sub_ids[j][neigh_j]
                # the connecting edges are in the cartesian product between neigh_i and neigh_j
                out = cartesian_product([neigh_i, neigh_j])
                src.append(out[:, 0])
                dst.append(out[:, 1])

        src = np.concatenate(src)
        dst = np.concatenate(dst)

        if metric is not None:
            w = metric(xyz[src], xyz[dst])
        else:
            w = np.ones_like(src)

        graph_to_add = csr_matrix((w, (src, dst)), shape=graph.shape)

        return graph.maximum(graph_to_add)

    return graph
