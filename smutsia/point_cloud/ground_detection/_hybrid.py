import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from scipy.sparse import find, csr_matrix
from smutsia.point_cloud.normals import get_normals
from smutsia.morphology.segmentation import quasi_flat_zones, z_nz_dist
from smutsia.utils import subset_backprojection
from smutsia.graph import cloud_knn_graph, cloud_spherical_graph, merge_graphs


def get_sub_cloud(xyz, subset):
    """
    Utils function that return sub point cloud
    Parameters
    ----------
    xyz: ndarray
        input point cloud

    subset: ndarray
        boolean array defining subset

    Returns
    -------
    sub_cloud: PyntCloud
        point_cloud made of points in subset
    """
    return PyntCloud(pd.DataFrame(xyz[subset], columns=['x', 'y', 'z']))


def is_comparable(xyz, cc1, cc2, max_dist=0.15, max_it=100, inter_perc=0.5):
    """
    Function that says if two connected components are mergeable fiting a plane between the two. If the inliers points
    are more than 50% of the second components then the two are comparable.

    Parameters
    ----------
    xyz: ndarray
        input point cloud

    cc1: ndarray
        boolean array that contains 1 at position i if i-th point belong to cc1

    cc2: ndarray
        boolean array that contains 1 at position i if i-th point belong to cc1

    max_dist: float
        max_dist used in RANSAC method

    max_it: int
        maximum number of iterations to use

    inter_perc: float
        percentage of points of cc2 that must be inliers in order to define cc2 comparable with cc1

    Returns
    -------
    comparable: bool
        True if cc1 and cc2 are comparable. False otherwise.
    """
    sub_cloud = get_sub_cloud(xyz, np.logical_or(cc1, cc2))
    sub_cloud.add_scalar_field('plane_fit', max_dist=max_dist, max_iterations=max_it, n_inliers_to_stop=None)

    back_proj = subset_backprojection(np.logical_or(cc1, cc2))
    back_is_plane = np.zeros(len(xyz), dtype=np.bool)
    back_is_plane[back_proj[sub_cloud.points.is_plane.values.astype(np.bool)]] = True
    inter_size = (np.logical_and(back_is_plane, cc2).sum()) / cc2.sum()

    if inter_size > inter_perc:
        return True
    else:
        return False


def merge_labels(xyz, cc, cc_min_size=40, ransac_max_dist=0.15, ransac_max_it=100, inter_perc=0.5):
    """
    Parameters
    ---------
    xyz: ndarray
        input point cloud

    cc: ndarray

    cc_min_size: int
        minimum size of cc to consider during merge process

    ransac_max_dist: float
        max dist to ransac plane used to measure if two ccs are comparable

    ransac_max_it: int
        maximum number of iteration ransac method should do during merging

    inter_perc: float
        minimum percentage of inliers to consider two cc comparable

    Returns
    -------
    ground: ndarray
        boolean array that for each point says if it belongs or not to the ground
    """
    # counting cc ans selecting only those bigger than cc_min_size points
    cc_ids, cc_size = np.unique(cc, return_counts=True)
    # sort cc according to their size
    iargsort = cc_size.argsort()[::-1]
    cc_size = cc_size[iargsort]
    cc_ids = cc_ids[iargsort]
    # select connected components bigger than cc_min_size
    sel_cc = cc_ids[cc_size > cc_min_size]

    # biggest cc is always considered as ground
    cc0 = cc == cc_ids[0]

    # copy cc0 to ground
    ground = np.copy(cc0)

    for i in sel_cc[1:]:
        if is_comparable(xyz, cc0, cc == i, max_dist=ransac_max_dist, max_it=ransac_max_it, inter_perc=inter_perc):
            ground = np.logical_or(ground, cc == i)

    return ground


def hybrid_ground_detection(cloud,
                            threshold,
                            knn_graph=10,
                            nb_layers=64,
                            res_yaw=2048,
                            method_normals='pca',
                            knn_normal=30,
                            cc_min_size=40,
                            ransac_max_dist=0.15,
                            ransac_max_it=100,
                            inter_perc=0.5):
    """
    Parameters
    ----------
    cloud: PyntCloud
        input point cloud

    threshold: float
        threshold value for lambda flat zones

    knn_graph: int
        number of nearest neighbors to connect

    nb_layers: int
        number of layers of the scanner

    res_yaw: int
        horizontal resolution of the spherical image

    method_normals: optional {'pca', 'spherical'}
        method to use to estimate normals

    knn_normal: int
        number of nearest neighbors to consider while estimating normals

    cc_min_size: int
        minimum size of connected components to merge

    ransac_max_dist: float
        max dist from the merging plane for inlier points

    ransac_max_it: int
        max number of iteration for ransac method

    inter_perc: float
        minimum ratio of inliners points to consider a cc as comparable with ground
    """
    # initialise knn graph
    k_graph = cloud_knn_graph(cloud.xyz, k=knn_graph)
    # initialise spherical graph
    spherical_graph = cloud_spherical_graph(cloud.xyz, nb_layers=nb_layers, res_yaw=res_yaw)
    # 3D is the union of knn graph and spherical graph
    graph = merge_graphs([k_graph, spherical_graph])

    # estimate normals
    normals = get_normals(cloud, method=method_normals, k=knn_normal)

    src, dst, _ = find(graph)
    # weight of the 3D graph are defined by z_nz_dist
    weights = z_nz_dist(cloud.xyz, normals, src, dst)

    # wg is the weighted graph that we are going to use
    wg = csr_matrix((weights, (src, dst)), shape=graph.shape)

    # extract quasi flat zones removing from the graph all the edges whose weight is bigger than threshold value
    cc = quasi_flat_zones(wg, threshold=threshold, debug_info=True)

    # finally we extract the ground analysing biggest connected components and merging the comparable between them
    ground = merge_labels(cloud.xyz, cc, cc_min_size=cc_min_size,
                          ransac_max_dist=ransac_max_dist, ransac_max_it=ransac_max_it, inter_perc=inter_perc)
    return ground


def connect_3d_graph(cloud, normals, wg):
    from scipy.sparse.csgraph import connected_components
    from smutsia.utils import set_distance, cartesian_product
    from scipy.spatial import cKDTree

    n_cc, cc_wg = connected_components(wg)
    dij = np.zeros((n_cc, n_cc))
    aminij = np.zeros((n_cc, n_cc, 2), dtype=np.int)
    for i in range(n_cc):
        for j in range(i + 1, n_cc):
            dij[i, j], (h, k) = set_distance(cloud.xyz[cc_wg == i], cloud.xyz[cc_wg == j], return_amin=True)
            # remap points to id in original point cloud
            h = np.arange(len(cloud.xyz))[cc_wg == i][h]
            k = np.arange(len(cloud.xyz))[cc_wg == j][k]
            aminij[i, j] = h, k
            aminij[j, i] = k, h
            dij[j, i] = dij[i, j]
    closest_cc = dij.argsort(axis=1)[:, 1:4]
    trees = [cKDTree(cloud.xyz[cc_wg == i]) for i in range(n_cc)]
    sub_ids = [np.arange(len(cloud.xyz))[cc_wg == i] for i in range(n_cc)]
    src_mini = []
    dst_mini = []
    for i in range(n_cc):
        for j in closest_cc[i]:
            x_i, x_j = aminij[i, j]
            _, neigh_i = trees[i].query(cloud.xyz[x_j], k=min(30, len(sub_ids[i]) - 1))
            _, neigh_j = trees[j].query(cloud.xyz[x_i], k=min(30, len(sub_ids[j]) - 1))

            neigh_i = sub_ids[i][neigh_i]
            neigh_j = sub_ids[j][neigh_j]
            out = cartesian_product([neigh_i, neigh_j])
            out = out[np.linalg.norm(cloud.xyz[out[:, 0]] - cloud.xyz[out[:, 1]], axis=1) < 12.0]
            src_mini.append(out[:, 0])
            dst_mini.append(out[:, 1])

    src_mini = np.concatenate(src_mini)
    dst_mini = np.concatenate(dst_mini)
    weights_mini = z_nz_dist(cloud.xyz, normals, src_mini, dst_mini)
    mini_graph = csr_matrix((weights_mini, (src_mini, dst_mini)), shape=wg.shape)

    return wg.maximum(mini_graph)


def select_cc(cc, z, cc_min_size=20, max_cc=10):
    cc_ids, cc_size = np.unique(cc, return_counts=True)
    iargsort = cc_size.argsort()[::-1]
    cc_size = cc_size[iargsort]
    cc_ids = cc_ids[iargsort]
    # select connected components bigger than cc_min_size
    sel_cc = cc_ids[cc_size > cc_min_size]
    z_vals = np.zeros_like(sel_cc)
    for n, ids in enumerate(sel_cc):
        z_vals[n] = z[cc == ids].mean()

    zarg = z_vals.argsort()
    sub_set = np.zeros_like(cc, dtype=np.bool)
    for ids in sel_cc[zarg][:max_cc]:
        sub_set += cc == ids
    return sub_set, subset_backprojection(sub_set)


def iterative_step(cloud, knn_normals=10, spherical_graph=None):
    normals = get_normals(cloud, method='pca', k=knn_normals)
    knn_graph = cloud_knn_graph(cloud.xyz, k=10)
    if spherical_graph is None:
        spherical_graph = cloud_spherical_graph(cloud.xyz, nb_layers=64, res_yaw=2048)
    # 3D graph
    graph = merge_graphs([knn_graph, spherical_graph])

    src, dst, _ = find(graph)
    # the weights are z / n_z
    weigths = z_nz_dist(cloud.xyz, normals, src, dst)
    # weighted 3D graph
    wg = csr_matrix((weigths, (src, dst)), shape=graph.shape)
    wg = connect_3d_graph(cloud, normals, wg)

    ncc, cc = quasi_flat_zones(wg, threshold=0.20, debug_info=True, return_ncc=True)
    return ncc, cc


def iterative_hybrid(cloud,
                     threshold,
                     knn_graph=10,
                     nb_layers=64,
                     res_yaw=2048,
                     method_normals='pca',
                     knn_normal=30,
                     cc_min_size=40,
                     ransac_max_dist=0.15,
                     ransac_max_it=100,
                     inter_perc=0.5):

    max_cc = 20  ## absolutely wrong!!! we must find a way to change this
    subcloud = get_sub_cloud(cloud.xyz, np.ones(len(cloud.xyz), dtype=np.bool))
    spherical_graph = cloud_spherical_graph(cloud.xyz, nb_layers=64, res_yaw=2048)
    ncc, cc0 = iterative_step(subcloud, knn_normals=knn_normal, spherical_graph=spherical_graph)
    backprop = np.arange(len(cloud.xyz))
    it = 0
    cc = cc0.copy()
    while ncc > max_cc and it < 10:
        print(it)
        subset, backprop_t = select_cc(cc, subcloud.xyz[:, 2], max_cc=max_cc)
        backprop = backprop[backprop_t]
        subcloud = get_sub_cloud(subcloud.xyz, subset)
        ncc, cc = iterative_step(subcloud, knn_normals=knn_normal,
                                 spherical_graph=spherical_graph[backprop, :][:, backprop])
        it += 1

    cc_ground = np.unique(cc0[backprop])

    ground = np.zeros_like(cc0, dtype=np.bool)
    for ids in cc_ground:
        ground += (cc0 == ids)

    return ground
