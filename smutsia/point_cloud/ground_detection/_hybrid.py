import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from scipy.sparse import find, csr_matrix
from smutsia.point_cloud.normals import get_normals
from smutsia.morphology.segmentation import quasi_flat_zones, z_nz_dist
from smutsia.utils import subset_backprojection
from smutsia.utils.graph import cloud_knn_graph, cloud_spherical_graph, merge_graphs


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
    # counting cc ans selecting only those bigger than 100 points
    cc_ids, cc_size = np.unique(cc, return_counts=True)
    # sort cc according to their size
    iargsort = cc_size.argsort()[::-1]
    cc_size = cc_size[iargsort]
    cc_ids = cc_ids[iargsort]
    sel_cc = cc_ids[cc_size > cc_min_size]

    # biggest cc is always considered as ground
    cc0 = cc == cc_ids[0]
    ground = cc0.copy()

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

    knn_graph = cloud_knn_graph(cloud.xyz, k=knn_graph)
    spherical_graph = cloud_spherical_graph(cloud.xyz, nb_layers=nb_layers, res_yaw=res_yaw)
    graph = merge_graphs([knn_graph, spherical_graph])
    normals = get_normals(cloud, method=method_normals, k=knn_normal)
    src, dst, _ = find(graph)
    weights = z_nz_dist(cloud.xyz, normals, src, dst)
    wg = csr_matrix((weights, (src, dst)), shape=graph.shape)

    cc = quasi_flat_zones(wg, threshold=threshold, debug_info=True)
    # plot_cloud(cloud.xyz, scalars=cc, cmap=plt.cm.tab20, point_size=1.5, interact=True, notebook=False)
    ground = merge_labels(cloud.xyz, cc, cc_min_size=cc_min_size,
                          ransac_max_dist=ransac_max_dist, ransac_max_it=ransac_max_it, inter_perc=inter_perc)
    # plot_cloud(cloud.xyz, scalars=ground, point_size=1.5, interact=True, notebook=False)

    return ground
