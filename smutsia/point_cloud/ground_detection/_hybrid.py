import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from scipy.sparse import find, csr_matrix
from smutsia.point_cloud.normals import get_normals
from smutsia.morphology.segmentation import quasi_flat_zones, z_nz_dist
from smutsia.utils.graph import cloud_knn_graph, cloud_spherical_graph, merge_graphs

def get_subCloud(xyz, subset):
    return PyntCloud(pd.DataFrame(xyz[subset],columns=['x', 'y', 'z']))


def subset_backprojection(bool_map):
    n = len(bool_map)
    return np.arange(n)[bool_map]


def is_mergeable(xyz, cc1, cc2):
    scalars = np.zeros(len(xyz), dtype=np.int)
    scalars[cc1] = 1
    scalars[cc2] = 2
    sub_cloud = get_subCloud(xyz, np.logical_or(cc1, cc2))
    sub_cloud.add_scalar_field('plane_fit', max_dist=0.15, max_iterations=100, n_inliers_to_stop=None)
    sub_cloud.points.is_plane.values.astype(np.bool)
    back_proj = subset_backprojection(np.logical_or(cc1, cc2))
    back_is_plane = np.zeros(len(xyz), dtype=np.bool)
    back_is_plane[back_proj[sub_cloud.points.is_plane.values.astype(np.bool)]] = True
    inter_size = (np.logical_and(back_is_plane, cc2).sum()) / cc2.sum()
    if inter_size > 0.5:
        return True
    else:
        return False


def merge_labels(xyz, cc):
    # counting cc ans selecting only those bigger than 100 points
    cc_ids, cc_size = np.unique(cc, return_counts=True)
    iargsort = cc_size.argsort()[::-1]
    cc_size = cc_size[iargsort]
    cc_ids = cc_ids[iargsort]
    sel_cc = cc_ids[cc_size > 40]

    cc0 = cc == cc_ids[0]
    ground = cc0.copy()
    for i in sel_cc[1:]:
        if is_mergeable(xyz, cc0, cc==i):
            ground = np.logical_or(ground, cc==i)

    return ground


def hybrid_ground_detection(cloud, threshold, knn_graph=10, nb_layers=64, res_yaw=2048, methdo_normals='pca', ):
    knn_graph = cloud_knn_graph(cloud.xyz, k=knn_graph)
    spherical_graph = cloud_spherical_graph(cloud.xyz, nb_layers=nb_layers, res_yaw=res_yaw)
    graph = merge_graphs([knn_graph, spherical_graph])
    normals = get_normals(cloud, method='pca', k=30)
    src, dst, _ = find(graph)
    weights = z_nz_dist(cloud.xyz, normals, src, dst)
    wg = csr_matrix((weights, (src, dst)), shape=graph.shape)

    cc = quasi_flat_zones(wg, threshold=threshold, debug_info=True)
    # plot_cloud(cloud.xyz, scalars=cc, cmap=plt.cm.tab20, point_size=1.5, interact=True, notebook=False)
    ground = merge_labels(cloud.xyz, cc)
    # plot_cloud(cloud.xyz, scalars=ground, point_size=1.5, interact=True, notebook=False)

    return ground
