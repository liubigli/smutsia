import numpy as np
from pyntcloud import PyntCloud
from smutsia.point_cloud import filter_points, get_sub_cloud

def naive_ransac(cloud, min_height, max_height, max_dist, max_iterations, n_inliers_to_stop=None):
    """
    Naive method that estimates the ground as a unique plane.

    Parameters
    ----------
    cloud: PyntCloud
        input point cloud

    min_height: float

    max_height: float

    max_dist: float
        max distance from plane

    max_iterations: int
        max number of iterations the RANSAC method needs to do

    n_inliers_to_stop: int or None
        suffiecient number of inliers to stop the method
    """
    points, indices = filter_points(cloud.points.values, height_range=(min_height, max_height), return_indices=True)
    sub_indices = np.arange(len(cloud.xyz))[indices]
    sub_cloud = get_sub_cloud(cloud.xyz, indices)

    sub_cloud.add_scalar_field('plane_fit',
                           max_dist=max_dist,
                           max_iterations=max_iterations,
                           n_inliers_to_stop=n_inliers_to_stop)

    # retrieve inliers points in the subcloud
    plane = sub_cloud.points.is_plane.values.astype(np.bool)

    # initialise ground vector
    ground = np.zeros(len(cloud.xyz), dtype=np.bool)

    ground[sub_indices[plane]] = True

    return ground


if __name__ == "__main__":
    import os
    from glob import glob
    from smutsia.utils.semantickitti import load_pyntcloud
    from smutsia.utils.viz import plot_cloud
    from definitions import SEMANTICKITTI_PATH
    basedir = os.path.join(SEMANTICKITTI_PATH, '08', 'velodyne')
    files = sorted(glob(os.path.join(basedir, '*.bin')))
    pc = load_pyntcloud(files[0], add_label=True)
    g = naive_ransac(pc, min_height=-4, max_height=-1, max_dist=0.2, max_iterations=200, n_inliers_to_stop=None)
    plot_cloud(pc.xyz, scalars=g, notebook=False)
    print("End")
