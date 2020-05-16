from pyntcloud import PyntCloud


def naive_ransac(cloud, max_dist, max_iterations, n_inliers_to_stop):
    """
    Naive method that estimates the ground as a unique plane.

    Parameters
    ----------
    cloud: PyntCloud
        input point cloud

    max_dist: float
        max distance from plane

    max_iterations: int
        max number of iterations the RANSAC method needs to do

    n_inliers_to_stop: int or None
        suffiecient number of inliers to stop the method
    """
    cloud.add_scalar_field('plane_fit',
                           max_dist=max_dist,
                           max_iterations=max_iterations,
                           n_inliers_to_stop=n_inliers_to_stop)

    return cloud.points.is_plane.values
