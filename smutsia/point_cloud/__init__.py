from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

__all__ = [
    'filter_points',
    'get_sub_cloud'
]


def filter_points(points,
                  side_range=None,
                  fwd_range=None,
                  height_range=None,
                  intensity_range=None,
                  horizontal_fov=None,
                  vertical_fov=None,
                  return_indices=False):
    """
    Returns filtered points based on side(y), forward(x) and height(z) range,
    horizontal and vertical field of view, and intensity.

    Parameters
    ----------
    points: ndarray
        input point cloud to filter

    side_range: tuple

    fwd_range: tuple

    height_range: tuple

    intensity_range: tuple

    horizontal_fov: tuple

    vertical_fov: tuple

    return_indices: bool

    Returns
    -------
    points: ndarray
        filtered point cloud
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    i = points[:, 3]

    mask = np.full_like(x, True)

    if side_range is not None:
        side_mask = np.logical_and((y > -side_range[1]), (y < -side_range[0]))
        mask = np.logical_and(mask, side_mask)

    if fwd_range is not None:
        fwd_mask = np.logical_and((x > fwd_range[0]), (x < fwd_range[1]))
        mask = np.logical_and(mask, fwd_mask)

    if height_range is not None:
        height_mask = np.logical_and((z > height_range[0]), (z < height_range[1]))
        mask = np.logical_and(mask, height_mask)

    if intensity_range is not None:
        intensity_mask = np.logical_and((i > intensity_range[0]), (i < intensity_range[1]))
        mask = np.logical_and(mask, intensity_mask)

    if horizontal_fov is not None:
        horizontal_fov_mask = np.logical_and(np.arctan2(y, x) > (-horizontal_fov[1] * np.pi / 180),
                                             np.arctan2(y, x) < (-horizontal_fov[0] * np.pi / 180))
        mask = np.logical_and(mask, horizontal_fov_mask)

    if vertical_fov is not None:
        distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        vertical_fov_mask = np.logical_and(np.arctan2(z, distance) < (vertical_fov[1] * np.pi / 180),
                                           np.arctan2(z, distance) > (vertical_fov[0] * np.pi / 180))
        mask = np.logical_and(mask, vertical_fov_mask)

    indices = np.argwhere(mask).flatten()
    if return_indices:
        return points[indices], indices

    return points[indices]


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
