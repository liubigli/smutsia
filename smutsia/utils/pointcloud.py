from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import laspy
import numpy as np


def write_las(points, filepath, labels=None, color=None):
    """
    Function to store point cloud to las file

    Parameters
    ----------
    points: ndarray
        Nx3 array representing euclidean coordinates of the points

    filepath: str
        path to las file

    labels: ndarray
        array to use to store user_data

    color: ndarray
        unit8 array containing color information for points
    """
    hdr = laspy.header.Header(file_version=1.4, point_format=2)

    outfile = laspy.file.File(filepath, mode="w", header=hdr)
    min_X, min_Y, min_Z = points.min(0)

    outfile.header.offset = [min_X, min_Y, min_Z]
    outfile.header.scale = [0.001, 0.001, 0.001]

    outfile.x = points[:, 0]
    outfile.y = points[:, 1]
    outfile.z = points[:, 2]
    # labels = np.ones(len(points))
    if labels is not None:
        outfile.user_data = labels

    if color is not None:
        outfile.red = color[:, 0].astype(np.uint8)
        outfile.green = color[:, 1].astype(np.uint8)
        outfile.blue = color[:, 2].astype(np.uint8)

    outfile.close()
