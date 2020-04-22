import numpy as np
import pandas as pd
from pyntcloud import PyntCloud


def load_label_file(bin_path, instances=False):
    """
    Utils function that read semantic-kitti labels
    TODO: create a specific library to read semantic-kitti labels and move this function there
    """
    labels = np.fromfile(bin_path, dtype=np.uint32).reshape(-1)

    seg_labels = labels & 0xFFFF

    if instances:
        inst = labels >> 16
        return seg_labels, inst

    return seg_labels


def load_bin_file(bin_path, n_cols=4):
    """
    Load a binary file and convert it into a numpy array.

    Parameters
    ----------
    bin_path: str
        path to bin file

    n_cols: int
        number of columns/features contained in the pc

    Returns
    -------
    point_cloud: ndarray
        loaded point cloud
    """
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, n_cols)


def load_kitti_pc(filename, add_label=False, instances=False):
    """
    Function that load a point cloud as numpy array
    Parameters
    ----------

    filename: str
        path to filename

    add_label: bool
        if true it loads labels coming from semantic-kitti dataset

    instances: bool
        if true it loads instance labels coming from semantic-kitti datased
    Returns
    -------
    points: ndarray
        loaded point cloud
    """
    points = load_bin_file(filename)

    if add_label:
        labels = load_label_file(filename.replace('velodyne', 'labels').replace('bin', 'label'), instances=instances)
        if instances:
            points = np.c_[points, labels[0], labels[1]]
        else:
            points = np.c_[points, labels]

    return points


def load_pyntcloud(filename, add_label=False, instances=False):
    """
    Parameters
    ----------
    filename: str
        path to pointcloud to read

    add_label: bool
        if True it adds also label to point cloud

    instances: bool
        if True it add also instances labels to point cloud

    Returns
    -------
    cloud: PyntCloud
        output pointcloud
    """
    points = load_bin_file(filename)
    cloud = PyntCloud(pd.DataFrame(points, columns=['x', 'y','z', 'i']))
    if add_label:
        labels = load_label_file(filename.replace('velodyne', 'labels').replace('bin', 'label'), instances=instances)
        if instances:
            cloud.points['labels'] = labels[0]
            cloud.points['instances'] = labels[1]
        else:
            cloud.points['labels'] = labels

    return cloud