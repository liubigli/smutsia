import os
import ntpath
import yaml
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from skimage.morphology import opening, rectangle


class SemanticKittiConfig:
    """
    Class that load the semantic kitti config file and helps to handle class ids
    """
    def __init__(self, config_file):
        """
        Parameters
        ----------
        config_file: str to config file
        """
        self.config_file = config_file
        self.config = self.load_semantic_kitti_config(config_file)
        labels2id, id2label = self._remap_classes()
        self.labels2id = labels2id  # hashmap used to generate labels for semantic-segmentation in DL
        self.id2label = id2label  # labels2id inverse hashmap
        self.label2color = self._color_map()
        self.labels2ground = self._label_to_ground_remap()

    @staticmethod
    def load_semantic_kitti_config(filename=""):
        """
        Function that load configuration file of semantic kitti dataset
        Parameters
        ----------
        filename: str
            file to load
        Returns
        -------
        config: dict
        """
        if len(filename) == 0:
            local_path = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(local_path, '..', 'semantic-kitti-api', 'config', 'semantic-kitti.yaml')

        with open(filename) as f:
            config = yaml.safe_load(f)

        return config

    def _color_map(self):
        """
        Map each label to a color RGB
        """
        max_id = max(self.config['color_map'])
        label2colors = np.zeros((max_id + 1, 3), dtype=np.uint8)
        for k, v in self.config['color_map'].items():
            label2colors[k] = v

        return label2colors

    def _remap_classes(self):
        """
        Maps semantic-kitti labels to classes to learn for semantic segmentation
        """
        learning_map = self.config['learning_map']
        learning_map_inv = self.config['learning_map_inv']

        remaps = np.zeros(max(learning_map.keys()) + 1, dtype=int)
        for k, v in learning_map.items():
            remaps[k] = v

        inv_remaps = np.zeros(max(learning_map_inv.keys()) + 1, dtype=int)
        for k, v in learning_map_inv.items():
            inv_remaps[k] = v

        return remaps, inv_remaps

    def _label_to_ground_remap(self):
        """
        Maps semantic-kitti labels to ground/not ground label
        """
        ground_items = np.array([40, 44, 48, 49, 60, 72])
        label_keys = list(self.config['learning_map'].keys())
        labels2ground = np.zeros(max(label_keys) + 1, dtype=np.int)
        labels2ground[ground_items] = 1
        return labels2ground


def retrieve_layers(points, max_layers=64):
    """
    Function that retrieve the layer for each point. We do the hypothesis that layer are stocked one after the other.
    And each layer is stocked in a clockwise (or anticlockwise) fashion.

    Parameters
    ----------
    points: ndarray
        input point cloud

    max_layers: int
        maximum number of conv to detect

    Returns
    -------
    conv: ndarray
        array containing for each point the id of corresponding layer in the scanner that acquired it
    """
    x = points[:, 0]
    y = points[:, 1]

    # compute the theta angles
    thetas = np.arctan2(y, x)
    op_thetas = opening(thetas.reshape(-1, 1), rectangle(20, 1))
    thetas = op_thetas.flatten()
    idx = np.ones(len(thetas))

    idx_pos = idx.copy()
    idx_pos[thetas < 0] = 0

    # since each layer is stocked in a clockwise fashion each time we do a 2*pi angle we can change layer
    # so we identify each time we do a round
    changes = np.arange(len(thetas) - 1)[np.ediff1d(idx_pos) == 1]
    changes += 1  # we add one for indexes reason

    # Stocking intervals. Each element of intervals contains min index and max index of points in the same layer
    intervals = []
    for i in range(len(changes)):
        if i == 0:
            intervals.append([0, changes[i]])
        else:
            intervals.append([changes[i - 1], changes[i]])

    intervals.append([changes[-1], len(thetas)])

    # check if we have retrieved all the conv
    if len(intervals) < max_layers:
        el = intervals.pop(0)
        # in case not we are going to explore again the vector of thetas on the initial part
        thex = np.copy(thetas[:el[1]])
        # we compute again the diffs between consecutive angles and we mark each time we have a negative difference
        diffs = np.ediff1d(thex)
        idx = diffs < 0
        ints = np.arange(len(idx))[idx]
        # the negative differences mark the end of a layer and the beginning of another
        new_intervals = []
        max_new_ints = min(len(ints), max_layers - len(intervals))
        for i in range(max_new_ints):
            if i == 0:
                new_intervals.append([0, ints[i]])
            elif i == max_new_ints - 1:
                new_intervals.append([ints[i], el[1]])
            else:
                new_intervals.append([ints[i], ints[i + 1]])
        intervals = new_intervals + intervals

    # for each element in interval we assign a label that identifies the layer
    layers = np.zeros(len(thetas), dtype=np.uint8)

    for n, el in enumerate(intervals[::-1]):
        layers[el[0]:el[1]] = max_layers - (n + 1)

    return layers


def subsample_pc(points, sub_ratio=2, return_layers=False):
    """
    Return sub sampled point cloud

    Parameters
    ----------
        points: ndarray
            input point cloud

        sub_ratio: int
            ratio to use to subsample point cloud

    Returns
    -------
        points: ndarray
            subsampled pointcloud
    """
    layers = retrieve_layers(points)
    # new_points = np.c_[points, conv]
    if return_layers:
        return points[layers % sub_ratio == 0], layers
    # sampling only points with even id
    return points[layers % sub_ratio == 0]


def add_layers(points):
    """
    Add a column containing layer ids to the array of points

    Parameters
    ----------
    points: ndarray
        input point cloud

    Returns
    -------
    new_points: ndarray
        point cloud with layer information
    """
    layers = retrieve_layers(points)

    new_points = np.c_[points, layers]

    return new_points


def load_label_file(bin_path, instances=False):
    """
    Utils function that read semantic-kitti labels

    Parameters
    ----------
    bin_path: str
        path to binary path

    instances: bool
        if True it loads also istance labels

    Returns
    -------
        seg_labels: ndarray
            point label array

        inst: ndarray
            instance label array
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


def load_pyntcloud(filepath, add_label=False, instances=False):
    """
    Parameters
    ----------
    filepath: str
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
    points = load_bin_file(filepath)
    cloud = PyntCloud(pd.DataFrame(points, columns=['x', 'y', 'z', 'i']))
    if add_label:
        labels = load_label_file(filepath.replace('velodyne', 'labels').replace('bin', 'label'), instances=instances)
        if instances:
            cloud.points['labels'] = labels[0]
            cloud.points['instances'] = labels[1]
        else:
            cloud.points['labels'] = labels

    basename = ntpath.basename(filepath)
    out = basename.split('.')[:-1]
    if len(out) == 1:
        filename = out[0]
    else:
        filename = '.'.join(out)

    dir_path = filepath.split('/')

    sequence = None
    try:
        seq_idx = dir_path.index('sequences')
        sequence = dir_path[seq_idx + 1]
    except ValueError:
        print("Sequence value not found in filepath")

    # adding to object attributes about filename and filepath
    cloud.filename = filename
    cloud.filepath = filepath
    cloud.sequence = sequence

    return cloud
