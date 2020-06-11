# inspired by https://github.com/PRBonn/lidar-bonnetal/blob/master/train/common/laserscan.py
import numpy as np
from smutsia.point_cloud.projection import Projection
from smutsia.point_cloud.normals import get_normals
from smutsia.utils.semantickitti import SemanticKittiConfig


class LaserScan:
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, projection=None, add_normals=False):
        """
        Parameters
        ----------
        projection: Projection

        add_normals: bool
        """
        self.projection = projection
        if projection.proj_type == 'spherical':
            self.proj_h = projection.res_pitch
        elif projection.proj_type == 'layers':
            self.proj_h = projection.nb_layers
        else:
            raise ValueError("Projecion can be only 'spherical' or 'layers', but was {}".format(projection.proj_type))

        self.proj_w = projection.res_yaw
        self.fov_up = projection.fov_pitch[0]
        self.fov_down = projection.fov_pitch[1]
        self.add_normals = add_normals

        self.__points = np.zeros((0, 3), dtype=np.float32)
        self.__reflectance = np.zeros((0, 1), dtype=np.float32)
        self.__proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)

    @property
    def points(self):
        return self.__points

    @points.setter
    def points(self, points):
        self.__points = points

    @property
    def reflectance(self):
        return self.__reflectance

    @reflectance.setter
    def reflectance(self, reflectance):
        self.__reflectance = reflectance

    @property
    def proj_range(self):
        return self.__proj_range

    @proj_range.setter
    def proj_range(self, proj_range):
        self.__proj_range = proj_range

    def reset(self):
        """
        Reset all value
        """
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.reflectance = np.zeros((0, 1), dtype=np.float32)
        self.proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)

    def size(self):
        """ return the size of the point cloud """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """
        Open raw scan adn fill in attributes
        """

        # reset just in case there was an open structure
        self.reset()

        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, but was {type}".format(type=type(filename)))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")
        pass

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # get xyz
        points = scan[:, :3]
        # get refectance values
        reflectance = scan[:, 3]

        self.set_points(points, reflectance)

    def set_points(self, points, reflectance):
        """Set scan attributes"""

        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if reflectance is not None and not isinstance(reflectance, np.ndarray):
            raise TypeError("Intensity should be numpy array")

        self.points = points

        if reflectance is not None:
            self.reflectance = reflectance
        else:
            self.reflectance = np.zeros((points.shape[0]), dtype=np.float32)

        if self.projection is not None:
            self.do_projection()

    def do_projection(self):

        rho = np.linalg.norm(self.points, axis=1)
        z = self.points[:, 2]
        aggr_func = ['max', 'min', 'mean']
        proj_img = self.projection.project_points_values(self.points, np.c_[rho, z, self.reflectance],
                                                         aggregate_func=aggr_func)
        if self.add_normals:
            norm_img = get_normals(cloud=self.points, method='spherical', proj=self.projection,
                                   res_yaw=self.proj_w, res_pitch=self.proj_h)

            proj_img = np.dstack([proj_img, norm_img])

        self.proj_range = proj_img


class SemanticKittiLaserScan(LaserScan):
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, projection=None, add_normals=False, skconfig=None):
        """
        Parameters
        ----------
        projection: Projection

        add_normals: bool

        skconfig: SemanticKittiConfig

        """
        super().__init__(projection=projection, add_normals=add_normals)
        self.skconfig = skconfig

        # semantic labels
        self.__sem_labels = np.zeros((0, 1), dtype=np.int32)
        self.__sem_labels_color = np.zeros((0, 3), dtype=np.uint8)

        # instance labels
        self.__inst_labels = np.zeros((0, 1), dtype=np.int32)
        self.__inst_labels_color = np.zeros((0, 3), dtype=np.uint8)

        # projected semantic labels
        self.__proj_sem_labels = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)
        self.__proj_sem_labels_color = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)

        # projected instance labels
        self.__proj_inst_labels = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)
        self.__proj_inst_labels_colors = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)

    def __check_proj_input_shape(self, array, nb_channels=1):
        """
        Parameters
        ----------
        array: np.ndarray

        nb_channels: int
        """

        nr, nc = array.shape[:2]

        if nr != self.proj_h or nc != self.proj_w:
            raise ValueError("Input array shape was expected to be {}x{},"
                             " but it is {}x{}".format(self.proj_h, self.proj_w, nr, nc))

        if nb_channels > 1:
            if array.shape[2] != nb_channels:
                raise ValueError("Input array was expected to have {} channels,"
                                 " but it has {} channels.".format(nb_channels, array.shape[2]))

    @property
    def sem_labels(self):
        return self.__sem_labels

    @sem_labels.setter
    def sem_labels(self, sem_labels):
        self.__sem_labels = sem_labels

    @property
    def sem_labels_color(self):
        return self.__sem_labels_color

    @sem_labels_color.setter
    def sem_labels_color(self, sem_labels_color):
        self.__sem_labels_color = sem_labels_color

    @property
    def inst_labels(self):
        return self.__inst_labels

    @inst_labels.setter
    def inst_labels(self, inst_labels):
        self.__inst_labels = inst_labels

    @property
    def inst_labels_color(self):
        return self.__inst_labels_color

    @inst_labels_color.setter
    def inst_labels_color(self, inst_labels_color):
        """
        Parameters
        ----------
        inst_labels_color: np.ndarray
        """
        nr, nc = inst_labels_color.shape
        if nc != 3:
            raise ValueError("Color array must contain 3 channels")
        self.__inst_labels_color = inst_labels_color

    @property
    def proj_sem_labels(self):
        return self.__proj_sem_labels

    @proj_sem_labels.setter
    def proj_sem_labels(self, proj_sem_labels):
        self.__check_proj_input_shape(proj_sem_labels)
        self.__proj_sem_labels = proj_sem_labels

    @property
    def proj_sem_labels_color(self):
        return self.__proj_sem_labels_color

    @proj_sem_labels_color.setter
    def proj_sem_labels_color(self, proj_sem_labels_color):
        self.__check_proj_input_shape(proj_sem_labels_color, nb_channels=3)
        self.__proj_sem_labels_color = proj_sem_labels_color

    @property
    def proj_inst_labels(self):
        return self.__proj_inst_labels

    @proj_inst_labels.setter
    def proj_inst_labels(self, proj_inst_labels):
        self.__check_proj_input_shape(proj_inst_labels)
        self.__proj_inst_labels = proj_inst_labels

    @property
    def proj_inst_labels_colors(self):
        return self.__proj_inst_labels_colors

    @proj_inst_labels_colors.setter
    def proj_inst_labels_colors(self, proj_inst_labels_colors):
        self.__check_proj_input_shape(proj_inst_labels_colors, nb_channels=3)
        self.__proj_inst_labels_colors = proj_inst_labels_colors

    def reset(self):
        """Reset scan members"""
        # super(SemanticKittiLaserScan, self).reset()
        super().reset()

        # semantic labels
        self.sem_labels = np.zeros((0, 1), dtype=np.int32)
        self.sem_labels_color = np.zeros((0, 3), dtype=np.uint8)

        # instance labels
        self.inst_labels = np.zeros((0, 1), dtype=np.int32)
        self.inst_labels_color = np.zeros((0, 3), dtype=np.uint8)

        # projected semantic labels
        self.proj_sem_labels = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)
        self.proj_sem_labels_color = np.zeros((self.proj_h, self.proj_w, 3), dtype=np.int32)

        # projected instance labels
        self.proj_inst_labels = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)
        self.proj_inst_labels_colors = np.zeros((self.proj_h, self.proj_w, 3), dtype=np.int32)

    def open_label(self, filename):
        """ Open raw scan and fill in attributes
        """
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))

        # set it
        self.set_label(label)

    def set_label(self, label):
        """ Set points for label not from file but from np
        """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_labels = label & 0xFFFF
            self.inst_labels = label >> 16
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert ((self.sem_labels + (self.inst_labels << 16) == label).all())

        if self.projection is not None:
            self.do_label_projection()

    def colorize(self):
        self.sem_labels_color = self.skconfig.label2color[self.sem_labels]
        self.inst_labels_color = self.skconfig.label2color[self.inst_labels]

    def do_label_projection(self):

        rho = np.linalg.norm(self.points, axis=1)

        aggr = ['min', 'argmin0', 'argmin0']

        img = self.projection.project_points_values(self.points,
                                                    np.c_[rho, self.sem_labels, self.inst_labels],
                                                    aggregate_func=aggr)

        self.proj_sem_labels = img[:, :, 1].astype(np.int32)
        self.proj_inst_labels = img[:, :, 2].astype(np.int32)

        if self.skconfig is not None:
            self.proj_sem_labels_color = self.skconfig.label2color[self.proj_sem_labels]
            # self.proj_inst_labels_colors = self.skconfig.label2color[self.proj_inst_labels]
