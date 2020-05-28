from abc import ABC, abstractmethod
import numpy as np
import re
from smutsia.utils.semantickitti import retrieve_layers
import smilPython as sm
from smutsia.utils.image import smil_2_np, np_2_smil


class AbstractProjector(ABC):
    """
    Abstract class that represent a general projector
    """
    def __init__(self):
        super(AbstractProjector, self).__init__()

    @abstractmethod
    def project_point(self, points):
        pass

    @abstractmethod
    def get_image_size(self, **kwargs):
        pass


class LinearProjector(AbstractProjector):

    def __init__(self, res_x, res_y):
        """

        Parameters
        ----------
        res_x: px / mt
        res_y: px / mt
        """
        self.res_x = res_x
        self.res_y = res_y
        super(LinearProjector, self).__init__()

    def project_point(self, points):

        if len(points.shape) < 2:
            points = np.atleast_2d(points)

        height, width = self.get_image_size(points=points)

        xmin, ymin, _, _ = self.get_bounding_xy(points)

        x = points[:, 0] - xmin
        y = points[:, 1] - ymin

        i_img_mapping = np.floor(x * self.res_x).astype(int)
        j_img_mapping = np.floor(y * self.res_y).astype(int)

        lidx = (i_img_mapping % height) * width + j_img_mapping

        return lidx, i_img_mapping, j_img_mapping

    @staticmethod
    def get_bounding_xy(points):
        """
        Auxiliary method that return bounding box in the xy plane

        Parameters
        ----------
        points: ndarray
            input point cloud containing euclidean coordinates of the input point cloud

        Returns
        -------
        min_X: float
            min x-value
        min_Y: float
            min y-value
        max_X: float
            max x-value
        max_Y: float
            max y-value
        """
        max_val = points.max(0)
        min_val = points.min(0)
        # changes proposed by BEA
        min_X = min_val[0]
        min_Y = min_val[1]
        max_X = max_val[0]
        max_Y = max_val[1]
        # inv_res_x = 1.0 / self.res_x
        # inv_res_y = 1.0 / self.res_y
        # min_X = np.floor(min_val[0] / inv_res_x) * inv_res_x
        # min_Y = np.floor(min_val[1] / inv_res_y) * inv_res_y

        return min_X, min_Y, max_X, max_Y

    def get_image_size(self, **kwargs):
        """
        Return the image size
        :param kwargs:
        :return:
        """
        points = kwargs['points']

        min_X, min_Y, max_X, max_Y = self.get_bounding_xy(points)
        height = np.ceil((max_X - min_X) * self.res_x).astype(int)
        width = np.ceil((max_Y - min_Y) * self.res_y).astype(int)

        return height, width


class SphericalProjector(AbstractProjector):

    def __init__(self, res_yaw, res_pitch, fov_yaw=None, fov_pitch=None):
        self.res_yaw = res_yaw
        self.res_pitch = res_pitch
        if fov_yaw is None:
            fov_yaw = [0.0, 2 * np.pi]

        if fov_pitch is None:
            fov_pitch = [0.0, np.pi]

        self.fov_yaw = fov_yaw
        self.fov_pitch = fov_pitch

        super(SphericalProjector, self).__init__()

    def filter_points(self, yaw, pitch):
        idx = np.ones_like(yaw, dtype=np.bool)

        idx = np.logical_and(idx, yaw <= self.fov_yaw[1])
        idx = np.logical_and(idx, yaw >= self.fov_yaw[0])
        idx = np.logical_and(idx, pitch <= self.fov_pitch[1])
        idx = np.logical_and(idx, pitch >= self.fov_pitch[0])
        return (1 - idx).astype(bool)

    def project_point(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        height, width = self.get_image_size()
        rho = np.linalg.norm(points, axis=1)
        yaw = np.arctan2(y, x) + np.pi
        pitch = np.arccos(z / rho)

        idx = self.filter_points(yaw=yaw, pitch=pitch)
        # get projections in image coords
        i_img_mapping = (pitch - min(self.fov_pitch)) / np.abs(self.fov_pitch[1] - self.fov_pitch[0])  # in [0.0, 1.0]
        j_img_mapping = (yaw - min(self.fov_yaw)) / np.abs(self.fov_yaw[1] - self.fov_yaw[0])  # in [0.0, 1.0]

        # scale to image size using angular resolution
        i_img_mapping *= height  # in [0.0, H]
        j_img_mapping *= width  # in [0.0, W]
        # i_img_mapping *=
        # round and clamp for use as index
        i_img_mapping = np.floor(i_img_mapping)
        i_img_mapping = np.minimum(height - 1, i_img_mapping)
        i_img_mapping = np.maximum(0, i_img_mapping).astype(np.int32)  # in [0,H-1]

        j_img_mapping = np.floor(j_img_mapping)
        j_img_mapping = np.minimum(width - 1, j_img_mapping)
        j_img_mapping = np.maximum(0, j_img_mapping).astype(np.int32)  # in [0,W-1]

        lidx = (i_img_mapping * width) + j_img_mapping

        lidx[idx] = -1
        i_img_mapping[idx] = -1
        j_img_mapping[idx] = -1

        return lidx, i_img_mapping, j_img_mapping

    def get_image_size(self, **kwargs):
        """
        Function that return the size of the projection image

        Parameters
        ----------
        kwargs: dict

        Returns
        -------
        height: int
            height of the proj image

        width: int
            width of the proj image
        """
        # fov_height = np.abs(self.fov_pitch[1] - self.fov_pitch[0])
        # fov_width = np.abs(self.fov_yaw[1] - self.fov_yaw[0])
        # height = np.ceil(fov_height * self.res_pitch).astype(int)
        # width = np.ceil(fov_width * self.res_yaw).astype(int)
        height = self.res_pitch
        width = self.res_yaw

        return height, width


class LayerProjector(AbstractProjector):
    def __init__(self, height=64, width=1024):
        self.proj_H = height
        self.proj_W = width
        super(AbstractProjector, self).__init__()

    def project_point(self, points):
        dims = points.shape[1]
        if dims <= 3:
            layers = retrieve_layers(points, max_layers=self.proj_H)
        else:
            layers = points[:, -1]
        # x coordinates
        x = points[:, 0]
        # y coordinates
        y = points[:, 1]

        # we need to invert orientation of y coordinates
        yaw = np.arctan2(-y, x)
        i_img_mapping = layers.astype(np.int)
        j_img_mapping = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        j_img_mapping *= self.proj_W  # in [0.0, W]

        j_img_mapping = np.floor(j_img_mapping)
        j_img_mapping = np.minimum(self.proj_W - 1, j_img_mapping)
        j_img_mapping = np.maximum(0, j_img_mapping).astype(np.int32)  # in [0,W-1]

        lidx = (i_img_mapping * self.proj_W) + j_img_mapping

        return lidx, i_img_mapping, j_img_mapping

    def get_image_size(self, **kwargs):
        return self.proj_H, self.proj_W


class Projection:
    def __init__(self,
                 proj_type,
                 res_x=0.0,
                 res_y=0.0,
                 res_pitch=0.0,
                 res_yaw=0.0,
                 fov_pitch=None,
                 fov_yaw=None,
                 nb_layers=None):
        """
        Projection class

        Parameters
        ----------
        proj_type: optional {'linear', 'spherical', 'layers'}

        res_x: float
            resolution along the rows of the image
        res_y: float
            resolution along the columns of the image

        res_pitch: float
            resolution along the pitch axis (to use in the spherical projection)

        res_yaw: float
            resolution along the yaw axis (to use in spherical projection)

        fov_pitch: list
            field of view interval containing min/max pitch angles

        fov_yaw: list
            field of view interval containing min/max yaw angles

        nb_layers: int
            number of layers in the scanner (to use in the layer projection)
        """
        # TODO: refactor parameters names
        self.res_x = res_x
        self.res_y = res_y
        self.res_pitch = res_pitch
        self.res_yaw = res_yaw
        self.nb_layers = nb_layers

        if fov_yaw is None:
            fov_yaw = [0.0, 2 * np.pi]

        if fov_pitch is None:
            fov_pitch = [0.0, np.pi]

        if proj_type == 'linear':
            self.projector = self.__initialize_linear_proj(res_x=res_x, res_y=res_y)
        elif proj_type == 'spherical':
            self.projector = self.__initialize_spherical_proj(res_yaw=res_yaw,
                                                              res_pitch=res_pitch,
                                                              fov_yaw=fov_yaw,
                                                              fov_pitch=fov_pitch)
        elif proj_type == 'layers':
            self.projector = self.__initialize_layer_proj(res_yaw=res_yaw, nb_layers=nb_layers)

        else:
            raise ValueError("proj_type value can be only 'linear', 'spherical' or 'layers',  "
                             "you passed {}".format(proj_type))

    def __initialize_linear_proj(self, res_x, res_y):
        return LinearProjector(res_x=res_x, res_y=res_y)

    def __initialize_spherical_proj(self, res_yaw, res_pitch, fov_yaw, fov_pitch):
        return SphericalProjector(res_yaw=res_yaw, res_pitch=res_pitch, fov_yaw=fov_yaw, fov_pitch=fov_pitch)

    def __initialize_layer_proj(self, res_yaw, nb_layers):
        return LayerProjector(height=nb_layers, width=res_yaw)

    def project_points_values(self, points, values, aggregate_func='max', rot=np.eye(3), b=0.0):
        """
        Function that project an array of values to an image

        Parameters
        ----------
        points: ndarray
            Array containing the point cloud

        values: ndarray
            Array containing the values to project

        aggregate_func: optional {'max', 'min', 'mean'}
            Function to use to aggregate the information in case of collision, i.e. when two or more points
            are projected to the same pixel.
            'max': take the maximum value among all the values projected to the same pixel
            'min': take the minimum value among all the values projected to the same pixel
            'mean': take the mean value among all the values projected to the same pixel

        rot: ndarray
            rigid transformation matrix to apply to input point cloud

        b: ndarray
            3 dimensional vector to shift input point cloud
        Returns
        -------
        proj_img: ndarray
            Image containing projected values
        """
        # verify that rot is a 3x3 matrix
        assert rot.shape[0] == 3
        assert rot.shape[1] == 3

        rot_points = np.dot(rot, points.T).T + b

        nr, nc = self.projector.get_image_size(points=rot_points)
        if len(values.shape) < 2:
            channel_shape = 1
            values = np.atleast_2d(values).T
        else:
            _, channel_shape = values.shape[:2]

        if channel_shape > 1:
            if type(aggregate_func) is str:
                aggregators = [aggregate_func] * channel_shape
            else:
                assert len(aggregate_func) == channel_shape
                aggregators = aggregate_func
        else:
            aggregators = [aggregate_func]
        # we verify that the length of the two arrays is the same
        # that is for each point we have a corresponding value to project
        assert len(rot_points) == len(values)
        # project points to image
        lidx, i_img_mapping, j_img_mapping = self.projector.project_point(rot_points)

        # initialize binned_feature map
        binned_values = np.zeros((nr * nc, values.shape[1]))

        if 'max' in aggregators or 'min' in aggregators:
            # auxiliary variables to compute minimum and maximum
            sidx = lidx.argsort()
            idx = lidx[sidx]
            # we select the indices of the first time a unique value in lidx appears
            # np.r_[True, idx[:1] != idx[1:]] is true if an element in idx is different than its successive
            # flat non zeros returns indices of values that are non zeros in the array
            m_idx = np.flatnonzero(np.r_[True, idx[:-1] != idx[1:]])

            unq_ids = idx[m_idx]

        if 'mean' in aggregators:
            # auxiliary vector to compute binned count
            count_input = np.ones_like(values[:, 0])
            binned_count = np.bincount(lidx, count_input, minlength=nr * nc)

        for i, func in zip(range(values.shape[1]), aggregators):

            if func == 'max':
                """
                Examples
                --------
                To take the running sum of four successive values:

                >>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
                array([ 6, 10, 14, 18])

                """
                binned_values[unq_ids, i] = np.maximum.reduceat(values[sidx, i], m_idx)

            elif func == 'min':
                binned_values[unq_ids, i] = np.minimum.reduceat(values[sidx, i], m_idx)

            elif func == 'sum':
                binned_values[:, i] = np.bincount(lidx, values[:, i], minlength=nr * nc)

            elif 'argmin' in func:
                arg_i = int(re.split(r"\D+", func)[1])
                argvalues = np.zeros_like(m_idx)
                # we need to reorder the values first
                out_val = values[sidx, i]
                arg_val = values[sidx, arg_i]
                for n, (start, end) in enumerate(zip(m_idx[:-1], m_idx[1:])):
                    argvalues[n] = out_val[start + arg_val[start:end].argmin()]
                # assign argvalues to binned values
                binned_values[unq_ids, i] = argvalues

            elif 'argmax' in func:
                arg_i = int(re.split(r"\D+", func)[1])
                argvalues = np.zeros_like(m_idx)
                # we need to reorder the values first
                out_val = values[sidx, i]
                arg_val = values[sidx, arg_i]
                for n, (start, end) in enumerate(zip(m_idx[:-1], m_idx[1:])):
                    argvalues[n] = out_val[start + arg_val[start:end].argmax()]
                binned_values[unq_ids, i] = argvalues

            else:  # otherwise we compute mean values
                binned_values[:, i] = np.bincount(lidx, values[:, i], minlength=nr * nc)
                binned_values[:, i] = np.divide(binned_values[:, i], binned_count, out=np.zeros_like(binned_count),
                                                where=binned_count != 0.0)

        # reshape binned_features to feature map
        binned_values_map = binned_values.reshape((nr, nc, values.shape[1]))

        if channel_shape == 1:
            binned_values_map = binned_values_map[:, :, 0]

        return binned_values_map


def project_img(projector, points, labels, res_z, min_z):
    """
    Parameters
    ----------
    projector: Projection
        Projection class

    points: ndarray
        input point cloud

    labels: ndarray
        array of labels

    res_z: float
        z resolution

    min_z: float
        minimum z value accepted
    """
    z = points[:, 2]
    # min_z = np.percentile(z, percent)
    # min_z = find_min_z(z, 0.2, 5)
    moved_z = z - min_z
    moved_z = np.clip(moved_z, a_min=0, a_max=moved_z.max())
    idx = np.where(z < min_z)

    np_z = (np.floor(moved_z * res_z) + 1).astype(int)
    mymax = np.amax(np_z) + 1
    np_z_min = np_z.copy()
    np_z_min[idx] = mymax

    values = np.c_[np_z_min, np_z, np.ones_like(z), labels]
    aggregators = ['min', 'max', 'sum', 'argmax1']
    img = projector.project_points_values(points, values, aggregate_func=aggregators)
    np_min = img[:, :, 0]
    np_min[np_min == mymax] = 1
    im_min = np_2_smil(np_min)
    im_max = np_2_smil(img[:, :, 1])
    im_acc = np_2_smil(np.clip(img[:, :, 2], 0, 255))

    im_class = np_2_smil(img[:, :, 3])

    sm.compare(im_acc, "==", 0, 0, im_class, im_class)

    return im_min, im_max, im_acc, im_class


def back_projection(proj, points, imres, pred_labels=None):
    np_labels = smil_2_np(imres)

    lidx, i_img_mapping, j_img_mapping = proj.projector.project_point(points)

    if pred_labels is None:
        pred_labels = np.zeros(len(i_img_mapping))
    for n in range(len(i_img_mapping)):
        coor_i = i_img_mapping[n]
        coor_j = j_img_mapping[n]

        my_lab = np_labels[coor_i, coor_j]
        if my_lab > 0:
            pred_labels[n] = np_labels[coor_i, coor_j]

    return pred_labels


def back_projection_ground(proj, points, res_z, im_min, im_ground, im_delta, delta_ground, min_z):
    """
    Parameters
    ----------
    proj: Projection

    points: ndarray

    res_z: float

    im_min: sm.Image

    im_ground: sm.Image

    im_delta: sm.Image

    delta_ground: float

    min_z: float

    Returns
    -------

    pred_labels: ndarray
    """
    # Le calcul de npZ (echelle image) a deja ete fait. Voir si on peut le recuperer...
    p_z = points[:, 2]
    # z_min = np.percentile(p_z, percent)
    # min_z = find_min_z(p_z, 0.2, 5)

    moved_z = p_z - min_z

    moved_z = np.clip(moved_z, a_min=0, a_max=np.max(moved_z))
    np_z = (np.floor(moved_z * res_z) + 1).astype(int)

    imtmp = sm.Image(im_min)
    mymax = im_min.getDataTypeMax()
    # min on ground, 255 elsewhere
    sm.compare(im_ground, ">", 0, im_min, mymax, imtmp)
    p_mntz = back_projection(proj, points, imtmp)
    p_dsmz = np_z - p_mntz

    delta = delta_ground * res_z

    p_delta = back_projection(proj, points, im_delta)
    p_delta = p_delta * delta
    # pixel labelled as ground (<mymax), and point not too far (deltaGround) from min & (predLabels != carId)
    idx = ((p_mntz < mymax) & (p_dsmz <= p_delta))

    pred_labels = np.zeros_like(p_z, dtype=np.bool)

    pred_labels[idx] = True

    return pred_labels

