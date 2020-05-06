from abc import ABC, abstractmethod
import numpy as np
import re

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

        min_val = points.min(0)
        xmin = min_val[0]
        ymin = min_val[1]
        x = points[:, 0] - xmin
        y = points[:, 1] - ymin

        i_img_mapping = np.floor(x * self.res_x).astype(int)
        j_img_mapping = np.floor(y * self.res_y).astype(int)

        lidx = ((i_img_mapping) % height) * width + j_img_mapping

        return lidx, i_img_mapping, j_img_mapping

    def get_image_size(self, **kwargs):
        """
        Return the image size
        :param kwargs:
        :return:
        """
        points = kwargs['points']
        max_val = points.max(0)
        min_val = points.min(0)
        height = np.ceil((max_val[0] - min_val[0]) * self.res_x).astype(int)
        width  = np.ceil((max_val[1] - min_val[1]) * self.res_y).astype(int)

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
        fov_height = np.abs(self.fov_pitch[1] - self.fov_pitch[0])
        fov_width = np.abs(self.fov_yaw[1] - self.fov_yaw[0])
        # height = np.ceil(fov_height * self.res_pitch).astype(int)
        # width = np.ceil(fov_width * self.res_yaw).astype(int)
        height = self.res_pitch
        width = self.res_yaw

        return height, width

class LayerProjector(AbstractProjector):
    def __init__(self):
        super(LayerProjector, self).__init__()

    def project_point(self, points):
        pass

    def get_image_size(self, **kwargs):
        pass


class Projection:
    def __init__(self, proj_type, res_x=0.0, res_y=0.0, res_pitch=0.0, res_yaw=0.0, fov_pitch = None, fov_yaw = None):
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

        """

        self.res_x = res_x
        self.res_y = res_y
        self.res_pitch = res_pitch
        self.res_yaw = res_yaw

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
            pass

        else:
            raise ValueError("proj_type value can be only 'linear', 'spherical' or 'layers',  "
                             "you passed {}".format(proj_type))


    def __initialize_linear_proj(self, res_x, res_y):
        return LinearProjector(res_x=res_x, res_y=res_y)

    def __initialize_spherical_proj(self, res_yaw, res_pitch, fov_yaw, fov_pitch):
        return SphericalProjector(res_yaw=res_yaw, res_pitch=res_pitch, fov_yaw=fov_yaw, fov_pitch=fov_pitch)

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


        Returns
        -------
        proj_img: ndarray
            Image containing projected values
        """

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
