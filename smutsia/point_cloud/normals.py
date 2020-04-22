import numpy as np
from pyntcloud import PyntCloud
from skimage.morphology import closing, square

# utils functions returning cos and sinus for yaw and pitch angle
def _cos_pitch_grid(shape, pitch):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = cos(i/res_az)


    Parameters
    ----------
    shape: tuple
        image dimension

    pitch: float
        pitch angles

    Returns
    -------
    cos_grid: ndarray
        Grid containing cos of the vertical angles
    """
    pitch_grid = np.repeat(pitch, shape[1]).reshape(shape)
    return np.cos(pitch_grid)


def _sin_pitch_grid(shape, pitch):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = sin(i/res_az)


    Parameters
    ----------
    shape: tuple
        image dimension

    pitch: ndarray
        array of pitch angles

    Returns
    -------
    sin_grid: ndarray
        Grid containing sin of the vertical angles
    """
    pitch_grid = np.repeat(pitch, shape[1]).reshape(shape)
    return np.sin(pitch_grid)


def _sin_yaw_grid(shape, yaw):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = sin(j/res_planar + np.pi)

    Parameters
    ----------
    shape: tuple
        image dimension

    yaw: ndarray
        array of yaw angles

    Returns
    -------
    sin_grid: ndarray
        Grid containing cosinus of the planar angles
    """

    yaw_grid = np.repeat(yaw, shape[0]).reshape(shape, order='F')
    return np.sin(yaw_grid)


def _cos_yaw_grid(shape, yaw):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = cos(j/res_planar + np.pi)

    Parameters
    ----------
    shape: tuple
        image dimension

    yaw: ndarray
        array of yay angles

    Returns
    -------

    cos_grid: ndarray
        Grid containing cos of the planar angles
    """
    yaw_grid = np.repeat(yaw, shape[0]).reshape(shape, order='F')
    return np.cos(yaw_grid)


def shift(key, array, axis=0):
    """
    Utils function that shift elements of a vector

    Parameters
    ----------

    key: int
        number of col/rows to shift

    array: ndarray
        Array/matrix to shift

    axis: {0,1} optional
        0: to shift axis 0 of the array
        1: to shift axis 1 of the array

    Returns
    -------
    shifted_array: ndarray
        Array shifted of key columns/rows
    """

    shape = array.shape
    shifted_array = np.zeros(shape, dtype=array.dtype)

    if axis == 0:
        if key >= 0:
            shifted_array[key:] = array[:shape[0] - key]
        else:
            shifted_array[:shape[0] + key] = array[-key:]
    elif axis == 1:
        if key >= 0:
            shifted_array[:, key:] = array[:, :shape[1] - key]
        else:
            shifted_array[:, :shape[1] + key] = array[:, -key:]
    else:
        raise ValueError('Method only implemented for shift in the 0 or 1 axis')

    return shifted_array


def shift_on_a_torus(key, array, axis=0):
    """
    Utils function that shift on a 2 dimensional thorus, i.e. we identify as equals the top and bottom frontier
    of image and the left and right frontier of image.

    Parameters
    ----------
    key: int
        number of col/rows to shift

    array: ndarray
        Array/matrix to shift

    axis: {0,1} optional
        0: to shift axis 0 of the array
        1: to shift axis 1 of the array

    Returns
    -------
    shifted_array: ndarray
        Array shifted of key columns/rows

    """
    shape = array.shape
    if axis == 0:
        # remember positive values of key shift array toward above and negative toward below
        return np.concatenate([array[key % shape[axis]:], array[:key % shape[axis]]], axis=axis)

    if axis == 1:
        # remember positive values of key shift array toward left and negative toward right
        return np.concatenate([array[:, key % shape[axis]:], array[:, :key % shape[axis]]], axis=axis)
    else:
        raise ValueError('Method only implemented for shift in the 0 or 1 axis')


def _pitch_rho_derivative(img, res_rho):
    """
    Function that compute the derivative of the rho coordinate along vertical axis
    Parameters
    ----------
    img: ndarray
        Input image

    res_rho: float
        Resolution of the rho coordinate

    Returns
    -------
    az_rho_derivative:  ndarray
        image of the derivative of rho coordinate along vertical axis

    """
    above = shift_on_a_torus(-1, img)
    below = shift_on_a_torus(1, img)
    pitch_deriv = (below.astype(np.float64) - above.astype(np.float64)) / (2 * res_rho)
    pitch_deriv[0] = (img[0].astype(np.float64) - img[1].astype(np.float64)) / res_rho
    pitch_deriv[-1] = (img[-1].astype(np.float64) - img[-2].astype(np.float64)) / res_rho
    return pitch_deriv


def _yaw_rho_derivative(img, res_rho):
    """
    Function that compute the derivative of the rho coordinate along vertical axis

    Parameters
    ----------
    img: ndarray
        Input image

    res_rho: float
        Resolution of the rho coordinate

    Returns
    -------
    yaw_rho_derivative:  ndarray
        image of the derivative of rho coordinate along vertical axis

    """
    right = shift_on_a_torus(-1, img, axis=1)
    left = shift_on_a_torus(1, img, axis=1)
    return (left.astype(np.float64) - right.astype(np.float64)) / (2 * res_rho)


def pitch_derivative(img, pitch, yaw, res_rho):
    """

    Parameters
    ----------
    img: ndarray
        input image

    pitch: ndarray
        array of pitch angles

    yaw: ndarray
        array of yaw angles

    res_rho: float
        Resolution of the rho coordinate

    Returns
    -------
    _pitch_derivative_: ndarray
        pitch derivative of the input img
    """
    # getting pitch_rho_derivative
    diff_rho = _pitch_rho_derivative(img, res_rho)

    nr, nc = img.shape[:2]
    # auxiliary cos and sin grid
    cos_pitch = _cos_pitch_grid((nr, nc), pitch)
    sin_pitch = _sin_pitch_grid((nr, nc), pitch)
    cos_yaw = _cos_yaw_grid((nr, nc), yaw)
    sin_yaw = _sin_yaw_grid((nr, nc), yaw)

    rho = img.copy()
    rho = rho.astype(np.float64) / res_rho

    # initialize img to return
    _pitch_derivative_ = np.zeros((nr, nc, 3))

    res_pitch = (shift(-1, pitch) - shift(1, pitch)) / 2
    res_pitch[0] = res_pitch[1]
    res_pitch[-1] = res_pitch[-2]

    res_pitch = np.repeat(res_pitch, nc).reshape(nr, nc)

    # derivative along x
    _pitch_derivative_[:, :, 0] = diff_rho * sin_pitch * cos_yaw + (rho * res_pitch) * cos_pitch * cos_yaw
    # derivative along y
    _pitch_derivative_[:, :, 1] = diff_rho * sin_pitch * sin_yaw + (rho * res_pitch) * cos_pitch * sin_yaw
    # derivative along z
    _pitch_derivative_[:, :, 2] = diff_rho * cos_pitch - (rho * res_pitch) * sin_pitch

    return _pitch_derivative_


def yaw_derivative(img, pitch, yaw, res_rho):
    """

    Parameters
    ----------
    img: ndarray
        input image

    pitch: ndarray
         array of pitch angles

    yaw: ndarray
         array of yaw angles

    res_rho: float
        Resolution of the rho coordinate

    Returns
    -------
    _yaw_derivative_: ndarray
        derivative of the input image along the planar coordinate
    """
    # getting yaw_rho_derivative
    diff_rho = _yaw_rho_derivative(img, res_rho)

    nr, nc = img.shape[:2]
    # auxiliary cos and sin grid
    cos_pitch = _cos_pitch_grid((nr, nc), pitch)
    sin_pitch = _sin_pitch_grid((nr, nc), pitch)
    cos_yaw = _cos_yaw_grid((nr, nc), yaw)
    sin_yaw = _sin_yaw_grid((nr, nc), yaw)

    rho = img.copy()
    rho = rho.astype(np.float64) / res_rho

    res_yaw = (shift(-1, yaw) - shift(1, yaw)) / 2
    res_yaw[0] = res_yaw[1]
    res_yaw[-1] = res_yaw[-2]

    res_yaw = np.repeat(res_yaw, nr).reshape((nr, nc), order='F')

    # initialize img to return
    _yaw_derivative_ = np.zeros((nr, nc, 3))

    # derivative along x
    _yaw_derivative_[:, :, 0] = diff_rho * sin_pitch * cos_yaw - rho * res_yaw * sin_pitch * sin_yaw
    # derivative along y
    _yaw_derivative_[:, :, 1] = diff_rho * sin_pitch * sin_yaw + rho * res_yaw * sin_pitch * cos_yaw
    # derivative along z
    _yaw_derivative_[:, :, 2] = diff_rho * cos_pitch

    return _yaw_derivative_


def estimate_normals_from_spherical_img(img, pitch, yaw, res_rho):
    """
    Function that estimates point cloud normals from a spherical img.
    It returns a nr x nc x 3 image such that for each pixel we have the estimated normal

    Parameters
    ----------
    img: ndarray
        input image

    pitch: ndarray
        azimuthal angles

    res_rho: float
        Resolution of coordinate rho

    yaw: ndarray
        resolution of coordinate planar


    Returns
    -------
    vr: ndarray
        matrix that at position i,j contains the normal inferred for pixel i,j
    """
    if yaw is None:
        yaw = np.linspace(0, 2*np.pi, len(img.shape[1]))

    if pitch is None:
        max_pitch = 115.0 * (180.0 / np.pi)
        min_pitch = 85.0 * (180.0 / np.pi)
        pitch = np.linspace(min_pitch, max_pitch, img.shape[0])

    pitch_deriv = pitch_derivative(img, pitch=pitch, yaw=yaw, res_rho=res_rho)
    yaw_deriv = yaw_derivative(img, pitch=pitch, yaw=yaw, res_rho=res_rho)

    vr = np.cross(pitch_deriv, yaw_deriv)
    vr_lenght = np.linalg.norm(vr, axis=2)
    idx = vr_lenght > 0

    for i in range(3):
        vr[idx, i] = np.divide(vr[idx, i], vr_lenght[idx])

    # we remove normals where we do not have information
    vr[img == 0] = [0, 0, 0]

    return vr


def get_normals(cloud, method='spherical', **kwargs):
    """

    Parameters
    ----------
    cloud: PyntCloud
        input point cloud
    method: optional {'spherical', 'pca'}
        method to use to estimate normals
    kwargs: dict
        optional paramters

    Returns
    -------
    normals: ndarray
        estimated normals
    """
    from smutsia.point_cloud.projection import Projection
    if method.lower() == 'spherical':
        pitch = kwargs.get("pitch", [85.0 * np.pi / 180.0, 115.0 * np.pi / 180.0])
        yaw = kwargs.get("yaw", [0, 2 * np.pi])
        res_pitch = kwargs.get("res_pitch", 64)
        res_yaw = kwargs.get("res_yaw", 2048)

        # initialise projector
        proj = Projection(proj_type='spherical', res_yaw=res_yaw, res_pitch=res_pitch, fov_yaw=yaw, fov_pitch=pitch)
        # project rho values
        rho_img = proj.project_points_values(cloud.xyz, np.linalg.norm(cloud.xyz, axis=1), aggregate_func='max')
        cl_rho = closing(rho_img, square(3))

        rho_img[rho_img == 0] = cl_rho[rho_img == 0]
        yaw_angles = np.linspace(yaw[0], yaw[1], res_yaw)
        pitch_angles = np.linspace(pitch[0], pitch[1], res_pitch)

        return estimate_normals_from_spherical_img(rho_img, pitch=pitch_angles, yaw=yaw_angles, res_rho=1.0)

    elif method.lower() == 'pca':
        k = kwargs.get("k", 10)
        kneighs = cloud.get_neighbors(k=k)
        cloud.add_scalar_field("normals", k_neighbors=kneighs)
        nx = cloud.points['nx(' + str(k + 1) + ')']
        ny = cloud.points['ny(' + str(k + 1) + ')']
        nz = cloud.points['nz(' + str(k + 1) + ')']
        return np.c_[nx, ny, nz]
    else:
        raise ValueError("Parameter method accept only be 'spherical' or 'pca'. The passed value is {}.".format(method))
