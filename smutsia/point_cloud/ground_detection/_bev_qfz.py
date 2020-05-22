import smilPython as sm
import numpy as np
from pyntcloud import PyntCloud
from smutsia.point_cloud.projection import Projection


class LambdaGDParameters:
    def __init__(self, my_lambda=2, delta_ground=0.2, nl=sm.HexSE()):
        self.my_lambda = my_lambda
        self.delta_ground = delta_ground
        self.nl = nl


def smil_2_np(im):
    """
    Auxiliary function to convert a smil image 2 numpy image

    Parameters
    ----------
    im: sm.Image
        input smil image

    Returns
    -------
    im_array: ndarray
        numpy image
    """
    # get image content
    im_array = im.getNumArray()
    # swap rows with columns
    im_array = np.swapaxes(im_array, 0, 1)
    return im_array


def np_2_smil(np_img):
    """
    Auxiliary function to convert a numpy img to smil image

    Parameters
    ----------
    np_img: ndarray
        input image

    Returns
    -------
    smil_im: sm.Image
        smil image
    """
    np_swap_img = np.swapaxes(np_img, 0, 1)

    im_shape = np_swap_img.shape

    if np_img.dtype == 'uint8':
        smil_im = sm.Image(im_shape[0], im_shape[1])
    elif np_img.dtype == 'uint16':
        temp_img = sm.Image(im_shape[0], im_shape[1])
        smil_im = sm.Image(temp_img, 'UINT16')
    else:
        Warning("Warning: {} copied to uint8".format(np_img.dtype))
        smil_im = sm.Image(im_shape[0], im_shape[1])

    imArray = smil_im.getNumArray()

    imArray[:, :] = np_swap_img

    return smil_im


def project_img(projector, points, labels, res_z, percent=0.5):
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

    percent: float
        percentile to clip z values
    """
    z = points[:, 2]
    min_z = np.percentile(z, percent)
    moved_z = z - min_z
    moved_z = np.clip(moved_z, a_min=0, a_max=moved_z.max())
    np_z = (np.floor(moved_z * res_z) + 1).astype(int)
    values = np.c_[np_z, np_z, np.ones_like(z), labels]
    aggregators = ['min', 'max', 'sum', 'argmax0']
    img = projector.project_points_values(points, values, aggregate_func=aggregators)

    im_min = np_2_smil(img[:, :, 0])
    im_max = np_2_smil(img[:, :, 1])
    im_acc = np_2_smil(img[:, :, 2])
    im_class = np_2_smil(img[:, :, 3])

    sm.compare(im_acc, "==", 0, 0, im_class, im_class)

    return im_min, im_max, im_acc, im_class


def label_with_measure(im, im_val, im_out, measure_str, nl=sm.Morpho.getDefaultSE()):
    # ----------------------------------------
    # Compute Blobs
    # ----------------------------------------
    im_label = sm.Image(im, "UINT16")
    sm.label(im, im_label, nl)
    blobs = sm.computeBlobs(im_label)

    if measure_str == "mean":
        meas_list = sm.measMeanVals(im_val, blobs)
    elif measure_str == "max":
        meas_list = sm.measMaxVals(im_val, blobs)
    elif measure_str == "min":
        meas_list = sm.measMinVals(im_val, blobs)
    elif measure_str == "mode":
        meas_list = sm.measModeVals(im_val, blobs)
    elif measure_str == "median":
        meas_list = sm.measMedianVals(im_val, blobs)
    elif measure_str == "nb":
        meas_list = sm.measNbVals(im_val, blobs)
    elif measure_str == "nbLab":
        meas_list = sm.measNbLabVals(im_val, blobs)
    else:
        raise ValueError("measure_str value {} not valid".format(measure_str))

    my_lut = sm.Map_UINT16_UINT16()
    if measure_str == "mean":
        for lbl in blobs.keys():
            my_lut[lbl] = int(meas_list[lbl][0])
    else:  # min,max...
        for lbl in blobs.keys():
            my_lut[lbl] = int(meas_list[lbl])
    imtmp16 = sm.Image(im_label)

    sm.applyLookup(im_label, my_lut, imtmp16)
    sm.copy(imtmp16, im_out)


def draw_dart(im, points, proj, h_scanner, alpha0, nb_layers=64):
    """im: as an input image just the size is important. As an output
           image it contains the dart board
    x0,y0,hScanner: scanner position
    alpha0: first angle
    res_x,res_y : spatial resolution of input image
    nb_layers

    The output draws a chess board according to the size of the each
    """

    # x0, y0 Je l'utilise avec une image smil...
    y0, x0 = get_scanner_xy(points, proj)

    # 5 pixels / m, 1 px = 20 cm
    res_x = proj.projector.res_x
    # 5 pixels / m , 1 px = 20 cm
    # res_y = proj.projector.res_y

    res_alpha = 26.9 / nb_layers
    radius_index = {}
    for i in range(nb_layers):
        angle = alpha0 - (res_alpha * i)
        angle_rad = ((90-angle) * np.pi)/180.0
        radius = int(np.round(abs(h_scanner * np.tan(angle_rad) * res_x)))

        if radius > (im.getWidth() + im.getHeight()):
            radius_index[i] = max(im.getWidth(), im.getHeight())
        else:
            radius_index[i] = radius

    # for each distance to scanner, get the layer index
    inverse_radius_index = {}
    index = 0

    # get the maximum index falling into the image
    imsize = max(im.getHeight(), im.getWidth())
    while imsize <= radius_index[index]:
        index = index + 1

    # for this index, get the corresponding radius
    # for larger radius assign max_index+1
    r = im.getHeight()+im.getWidth()
    while r > radius_index[index]:
        inverse_radius_index[r] = index + 1
        r = r - 1

    # each r (radius) has its layer number (inverse_radius_index).
    # index0 close to horizontal, the maximum index close to vertical
    while r > 0:
        while r > radius_index[index]:
            inverse_radius_index[r] = index + 1
            r = r - 1
        index = index + 1
        if index == nb_layers:
            break
    # close to the scanner (masked zone)
    while r >= 0:
        inverse_radius_index[r] = nb_layers + 1
        r = r - 1
    im_r = sm.Image(im, "UINT16")
    im_theta = sm.Image(im, "UINT16")
    im_label = sm.Image(im, "UINT16")
    # fill the image with radius and angular sector
    for x in range(im.getWidth()):
        for y in range(im.getHeight()):
            deltax, deltay = x-x0, y-y0
            if deltax == 0 and deltay == 0:
                theta = 0
            else:
                theta = int(np.round(180+(180*np.arctan2(deltay, deltax))/(2*np.pi)))
            r = int(np.sqrt(deltax*deltax+deltay*deltay))

            value = min(255, inverse_radius_index[r])
            im_r.setPixel(x, y, value)
            im_theta.setPixel(x, y, theta)

    # label 2 partitions
    sm.labelWithoutFunctor2Partitions(im_r, im_theta, im_label, sm.CrossSE())

    return im_label


def get_scanner_xy(points, proj):
    """ get x0,y0 coordinates of the scanner location """

    # Find the pixel corresponding to (x=0,y=0)
    res_x = proj.projector.res_x  # 5 pixels / m, 1 px = 20 cm
    res_y = proj.projector.res_y  # 5 pixels / m , 1 px = 20 cm

    min_x, min_y, min_z = points.min(0)

    # the first coordinate is associated to the row coordinate of the image
    y0 = int(np.floor((0 - min_y) * res_y).astype(np.int))
    # the second coordinate is associated to the column coordinate of the image
    x0 = int(np.floor((0 - min_x) * res_x).astype(np.int))

    return x0, y0


def compute_circle(points, proj, im_max, nl):

    # je l'utilise avec une image smil... (y et x inverse expres)
    y0, x0 = get_scanner_xy(points, proj)

    # Get the circle where the scanner is located
    im_mark = sm.Image(im_max)
    im_tmp, im_circle = sm.Image(im_max), sm.Image(im_max)

    im_mark.setPixel(x0, y0, 255)

    # empty pixels
    sm.compare(im_max, "==", 0, 255, 0, im_tmp)

    # get the circle
    sm.build(im_mark, im_tmp, im_circle, nl)

    # Pb circle trop grand (image 900. Restreindre à une fenetre de 10m x 10m
    sm.fill(im_tmp, 0)

    circle_size = int(5.5 * proj.projector.res_x)
    xinit, yinit = x0 - circle_size, y0 - circle_size
    sm.copy(im_circle, xinit, yinit, 2*circle_size, 2*circle_size, im_tmp, xinit, yinit)
    sm.copy(im_tmp, im_circle)

    return im_circle


def dart_interp(points, proj, im, im_interp, nl):
    """ input: points 3D. Required to compute the x0,y0 of the scanner

    im: the image to be interpolated
    imInterp: the output image
    nl: neighborhood

    Each chess board sector takes the value of the pixel inside, but only if it is alone
    """
    # Une classe avec toute cette info serait utile, plutot que de definir ces variables plusieurs fois...

    nb_layers = 64
    alpha0 = 0
    h_scanner = 1.73

    # get chess board ## TODO: define immax
    im_dart = draw_dart(im, points, proj, h_scanner, alpha0, nb_layers)

    mymax = sm.maxVal(im)
    sm.compare(im, "==", 0, mymax + 1, im, im)

    # propagation de la valeur min (!=0) sur toute la cellule
    label_with_measure(im_dart, im, im_interp, "min", nl)
    sm.compare(im_interp, "==", mymax + 1, 0, im_interp, im_interp)  # empty cells have max-value
    sm.compare(im, "==", mymax + 1, 0, im, im)  # mymax+1 -> 0 again

    sm.compare(im, "==", 0, im_interp, im, im_interp)  # only empty pixels are interpolated

    return im_dart


def im_dart_interp(points, proj, im_max, nl):
    im_interp = sm.Image(im_max)

    im_dart = dart_interp(points, proj, im_max, im_interp, nl)

    return im_interp, im_dart


def ground_detection_min_circle(params, points, proj, res_z, im_min, im_max):
    """
    Parameters
    ----------
    params: LambdaGDParameters

    points: ndarray

    proj: Projection

    im_min: sm.Image

    im_max: sm.Image
    """
    my_lambda, nl = params.my_lambda, params.nl
    im_ground = sm.Image(im_min)
    im_circle = compute_circle(points, proj, im_max, nl)
    im_tmp = sm.Image(im_circle)
    sm.dilate(im_circle, im_tmp, nl(4))

    sm.compare(im_tmp, ">", im_circle, im_max, 0, im_tmp)
    histo = sm.histogram(im_tmp)
    del (histo[0])

    hist_keys = histo.keys()
    hist_val = histo.values()

    my_min = 0
    for k in range(len(hist_keys)):
        if hist_val[k] > 0:
            my_min = hist_keys[k]
            break

    sm.threshold(im_tmp, my_min, min(255, my_min + 5), im_ground)

    sm.sub(im_max, im_min, im_tmp)

    # put to zero all the non zero pixels in imGround
    sm.compare(im_tmp, ">", int(np.round(0.3 * res_z)), 0, im_ground, im_ground)
    se_2x2 = sm.StrElt(True, [0, 1, 2, 3])
    sm.open(im_ground, im_ground, se_2x2)

    im_interp, im_dart = im_dart_interp(points, proj, im_max, nl)

    # Lambda flat zones
    im_label = sm.Image(im_interp, "UINT32")
    sm.lambdaLabel(im_interp, my_lambda, im_label, nl)

    label_with_measure(im_label, im_ground, im_ground, "max", nl)

    # empty pixels set to 0 again
    sm.compare(im_max, "==", 0, 0, im_ground, im_ground)

    # evaluate
    # conf_mat, conf_mat_norm = evaluate(im_class, im_ground, selectedId, condensedId)

    return im_ground, im_interp


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


def back_projection_ground(proj, points, res_z, im_min, im_ground, delta_ground, percent=0.5):

    # Le calcul de npZ (echelle image) a deja ete fait. Voir si on peut le recuperer...
    p_z = points[:, 2]
    z_min = np.percentile(p_z, percent)
    moved_z = p_z - z_min
    moved_z = np.clip(moved_z, a_min=0, a_max = np.max(moved_z))
    npZ = (np.floor(moved_z * res_z) + 1).astype(int)

    imtmp = sm.Image(im_min)
    mymax = im_min.getDataTypeMax()
    # min on ground, 255 elsewhere
    sm.compare(im_ground, ">", 0, im_min, mymax, imtmp)
    p_mntz = back_projection(proj, points, imtmp)
    p_dsmz = npZ - p_mntz

    delta = delta_ground * res_z
    # pixel labelled as ground (<mymax), and point not too far (deltaGround) from min
    # & (predLabels != carId)
    idx = ((p_mntz < mymax) & (p_dsmz <= delta))

    pred_labels = np.zeros_like(p_z, dtype=np.bool)

    pred_labels[idx] = True

    return pred_labels


def dart_ground_detection(cloud, threshold, delta_ground, res_x, res_y, res_z):
    """
    Parameters
    ----------
    cloud: PyntCloud

    threshold: int

    delta_ground: float

    res_x: float

    res_y: float

    res_z: float
    """
    params = LambdaGDParameters(my_lambda=threshold, delta_ground=delta_ground)

    points = cloud.xyz
    labels = cloud.points.labels.values.astype(int)
    proj = Projection(proj_type='linear', res_x=res_x, res_y=res_y)
    im_min, im_max, im_acc, im_class = project_img(proj, points=points, labels=labels, res_z=res_z)

    im_ground, im_interp = ground_detection_min_circle(params=params,
                                                       points=points,
                                                       proj=proj,
                                                       res_z=res_z,
                                                       im_min=im_min,
                                                       im_max=im_max)

    ground = back_projection_ground(proj=proj, points=points, res_z=res_z, im_min=im_min, im_ground=im_ground,
                                    delta_ground=delta_ground)
    return ground
