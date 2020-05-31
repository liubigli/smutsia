import os
import smilPython as sm
import numpy as np
from pyntcloud import PyntCloud
from smutsia.point_cloud.projection import Projection, project_img, back_projection_ground
from smutsia.utils.image import np_2_smil, smil_2_np, label_with_measure


class LambdaGDParameters:
    def __init__(self, my_lambda=2, delta_ground=0.2, delta_h_circle=0.5, nl=sm.HexSE()):
        self.my_lambda = my_lambda
        self.delta_ground = delta_ground
        self.delta_h_circle = delta_h_circle
        self.nl = nl


def find_min_z(zL, step):
    # TODO: Ask Bea the reason why this function. Apparently minPercent is not used
    # histogram of zL, step = 0.2. minZ is set to the value over 0
    # with at maximum 5% of points under it.

    mybins = np.arange(np.amin(zL), np.amax(zL), step)
    myhisto = np.histogram(zL, mybins)
    mycount = myhisto[0]
    idx = np.where(mycount > 100)

    minZ = myhisto[1][idx[0][0]]

    return minZ


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
        angle_rad = ((90-angle) * np.pi) / 180.0
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
    r = im.getHeight() + im.getWidth()
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

    im_label = sm.Image(im, "UINT16")

    # Start faster version that generates dart
    # convert the dict to a numpy array
    max_r = max(inverse_radius_index.keys())

    arr_inv_radius = np.zeros(max_r + 1)
    for k in inverse_radius_index.keys():
        arr_inv_radius[k] = inverse_radius_index[k]

    # fill the image with radius and angular sector
    nr, nc = smil_2_np(im).shape
    np_rows = np.repeat(np.arange(nr), nc).reshape(nr, nc)
    np_cols = np.repeat(np.arange(nc), nr).reshape((nr, nc), order='F')
    deltax = np_cols - x0
    deltay = np_rows - y0
    np_theta = np.round(180+(180*np.arctan2(deltay, deltax))/(2*np.pi)).astype(int)
    # smil and numpy have swapped axes
    np_theta[y0, x0] = 0

    np_r = np.sqrt(deltax**2 + deltay**2).astype(int)
    np_r = arr_inv_radius[np_r]

    im_r = np_2_smil(np_r)
    im_theta = np_2_smil(np_theta)

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

    # BMI
    sm.compare(im_interp, "==", mymax + 1, 0, im_interp, im_interp)  # empty cells have max-value
    sm.compare(im, "==", mymax + 1, 0, im, im)  # mymax+1 -> 0 again

    im_obj = sm.Image(im)
    sm.sub(im, im_interp, im_obj)

    sm.compare(im, "==", 0, im_interp, im, im_interp)  # only empty pixels are interpolated

    # return im_chess, imObj
    return im_dart, im_obj


def im_dart_interp(points, proj, im_max, nl):
    im_interp = sm.Image(im_max)

    im_dart, im_obj = dart_interp(points, proj, im_max, im_interp, nl)

    return im_interp, im_dart, im_obj


def ground_detection_min_circle(params, points, proj, res_z, im_min, im_max):
    """
    Parameters
    ----------
    params: LambdaGDParameters

    points: ndarray

    proj: Projection

    res_z: float

    im_min: sm.Image

    im_max: sm.Image
    """
    my_lambda, nl = params.my_lambda, params.nl
    im_ground = sm.Image(im_min)
    im_circle = compute_circle(points, proj, im_max, nl)
    im_tmp = sm.Image(im_circle)
    # NEW:
    sm.dilate(im_circle, im_tmp, nl(1 * proj.res_x))
    # OLD: sm.dilate(im_circle, im_tmp, nl(4))

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

    # NEW:
    delta = int(params.delta_h_circle * res_z)
    sm.threshold(im_tmp, my_min, min(255, my_min + delta), im_ground)
    # OLD:
    # sm.threshold(im_tmp, my_min, min(255, my_min + 5), im_ground)

    sm.sub(im_max, im_min, im_tmp)

    # put to zero all the non zero pixels in imGround
    sm.compare(im_tmp, ">", int(np.round(0.3 * res_z)), 0, im_ground, im_ground)
    # NEW:
    if proj.res_x > 1:
        # open with se_2x2
        se_2x2 = sm.StrElt(True, [0, 1, 2, 3])
        sm.open(im_ground, im_ground, se_2x2)

    # OLD:
    # se_2x2 = sm.StrElt(True, [0, 1, 2, 3])
    # sm.open(im_ground, im_ground, se_2x2)
    im_interp, im_dart, im_obj = im_dart_interp(points, proj, im_max, nl)

    # Lambda flat zones
    im_label = sm.Image(im_interp, "UINT32")
    sm.lambdaLabel(im_interp, my_lambda, im_label, nl)

    label_with_measure(im_label, im_ground, im_ground, "max", nl)

    # todo: parametrize the 3 value
    sm.compare(im_obj, ">", 3, 0, im_ground, im_ground)

    # empty pixels set to 0 again
    sm.compare(im_max, "==", 0, 0, im_ground, im_ground)

    # evaluate
    # conf_mat, conf_mat_norm = evaluate(im_class, im_ground, selectedId, condensedId)

    return im_ground, im_interp


def evaluate_2d_pred(im_class, im_pred, selected_id, condensed_id, classes, savedir, filename):
    """
    Auxiliary function to evaluate prediction in 2D

    Parameters
    ----------
    im_class: ndarray

    im_pred: ndarray

    selected_id: list

    condensed_id: list

    classes: list

    savedir: str

    filename: str
    """
    from smutsia.utils.scores import get_confusion_matrix, condense_confusion_matrix, normalize_confusion_matrix
    from smutsia.utils.viz import plot_confusion_matrix
    #  40: "road",  44: "parking",  48: "sidewalk",  49: "other-ground", 10: "car", 50: "building"
    sm.compare(im_pred, ">", 0, 40, 0, im_pred)

    np_gt = smil_2_np(im_class)
    np_pred = smil_2_np(im_pred)
    conf_mat1, conf_mat_norm1 = get_confusion_matrix(np_gt.flat, np_pred.flat, selected_id)

    conf_mat = condense_confusion_matrix(conf_mat1, selected_id, condensed_id)
    conf_mat = conf_mat.astype(int)
    conf_mat_norm = normalize_confusion_matrix(conf_mat)

    plot_confusion_matrix(conf_mat_norm, classes=classes, normalize=True,
                          savefig=os.path.join(savedir, filename + '.eps'),
                          title='Confusion Matrix ' + filename.upper())


def dart_ground_detection(cloud,
                          threshold,
                          delta_ground,
                          delta_h_circle,
                          res_x,
                          res_y,
                          res_z,
                          savedir='',
                          select_id=None,
                          condense_id=None,
                          classes=None):
    """
    Parameters
    ----------
    cloud: PyntCloud

    threshold: int

    delta_ground: float

    delta_h_circle: float

    res_x: float

    res_y: float

    res_z: float

    savedir: str
        path to which save plots of the method

    select_id: list
        id to use to compute 2D confusion matrix

    condense_id: list
        id to use to compute 2D confusion matrix

    classes: list
        list with corresponding names of classes
    """
    params = LambdaGDParameters(my_lambda=threshold, delta_ground=delta_ground, delta_h_circle=delta_h_circle)

    points = cloud.xyz
    labels = cloud.points.labels.values.astype(int)
    proj = Projection(proj_type='linear', res_x=res_x, res_y=res_y)
    # select the criteria to use to find min z
    # z_min = np.percentile(p_z, percent)
    # min_z = find_min_z(points[:, 2], 0.2, 5)
    min_z = find_min_z(points[:, 2], 0.2)
    im_min, im_max, im_acc, im_class = project_img(proj, points=points, labels=labels, res_z=res_z, min_z=min_z,
                                                   filter_outliers=True)

    im_ground, im_interp = ground_detection_min_circle(params=params,
                                                       points=points,
                                                       proj=proj,
                                                       res_z=res_z,
                                                       im_min=im_min,
                                                       im_max=im_max)

    if len(savedir) > 0:
        from smutsia.utils.viz import inspect_missclass
        fn = ''
        if hasattr(cloud, 'sequence'):
            fn += cloud.sequence + '_'
        if hasattr(cloud, 'filename'):
            fn += cloud.filename + '_'
        fn += '{}'

        sm.write(im_min, os.path.join(savedir, fn.format('min.png')))
        sm.write(im_max, os.path.join(savedir, fn.format('max.png')))
        sm.write(im_acc, os.path.join(savedir, fn.format('acc.png')))
        sm.write(im_class, os.path.join(savedir, fn.format('class.png')))
        bike_ids = [11, 15, 31, 32]
        car_ids = [10, 13, 18]
        build_id = [50]
        inspect_missclass(im_class, im_ground, bike_ids, im_max, im_min, savedir, fn.format('bikes'))
        inspect_missclass(im_class, im_ground, car_ids, im_max, im_min, savedir, fn.format('cars'))
        inspect_missclass(im_class, im_ground, build_id, im_max, im_min, savedir, fn.format('build'))

        if condense_id is not None and select_id is not None:
            # evaluate score on 2D image
            evaluate_2d_pred(im_class, im_ground,
                             selected_id=select_id, condensed_id=condense_id, classes=classes,
                             savedir=savedir, filename=fn.format('2D'))

    im_label = sm.Image(im_min, "UINT16")
    sm.lambdaLabel(im_min, 1, im_label, params.nl)
    im_ground2 = sm.Image(im_ground)
    label_with_measure(im_label, im_ground, im_ground2, "max", params.nl)
    im_delta = sm.Image(im_ground)
    sm.compare(im_ground, ">", 0, 4, 0, im_delta)
    sm.compare(im_ground2, ">", im_ground, 1, im_delta, im_delta)
    sm.copy(im_ground2, im_ground)

    ground = back_projection_ground(proj=proj, points=points, res_z=res_z, im_min=im_min, im_ground=im_ground,
                                    delta_ground=delta_ground, min_z=min_z, im_delta=im_delta)

    return ground


if __name__ == "__main__":
    from glob import glob
    from smutsia.utils.semantickitti import load_pyntcloud
    from smutsia.utils.viz import plot_cloud
    from definitions import SEMANTICKITTI_PATH
    basedir = os.path.join(SEMANTICKITTI_PATH, '08', 'velodyne')
    files = sorted(glob(os.path.join(basedir, '*.bin')))
    pc = load_pyntcloud(files[0], add_label=True)
    out = dart_ground_detection(cloud=pc,
                                threshold=2,
                                delta_ground=0.05,
                                delta_h_circle=0.5,
                                res_x=5,
                                res_y=5,
                                res_z=10)

    plot_cloud(pc.xyz, scalars=out, notebook=False)
    print("END")
