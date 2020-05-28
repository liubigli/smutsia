import os
import numpy as np
import smilPython as sm
from smutsia.point_cloud.projection import Projection, project_img, back_projection_ground
from smutsia.utils.image import smil_2_np, np_2_smil


class ClothParameters:
    def __init__(self, rigidness, local_rigidness, res_x, res_z, avg_mode, mass_mode):

        self.rigidness = rigidness
        self.local_rigidness = local_rigidness

        self.res_x = res_x
        self.res_y = res_x
        self.res_z = res_z

        self.ngb_list = [
            (-1, -1),
            (-1, +0),
            (-1, +1),
            (+0, -1),
            (+0, +1),
            (+1, -1),
            (+1, +0),
            (+1, +1),
            (+2, +0),
            (-2, +0),
            (+0, +2),
            (+0, -2)]

        self.time_step = 0.65
        self.gravity = 9.8 / self.res_z
        if mass_mode == "one":
            self.mass = 1
        elif mass_mode == "res_x":
            self.mass = 1 / (self.res_x * self.res_x)

        self.step = (self.gravity / self.mass) * self.time_step * self.time_step

        # Performance / precision parameters
        self.epsilon = 0.01
        self.max_iter = 200
        self.debug = False
        self.debug_im = False

        self.avg_mode = avg_mode

        self.deltaGround = 0.2


def apply_gravity(clothP, prevCloth, cloth, unmovable, updatedCloth):
    # apply gravity
    updatedCloth[:, :] = ((2 * cloth[:, :]) - prevCloth[:, :]) + clothP.step

    # unmovable pixels keep their previous value
    idx = np.where(unmovable != 0)
    updatedCloth[idx] = cloth[idx]


def intersection_check(npMin, updatedCloth, unmovable):
    idx = np.where(updatedCloth >= npMin)
    # nidx = np.where(updatedCloth<npMin)

    unmovable[idx] = 255

    updatedCloth[idx] = npMin[idx]


def average_blur(npIm, npMask, ngbList, avg_mode):
    myShape = npIm.shape
    im1 = np.zeros(myShape)
    tmpMask = np.zeros(myShape)
    imNb = np.zeros(myShape)
    imavg = np.zeros(myShape)
    w, h = myShape
    for deltax, deltay in ngbList:
        im1[:, :] = npIm[:, :]
        tmpMask[:, :] = npMask[:, :]
        if deltax >= 0:
            if deltay >= 0:
                im1[deltax:w, deltay:h] = npIm[0:w-deltax, 0:h-deltay]
                tmpMask[deltax:w, deltay:h] = npMask[0:w-deltax, 0:h-deltay]
            else:
                deltay =- deltay
                im1[deltax:w, 0:h-deltay] = npIm[0:w-deltax, deltay:h]
                tmpMask[deltax:w, 0:h-deltay] = npMask[0:w-deltax, deltay:h]
        else:
            deltax =- deltax
            if deltay >= 0:
                im1[0:w-deltax, deltay:h] = npIm[deltax:w, 0:h-deltay]
                tmpMask[0:w-deltax, deltay:h] = npMask[deltax:w, 0:h-deltay]
            else:
                deltay =- deltay
                im1[0:w-deltax, 0:h-deltay] = npIm[deltax:w, deltay:h]
                tmpMask[0:w-deltax, 0:h-deltay] = npMask[deltax:w, deltay:h]

        if avg_mode == "mask":  # only under mask (better results!)
            idx = np.where(tmpMask > 0)
            imavg[idx] = imavg[idx] + im1[idx]
            imNb[idx] = imNb[idx]+1
        elif avg_mode == "all":
            imavg = imavg + im1
            imNb = imNb+1

    idx = np.where(imNb > 0)
    imavg[idx] = imavg[idx] / imNb[idx]

    return imavg


def apply_internal_force(clothP, npMask, unmovable, updatedCloth):
    # BMI: unmovable NOT USED IN THIS FUNCTION!
    for i in range(clothP.rigidness):

        avg_ngb = average_blur(updatedCloth, npMask, clothP.ngb_list, clothP.avg_mode)

        # myShape = updatedCloth.shape
        # internal_force = np.zeros(myShape)
        internal_force = avg_ngb - updatedCloth
        internal_force = internal_force* clothP.local_rigidness

        # Move movable pixels
        idx = np.where(unmovable==0)  # ADDED BMI
        updatedCloth[idx] = updatedCloth[idx] + internal_force[idx]


def computeMaxMovement(cloth, updatedCloth, mask_image):
    myShape = cloth.shape
    abs_diff = np.zeros(myShape)
    abs_diff[:, :] = abs(cloth[:, :] - updatedCloth[:, :])

    idx = np.where(mask_image == 0)
    abs_diff[idx] = 0
    mymax = np.amax(abs_diff)
    return mymax


def cloth_simulation(clothP, npMin, npMask):
    w, h = npMin.shape
    cloth = np.zeros((w, h))
    prevCloth = np.zeros((w, h))
    updatedCloth = np.zeros((w, h))

    myShape = cloth.shape
    unmovable = np.zeros(myShape)

    max_movement = clothP.epsilon + 1
    epsilon = clothP.epsilon
    max_iter = clothP.max_iter

    iter = 0

    while max_movement > epsilon and iter < max_iter:

        apply_gravity(clothP, prevCloth, cloth, unmovable, updatedCloth)

        intersection_check(npMin, updatedCloth, unmovable)

        apply_internal_force(clothP, npMask, unmovable, updatedCloth)

        #    applyFriction(unmovable(), cloth(), updatedCloth())
        max_movement = computeMaxMovement(cloth, updatedCloth, npMask)
        if clothP.debug:
            print("MAXMOV============max_mouv", round(max_movement, 2))

        prevCloth[:, :] = cloth[:, :]
        cloth[:, :] = updatedCloth[:, :]
        iter += 1
    #  postProcessing(genP,unmovable, npMin, npMask, cloth)
    return updatedCloth


def cloth_simulation_filtering(cloud, res_x, res_z, avg_mode, mass_mode, rigidness, local_rigidness):
    clothP = ClothParameters(rigidness, local_rigidness, res_x, res_z, avg_mode, mass_mode)
    points = cloud.xyz
    labels = cloud.points.labels.values.astype(int)
    proj = Projection(proj_type='linear', res_x=res_x, res_y=res_x)

    # select the criteria to use to find min z
    min_z = np.percentile(cloud.xyz[:, 2], 0.5)

    im_min, im_max, im_acc, im_class = project_img(proj, points=points, labels=labels, res_z=res_z, min_z=min_z)
    imMask = sm.Image(im_min)
    sm.compare(im_min, "==", 255, 0, 255, imMask)
    npMin = smil_2_np(im_min)
    npMask = smil_2_np(imMask)

    updatedCloth = cloth_simulation(clothP, npMin, npMask)
    im_cloth = np_2_smil(updatedCloth)
    im_ground = sm.Image(im_cloth)
    sm.compare(im_cloth, ">", 0, 255, 0, im_ground)

    ground = back_projection_ground(proj=proj, points=points, res_z=res_z, im_min=im_cloth, im_ground=im_ground,
                                    delta_ground=clothP.deltaGround, min_z=min_z)

    return ground


if __name__ == "__main__":
    from glob import glob
    from smutsia.utils.semantickitti import load_pyntcloud
    from definitions import SEMANTICKITTI_PATH
    basedir = os.path.join(SEMANTICKITTI_PATH, '08', 'velodyne')
    files = sorted(glob(os.path.join(basedir, '*.bin')))
    pc = load_pyntcloud(files[0], add_label=True)
